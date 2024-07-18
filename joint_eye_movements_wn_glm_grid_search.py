import argparse
import itertools
import pickle
from collections import namedtuple
from typing import Dict, List, Tuple, Callable

import numpy as np
import torch
import tqdm
import visionloader as vl
from whitenoise import RandomNoiseFrameGenerator

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from basis_functions.spatial_basis_functions import make_spatial_basis_from_hyperparameters
from fit_multimovie_ct_glm import compute_full_filters
from glm_precompute.ct_glm_precompute import multidata_precompute_spat_filter_apply, multidata_bin_center_spikes, \
    multidata_bin_coupling_spikes
from lib.data_utils.dynamic_data_util import preload_bind_jittered_movie_patches_to_synchro, \
    preload_bind_wn_movie_patches
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import CroppedSTABoundingBox
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_specific_hyperparams.jitter_glm_hyperparameters import \
    CROPPED_JITTER_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE
from lib.dataset_specific_ttl_corrections.ttl_interval_constants import WN_FRAME_RATE, WN_N_FRAMES_PER_TRIGGER, \
    SAMPLE_RATE
from lib.dataset_specific_ttl_corrections.wn_ttl_structure import WhiteNoiseSynchroSection
from new_style_fit_multimovie_ct_glm import new_style_wn_regularized_precompute_convs_and_fit_glm
from optimization_encoder.ct_glm import fused_bernoulli_neg_ll_loss
from optimization_encoder.multimovie_ct_glm import MM_PreappliedSpatialSingleCellEncodingLoss
from optimization_encoder.separable_trial_glm import ProxSolverParameterGenerator
from optimization_encoder.trial_glm import FittedGLM

INCLUDED_BLOCKS_BY_NAME = {
    # these were for 2018-08-07-5
    '2018-08-07-5': {
        'data009': [1, 2, 3, 4, 6, 7, 8, 9],
        'data010': [1, 2, 3, 4, 6, 7, 8, 9],
    },

    # these are for 2019-11-07-0
    '2019-11-07-0': {
        'data005': [1, 2, 3, 4, 6, 7, 8, 9],
        'data006': [1, 2, 3, 4, 6, 7, 8, 9],
    },

    # these are for 2018-11-12-5
    '2018-11-12-5': {
        'data004': [1, 2, 3, 4, 6, 7, 8, 9],
        'data005': [1, 2, 3, 4, 6, 7, 8, 9],
    }
}

INCLUDED_TEST_BLOCKS_BY_NAME = {
    # these were for 2018-08-07-5
    '2018-08-07-5': {
        'data009': [0, 5],
        'data010': [0, 5],
    },

    # these are for 2019-11-07-0
    '2019-11-07-0': {
        'data005': [0, 5],
        'data006': [0, 5],
    },

    # these are for 2018-11-12-5
    '2018-11-12-5': {
        'data004': [0, 5],
        'data005': [0, 5],
    }
}

# We only use a small number of good manually identified cells for the grid
# search. This is to improve the computational efficiency of the grid search

KNOWN_GOOD_CELLS = {
    '2018-08-07-5': {
        'ON parasol': [1046, 100, 806, 678],
        'OFF parasol': [1092, 1130, 772, 66],
        'ON midget': [1026, 379, 695, 857],
        'OFF midget': [8, 687, 777, 1147],
    },
    '2019-11-07-0': {
        'ON parasol': [395, 423, 1337, 1008],
        'OFF parasol': [787, 633, 1061, 954],
        'ON midget': [1171, 412, 1278, 478],
        'OFF midget': [781, 824, 1222, 1253],
    },
    '2018-11-12-5': {
        'ON parasol': [312, 634, 1122],
        'OFF parasol': [933, 358, 1027],
        'ON midget': [1, 957, 326, ],
        'OFF midget': [623, 43, 1007, ],
    }
}

L1_GRID = np.logspace(np.log10(1e-8), np.log10(1e-5), 4)
# L21_GRID = np.logspace(np.log10(1e-8), np.log10(1e-5), 4)
L21_GRID = np.logspace(np.log10(1e-4), np.log10(1e-0), 5)
# L21_GRID = np.array([1e-4, ])


def enumerated_product(*args):
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))


def evaluate_eye_movement_glm_model_fit(
        jittered_movie_blocks: List[ddu.LoadedBrownianMovieBlock],
        image_transform: Callable[[torch.Tensor], torch.Tensor],
        center_cell_wn: Tuple[str, int],
        cells_ordered: OrderedMatchedCellsStruct,
        coupled_cells: Dict[str, List[int]],
        spat_filt: np.ndarray,
        timecourse_td: np.ndarray,
        feedback_td: np.ndarray,
        coupling_td: np.ndarray,
        bias: np.ndarray,
        loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        bin_width_samples: int,
        device: torch.device,
        jitter_spike_times: float = 0.0) -> float:
    # Step 0: unpack everything that was a Dict into a List
    # Also bin spikes, and compute frame overlaps in the same loop
    # to make sure that the ordering of the lists doesn't get messed up
    timebins_all = ddu.multimovie_construct_natural_movies_timebins(jittered_movie_blocks,
                                                                    bin_width_samples)

    frame_sel_and_overlaps_all = ddu.multimovie_compute_interval_overlaps(jittered_movie_blocks,
                                                                          timebins_all)

    center_spikes_all_torch = multidata_bin_center_spikes(
        jittered_movie_blocks, timebins_all,
        center_cell_wn, cells_ordered,
        device,
        jitter_spike_times=jitter_spike_times
    )

    coupling_spikes_all_torch = multidata_bin_coupling_spikes(
        jittered_movie_blocks, timebins_all,
        coupled_cells, cells_ordered,
        device,
        jitter_spike_times=jitter_spike_times
    )

    # pre-compute the application of the spatial basis
    # with the fixed timecourse on the upsampled movie
    # and store the result on GPU
    precomputed_spatial_convolutions = multidata_precompute_spat_filter_apply(
        jittered_movie_blocks,
        frame_sel_and_overlaps_all,
        spat_filt,
        image_transform,
        device
    )

    # now construct the evaluation model
    evaluation_model = MM_PreappliedSpatialSingleCellEncodingLoss(
        timecourse_td,
        feedback_td,
        coupling_td,
        bias,
        loss_callable,
        dtype=torch.float32
    ).to(device)

    # evaluate the loss separately for each block,
    # since we substantial differences between blocks is a
    # useful signal for determining whether the cell matching is busted
    loss_per_block = []
    for (precomputed_spatial_conv, center_spike, coupling_spike) in \
            zip(precomputed_spatial_convolutions, center_spikes_all_torch, coupling_spikes_all_torch):
        total_loss_torch = evaluation_model([precomputed_spatial_conv, ],
                                            [center_spike, ],
                                            [coupling_spike, ]).item()

        loss_per_block.append(total_loss_torch)

    # clean up GPU explicitly
    del precomputed_spatial_convolutions, center_spikes_all_torch
    del coupling_spikes_all_torch

    return np.mean(loss_per_block)


Joint_WN_Jitter_GLM_Hyperparams = namedtuple('Joint_WN_Jitter_GLM_Hyperparams',
                                             ['l1_sparse', 'l21_group_sparse', 'wn_weight'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Grid search for hyperparameters for the joint WN-flashed nscenes LNBRC (coupled) fit')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('cell_type', type=str, help='particular cell type to fit')
    parser.add_argument('save_path', type=str, help='path to save grid search pickle file')
    parser.add_argument('-n', '--outer_iter', type=int, default=2,
                        help='number of outer coordinate descent iterations to use')
    parser.add_argument('-m', '--seconds_wn', type=int, default=300,
                        help='number of seconds of white noise data to use')
    parser.add_argument('-j', '--spike_time_jitter', type=float, default=0.0,
                        help='samples Gaussian SD to jitter spike times by')
    parser.add_argument('-w', '--wn_weight', type=float, default=1e-2,
                        help='Weight to place on the WN loss function')
    args = parser.parse_args()

    seconds_white_noise = args.seconds_wn
    cell_type = args.cell_type

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    #### Now load the natural scenes dataset #################################
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)
    image_rescale_lambda_torch = du.make_torch_image_transform_lambda(image_rescale_low, image_rescale_high)

    n_bins_before = config_settings[dcp.TimebinningSection.NBINS_BEFORE_TRANS]
    n_bins_after = config_settings[dcp.TimebinningSection.NBINS_AFTER_TRANS]
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]
    bin_width_time_ms = samples_per_bin // 20

    ################################################################
    # Load the natural scenes Vision datasets and determine what the
    # train and test partitions are
    jittered_nscenes_dataset_info_list = config_settings[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]

    create_jittered_test_dataset = (dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_jittered_heldout_dataset = (dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_jittered_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_jittered_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]

    # calculate timing for the nscenes dataset
    loaded_synchronized_datasets = ddu.load_jittered_nscenes_dataset_and_timebin(
        jittered_nscenes_dataset_info_list,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
    )

    ###################################################################
    # Load the white noise reference dataset since we are joint fitting
    ref_dataset_info = config_settings['ReferenceDataset'] # type: dcp.DatasetInfo
    reference_dataset = vl.load_vision_data(ref_dataset_info.path,
                                            ref_dataset_info.name,
                                            include_sta=True,
                                            include_neurons=True,
                                            include_params=True)
    ttl_times_whitenoise = reference_dataset.get_ttl_times()

    wn_frame_generator = RandomNoiseFrameGenerator.construct_from_xml(ref_dataset_info.wn_xml)
    wn_interval = wn_frame_generator.refresh_interval

    wn_synchro_block = WhiteNoiseSynchroSection(
        wn_frame_generator,
        ttl_times_whitenoise,
        WN_N_FRAMES_PER_TRIGGER // wn_interval,
        SAMPLE_RATE)

    loaded_wn_movie = ddu.LoadedWNMovies(ref_dataset_info.path,
                                         ref_dataset_info.name,
                                         reference_dataset,
                                         wn_synchro_block)

    ##########################################################################
    # Load precomputed cell matches, crops, etc.
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)
        blurred_stas_by_type = pickle.load(pfile)
        uncropped_window_sel = pickle.load(pfile)

    ####### Load the previously identified interactions #########################
    with open(config_settings['featurized_interactions_ordered'], 'rb') as picklefile:
        pairwise_interactions = pickle.load(picklefile)  # type: InteractionGraph

    ###### Get model hyperparameters ###########################################
    ref_ds_lookup_key = dcp.generate_lookup_key_from_dataset_info(ref_dataset_info)
    piece_name, ref_dataset_name = ref_ds_lookup_key
    model_fit_hyperparameters = \
        CROPPED_JITTER_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE[ref_ds_lookup_key][bin_width_time_ms]()[cell_type]

    cell_distances_by_type = model_fit_hyperparameters.neighboring_cell_dist
    timecourse_basis = model_fit_hyperparameters.timecourse_basis
    feedback_basis = model_fit_hyperparameters.feedback_basis
    coupling_basis = model_fit_hyperparameters.coupling_basis

    wn_rel_downsample = wn_frame_generator.stixel_height // (2 * nscenes_downsample_factor)
    (n_spatbasis_h, n_spatbasis_w), spat_basis_type = model_fit_hyperparameters.spatial_basis

    ###### Compute the STA timecourse, and use that as the initial guess for ###
    ###### the timecourse ######################################################
    # Load the timecourse initial guess
    relev_cell_ids = cells_ordered.get_reference_cell_order(cell_type)
    with open(config_settings[dcp.OutputSection.INITIAL_GUESS_TIMECOURSE], 'rb') as pfile:
        timecourses_by_type = pickle.load(pfile)
        avg_timecourse = timecourses_by_type[cell_type]

    if samples_per_bin > 20:
        sta_downsample_factor = samples_per_bin // 20
        avg_timecourse = avg_timecourse[::sta_downsample_factor]

    initial_timevector_guess = np.linalg.solve(timecourse_basis @ timecourse_basis.T,
                                               timecourse_basis @ avg_timecourse)

    ##########################################################################
    # Do the fitting for each cell
    relevant_ref_cell_ids = KNOWN_GOOD_CELLS[piece_name][cell_type]
    n_cells = len(relevant_ref_cell_ids)

    fitted_models_dict = {}
    test_loss_matrix = np.zeros((n_cells, L1_GRID.shape[0], L21_GRID.shape[0]))
    train_loss_matrix = np.zeros((n_cells, L1_GRID.shape[0], L21_GRID.shape[0]))
    train_set_loss_matrix = np.zeros((n_cells, L1_GRID.shape[0], L21_GRID.shape[0]))

    model_fit_pbar = tqdm.tqdm(total=n_cells)
    for idx, cell_id in enumerate(relevant_ref_cell_ids):

        print(f"idx {idx}, cell_id {cell_id}")

        coupled_cells_subset = {}  # type: Dict[str, List[int]]
        all_coupled_cell_ids_ordered = []  # type: List[int]
        for coupled_cell_type in ct_order:
            max_coupled_distance_typed = cell_distances_by_type[coupled_cell_type]
            interaction_edges = pairwise_interactions.query_cell_interaction_edges(cell_id, coupled_cell_type)
            coupled_cell_ids = [x.dest_cell_id for x in interaction_edges if
                                x.additional_attributes['distance'] < max_coupled_distance_typed]
            coupled_cells_subset[coupled_cell_type] = coupled_cell_ids
            all_coupled_cell_ids_ordered.extend(coupled_cell_ids)

        ### Get cropping objects ########################################
        cell_ix = relev_cell_ids.index(cell_id)
        crop_bbox = bounding_boxes_by_type[cell_type][cell_ix]  # type: CroppedSTABoundingBox
        crop_bounds_h, crop_bounds_w = crop_bbox.make_cropping_sliceobj(
            crop_hlow=crop_height_low,
            crop_hhigh=crop_height_high,
            crop_wlow=crop_width_low,
            crop_whigh=crop_width_high,
            downsample_factor=nscenes_downsample_factor,
            return_bounds=True
        )

        sta_cropping_slice_h, sta_cropping_slice_w = crop_bbox.make_cropping_sliceobj(
            crop_hlow=crop_height_low,
            crop_hhigh=crop_height_high,
            crop_wlow=crop_width_low,
            crop_whigh=crop_width_high,
            downsample_factor=wn_rel_downsample * nscenes_downsample_factor
        )

        hhh = crop_bounds_h[1] - crop_bounds_h[0]
        www = crop_bounds_w[1] - crop_bounds_w[0]

        spat_spline_basis = make_spatial_basis_from_hyperparameters(crop_bounds_h, crop_bounds_w,
                                                                    n_spatbasis_h, n_spatbasis_w,
                                                                    spat_basis_type)

        spat_spline_basis_imshape = spat_spline_basis.T.reshape(-1, hhh, www)

        ##########################################################################
        # Load the frames
        training_movie_blocks = preload_bind_jittered_movie_patches_to_synchro(
            loaded_synchronized_datasets,
            INCLUDED_BLOCKS_BY_NAME[piece_name],
            (crop_bounds_h, crop_bounds_w),
            ddu.PartitionType.TRAIN_PARTITION
        )

        training_wn_movie_blocks = preload_bind_wn_movie_patches(
            loaded_wn_movie,
            [(0, int(WN_FRAME_RATE * args.seconds_wn // wn_interval)), ],
            (sta_cropping_slice_h, sta_cropping_slice_w)
        )

        test_movie_blocks = preload_bind_jittered_movie_patches_to_synchro(
            loaded_synchronized_datasets,
            INCLUDED_TEST_BLOCKS_BY_NAME[piece_name],
            (crop_bounds_h, crop_bounds_w),
            ddu.PartitionType.TEST_PARTITION
        )

        hyperparams_pbar = tqdm.tqdm(total=L1_GRID.shape[0] * L21_GRID.shape[0])
        for ix, grid_params in enumerated_product(L1_GRID, L21_GRID):
            l1_ix, l21_ix = ix
            l1_hyperparam, l21_hyperparam = grid_params

            solver_schedule = (ProxSolverParameterGenerator(1000, 250), ProxSolverParameterGenerator(250, 250))

            loss, glm_fit_params = new_style_wn_regularized_precompute_convs_and_fit_glm(
                training_movie_blocks,
                training_wn_movie_blocks,
                args.wn_weight,
                spat_spline_basis.T.reshape(-1, hhh, www),
                timecourse_basis,
                feedback_basis,
                coupling_basis,
                (cell_type, cell_id),
                coupled_cells_subset,
                cells_ordered,
                samples_per_bin,
                image_rescale_lambda_torch,
                fused_bernoulli_neg_ll_loss,
                l21_hyperparam,  # group sparsity
                l1_hyperparam,  # spatial sparsity
                args.outer_iter,
                1e-5,
                solver_schedule,
                device,
                jitter_spike_times=args.spike_time_jitter,
                initial_guess_timecourse=initial_timevector_guess[None, :],
                trim_spikes_seq=coupling_basis.shape[1] - 1,
                log_verbose_ascent=True,
                movie_spike_dtype=torch.float16
            )

            coupling_w, feedback_w, spat_w, timecourse_w, bias = glm_fit_params

            spat_filters, timecourse_td, feedback_td, coupling_td = compute_full_filters(
                spat_w,
                spat_spline_basis.T,
                timecourse_w,
                timecourse_basis,
                feedback_w,
                feedback_basis,
                coupling_w,
                coupling_basis,
            )

            mean_test_loss = evaluate_eye_movement_glm_model_fit(
                test_movie_blocks,
                image_rescale_lambda_torch,
                (cell_type, cell_id),
                cells_ordered,
                coupled_cells_subset,
                spat_filters.reshape(-1),
                timecourse_td.squeeze(0),
                feedback_td,
                coupling_td,
                bias,
                fused_bernoulli_neg_ll_loss,
                samples_per_bin,
                device,
                jitter_spike_times=args.spike_time_jitter
            )

            train_set_loss_matrix[idx, l1_ix, l21_ix] = loss
            test_loss_matrix[idx, l1_ix, l21_ix] = mean_test_loss

            additional_fit_params = {
                'fista_schedule': (ProxSolverParameterGenerator(1000, 250), ProxSolverParameterGenerator(250, 250)),
                'outer_iter_num': args.outer_iter
            }

            model_summary_parameters = FittedGLM(cell_id, spat_w, bias, timecourse_w, feedback_w,
                                                 (coupling_w, np.array(all_coupled_cell_ids_ordered)),
                                                 additional_fit_params, loss, None)

            if (l1_hyperparam, l21_hyperparam) not in fitted_models_dict:
                fitted_models_dict[(l1_hyperparam, l21_hyperparam)] = {}

            fitted_models_dict[(l1_hyperparam, l21_hyperparam)][cell_id] = model_summary_parameters

            hyperparams_pbar.update(1)

        hyperparams_pbar.close()

        model_fit_pbar.update(1)

        torch.cuda.empty_cache()

    model_fit_pbar.close()

    with open(args.save_path, 'wb') as pfile:
        pickle_obj = {
            'models': fitted_models_dict,
            'L1': L1_GRID,
            'L21': L21_GRID,
            'test_loss': test_loss_matrix,
            'train_loss': train_set_loss_matrix,
            'train_opt_loss': train_loss_matrix,
            'timecourse_basis': timecourse_basis,
            'feedback_basis': feedback_basis,
            'coupling_basis': coupling_basis,
            'distances': cell_distances_by_type,
            'spike_time_jitter': args.spike_time_jitter,
            'coord_desc_outer_iter': args.outer_iter
        }

        pickle.dump(pickle_obj, pfile)
