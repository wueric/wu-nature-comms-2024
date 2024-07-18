import argparse
import itertools
import pickle
from typing import Dict, List, Tuple, Callable

import numpy as np
import torch
import tqdm

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from basis_functions.spatial_basis_functions import make_spatial_basis_from_hyperparameters
from fit_multimovie_ct_glm import compute_full_filters
from glm_precompute.ct_glm_precompute import repeats_blocks_extract_center_spikes, \
    repeats_blocks_extract_coupling_spikes, \
    repeats_blocks_precompute_spatial_filter_apply
from lib.data_utils.dynamic_data_util import RepeatsJitteredMovieDataloader, ShuffledRepeatsJitteredMovieDataloader
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import CroppedSTABoundingBox
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_specific_hyperparams.jitter_glm_hyperparameters import \
    CROPPED_JITTER_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE
from new_style_fit_multimovie_ct_glm import new_style_repeats_precompute_convs_and_fit_glm
from optimization_encoder.ct_glm import fused_bernoulli_neg_ll_loss
from optimization_encoder.multimovie_ct_glm import MM_PreappliedSpatialSingleCellEncodingLoss
from optimization_encoder.separable_trial_glm import ProxSolverParameterGenerator
from optimization_encoder.trial_glm import FittedGLM

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


def evaluate_repeats_glm_model_fit(
        repeats_test_blocks: List[ddu.RepeatBrownianTrainingBlock],
        image_transform_torch: Callable[[torch.Tensor], torch.Tensor],
        center_cell_wn: Tuple[str, int],
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        stimulus_cropping_bounds: Tuple[Tuple[int, int], Tuple[int, int]],
        spat_filt: np.ndarray,
        timecourse_td: np.ndarray,
        feedback_td: np.ndarray,
        coupling_td: np.ndarray,
        bias: np.ndarray,
        loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device) -> float:
    ns_frame_sel_and_overlaps_all = ddu.repeat_training_compute_interval_overlaps(repeats_test_blocks)

    ns_center_spikes_torch = repeats_blocks_extract_center_spikes(
        repeats_test_blocks,
        center_cell_wn,
        cells_ordered,
        device
    )

    ns_coupled_spikes = repeats_blocks_extract_coupling_spikes(
        repeats_test_blocks,
        coupled_cells,
        cells_ordered,
        device
    )

    ns_precomputed_spatial_convolutions = repeats_blocks_precompute_spatial_filter_apply(
        repeats_test_blocks,
        ns_frame_sel_and_overlaps_all,
        spat_filt,
        stimulus_cropping_bounds,
        image_transform_torch,
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

    loss_per_block = []
    for (precomputed_spatial_conv, center_spike, coupling_spike) in \
            zip(ns_precomputed_spatial_convolutions, ns_center_spikes_torch, ns_coupled_spikes):
        total_loss_torch = evaluation_model([precomputed_spatial_conv, ],
                                            [center_spike[None, :], ],
                                            [coupling_spike, ]).item()

        loss_per_block.append(total_loss_torch)

    # clean up GPU explicitly
    del ns_precomputed_spatial_convolutions, ns_center_spikes_torch, ns_coupled_spikes
    del evaluation_model

    return np.mean(loss_per_block)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Grid search for hyperparameters for the repeats-only eye movements nscenes GLM fit')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('cell_type', type=str, help='particular cell type to fit')
    parser.add_argument('save_path', type=str, help='path to save grid search pickle file')
    parser.add_argument('-n', '--outer_iter', type=int, default=2,
                        help='number of outer coordinate descent iterations to use')
    parser.add_argument('-t', '--test_partition', type=int, default=10,
                        help='Number of stimuli in the test partition, counted from the end')
    parser.add_argument('-sh', '--shuffle', action='store_true', default=False,
                        help='Train on shuffled repeats')
    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    reference_piece_path = config_settings['ReferenceDataset'].path

    cell_type = args.cell_type
    ref_dataset_info = config_settings[dcp.ReferenceDatasetSection.OUTPUT_KEY]  # type: dcp.DatasetInfo

    piece_name, ref_datarun_name = dcp.awsify_piece_name_and_datarun_lookup_key(
        ref_dataset_info.path,
        ref_dataset_info.name)

    ################################################################
    # Load the cell types and matching
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    relevant_cell_ids = cells_ordered.get_reference_cell_order(cell_type)

    ####### Load the previously identified interactions #########################
    with open(config_settings['featurized_interactions_ordered'], 'rb') as picklefile:
        pairwise_interactions = pickle.load(picklefile)  # type: InteractionGraph

    ################################################################
    # Load some of the model fit parameters
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]
    bin_width_time_ms = samples_per_bin // 20

    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)
        blurred_stas_by_type = pickle.load(pfile)

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda_torch = du.make_torch_image_transform_lambda(image_rescale_low, image_rescale_high)

    #################################################################
    # Load the raw data
    create_test_dataset = (dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]

    nscenes_dataset_info_list = config_settings[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]

    loaded_synchronized_datasets = ddu.load_jittered_nscenes_dataset_and_timebin(
        nscenes_dataset_info_list,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
    )

    # FIXME see if we can get a better way to do this
    height, width = 160, (320 - crop_width_low - crop_width_high)

    # grab the repeat frames, bin spikes, etc.
    # construct the training blocks
    if args.shuffle:

        shuffle_jitter_dataloader = ShuffledRepeatsJitteredMovieDataloader(
            loaded_synchronized_datasets,
            cells_ordered,
            samples_per_bin,
            crop_w_ix=(crop_width_low, 320 - crop_width_low),  # FIXME
            image_rescale_lambda=None,
            n_shuffle_at_a_time=args.shuffle
        )

        n_stimuli = len(shuffle_jitter_dataloader)
        max_train_stimuli = n_stimuli - args.test_partition

        repeat_training_blocks = ddu.construct_repeat_training_blocks_from_shuffled_repeats_dataloader(
            shuffle_jitter_dataloader,
            list(np.r_[0:max_train_stimuli]))

        repeat_test_blocks = ddu.construct_repeat_training_blocks_from_shuffled_repeats_dataloader(
            shuffle_jitter_dataloader,
            list(np.r_[max_train_stimuli:n_stimuli])
        )

    else:
        repeat_jitter_dataloader = RepeatsJitteredMovieDataloader(
            loaded_synchronized_datasets,
            cells_ordered,
            samples_per_bin,
            crop_w_ix=(crop_width_low, 320 - crop_width_low),  # FIXME
            image_rescale_lambda=None,
        )

        n_stimuli = len(repeat_jitter_dataloader)
        max_train_stimuli = n_stimuli - args.test_partition

        repeat_training_blocks = ddu.construct_repeat_training_blocks_from_data_repeats_dataloader(
            repeat_jitter_dataloader,
            list(np.r_[0:max_train_stimuli]))

        repeat_test_blocks = ddu.construct_repeat_training_blocks_from_data_repeats_dataloader(
            repeat_jitter_dataloader,
            list(np.r_[max_train_stimuli:n_stimuli])
        )

    ref_ds_lookup_key = dcp.generate_lookup_key_from_dataset_info(ref_dataset_info)
    piece_name, ref_dataset_name = ref_ds_lookup_key
    model_fit_hyperparameters = \
        CROPPED_JITTER_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE[ref_ds_lookup_key][bin_width_time_ms]()[cell_type]

    cell_distances_by_type = model_fit_hyperparameters.neighboring_cell_dist
    timecourse_basis = model_fit_hyperparameters.timecourse_basis
    feedback_basis = model_fit_hyperparameters.feedback_basis
    coupling_basis = model_fit_hyperparameters.coupling_basis

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

        hhh = crop_bounds_h[1] - crop_bounds_h[0]
        www = crop_bounds_w[1] - crop_bounds_w[0]

        spat_spline_basis = make_spatial_basis_from_hyperparameters(crop_bounds_h, crop_bounds_w,
                                                                    n_spatbasis_h, n_spatbasis_w,
                                                                    spat_basis_type)

        spat_spline_basis_imshape = spat_spline_basis.T.reshape(-1, hhh, www)
        hyperparams_pbar = tqdm.tqdm(total=L1_GRID.shape[0] * L21_GRID.shape[0])
        for ix, grid_params in enumerated_product(L1_GRID, L21_GRID):
            l1_ix, l21_ix = ix
            l1_hyperparam, l21_hyperparam = grid_params

            solver_schedule = (ProxSolverParameterGenerator(1000, 250), ProxSolverParameterGenerator(250, 250))

            loss, glm_fit_params = new_style_repeats_precompute_convs_and_fit_glm(
                repeat_training_blocks,
                spat_spline_basis_imshape,
                timecourse_basis,
                feedback_basis,
                coupling_basis,
                (cell_type, cell_id),
                coupled_cells_subset,
                cells_ordered,
                (crop_bounds_h, crop_bounds_w),
                image_rescale_lambda_torch,
                fused_bernoulli_neg_ll_loss,
                l21_hyperparam,
                l1_hyperparam,
                args.outer_iter,
                1e-9,
                solver_schedule,
                device,
                initial_guess_timecourse=initial_timevector_guess[None, :],
                trim_spikes_seq=feedback_basis.shape[1] - 1,
                log_verbose_ascent=True,
                movie_spike_dtype=torch.float16,
            )

            coupling_w, feedback_w, spat_w, timecourse_w, bias = glm_fit_params

            spat_filters, timecourse_td, feedback_td, coupling_td = compute_full_filters(
                spat_w,
                spat_spline_basis_imshape.reshape(-1, hhh * www),
                timecourse_w,
                timecourse_basis,
                feedback_w,
                feedback_basis,
                coupling_w,
                coupling_basis,
            )

            mean_test_loss = evaluate_repeats_glm_model_fit(
                repeat_test_blocks,
                image_rescale_lambda_torch,
                (cell_type, cell_id),
                coupled_cells_subset,
                cells_ordered,
                (crop_bounds_h, crop_bounds_w),
                spat_filters.reshape(-1),
                timecourse_td.squeeze(0),
                feedback_td,
                coupling_td,
                bias,
                fused_bernoulli_neg_ll_loss,
                device
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
            'coord_desc_outer_iter': args.outer_iter
        }

        pickle.dump(pickle_obj, pfile)
