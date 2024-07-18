import argparse
import pickle
from typing import Dict, List

import numpy as np
import torch

import tqdm

import visionloader as vl
from whitenoise import RandomNoiseFrameGenerator

from basis_functions.spatial_basis_functions import make_spatial_basis_from_hyperparameters
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import CroppedSTABoundingBox
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
from lib.dataset_specific_hyperparams.jitter_glm_hyperparameters import \
    CROPPED_JITTER_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE
from lib.dataset_specific_ttl_corrections.wn_ttl_structure import WhiteNoiseSynchroSection
from lib.dataset_specific_ttl_corrections.ttl_interval_constants import WN_FRAME_RATE, WN_N_FRAMES_PER_TRIGGER, \
    SAMPLE_RATE
from new_style_fit_multimovie_ct_glm import new_style_wn_regularized_precompute_convs_and_fit_glm
from optimization_encoder.separable_trial_glm import ProxSolverParameterGenerator
from optimization_encoder.trial_glm import FittedGLM, FittedGLMFamily
from optimization_encoder.ct_glm import fused_bernoulli_neg_ll_loss
from fit_multimovie_ct_glm import compute_full_filters
from lib.data_utils.dynamic_data_util import preload_bind_jittered_movie_patches_to_synchro, \
    preload_bind_wn_movie_patches

# FIXME eventually don't hardcode this
INCLUDED_BLOCKS_BY_NAME = {
    # these were for 2018-08-07-5
    'data009' : [1, 2, 3, 4, 6, 7, 8, 9],
    'data010' : [1, 2, 3, 4, 6, 7, 8, 9],

    ## these are for 2019-11-07-0
    #'data005': [1, 2, 3, 4, 6, 7, 8, 9],
    #'data006': [1, 2, 3, 4, 6, 7, 8, 9]

    ## these are for 2018-11-12-5
    'data005': [1, 2, 3, 4, 6, 7, 8, 9],
    'data004': [1, 2, 3, 4, 6, 7, 8, 9]
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Fit LNBRC (coupled) encoding models jointly to eye movements stimulus and white noise. No multiprocessing.')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('cell_type', type=str, help='particular cell type to fit')
    parser.add_argument('save_path', type=str, help='path to save grid search pickle file')
    parser.add_argument('-n', '--outer_iter', type=int, default=2,
                        help='number of outer coordinate descent iterations to use')
    parser.add_argument('-m', '--seconds_wn', type=int, default=300,
                        help='number of seconds of white noise data to use')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-s', '--stupid', action='store_true', default=False)
    parser.add_argument('-l', '--l1_sparse', type=float, default=1e-6,
                        help='hyperparameter for L1 spatial filter sparsity')
    parser.add_argument('-g', '--l21_couple', type=float, default=1e-5,
                        help='hyperparameter for L21 group sparsity for coupling filters')
    parser.add_argument('-w', '--wn_weight', type=float, default=1e-2,
                        help='Weight to place on the WN loss function')
    parser.add_argument('-j', '--jitter', type=float, default=0.0,
                        help='SD amount to jitter the spike times by, in samples; Default 0')
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
    nscenes_dataset_info_list = config_settings['NScenesDatasets']

    create_test_dataset = (dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]

    # calculate timing for the nscenes dataset
    loaded_synchronized_datasets = ddu.load_jittered_nscenes_dataset_and_timebin(
        nscenes_dataset_info_list,
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
    wn_rel_downsample = wn_frame_generator.stixel_height // (2 * nscenes_downsample_factor)


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
    # FIXME we need a new hyperparameter dict
    ref_ds_lookup_key = dcp.generate_lookup_key_from_dataset_info(ref_dataset_info)
    model_fit_hyperparameters = CROPPED_JITTER_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE[ref_ds_lookup_key][bin_width_time_ms]()[cell_type]

    cell_distances_by_type = model_fit_hyperparameters.neighboring_cell_dist
    timecourse_basis = model_fit_hyperparameters.timecourse_basis
    feedback_basis = model_fit_hyperparameters.feedback_basis
    coupling_basis = model_fit_hyperparameters.coupling_basis

    l1_spat_sparse = args.l1_sparse
    l21_group_sparse = args.l21_couple
    wn_model_weight = args.wn_weight
    n_iter_outer = args.outer_iter

    n_bins_binom = model_fit_hyperparameters.n_bins_binom
    use_binom_loss = (n_bins_binom is not None)

    (n_spatbasis_h, n_spatbasis_w), spat_basis_type = model_fit_hyperparameters.spatial_basis

    ###### Compute the STA timecourse, and use that as the initial guess for ###
    ###### the timecourse ######################################################
    # Load the timecourse initial guess
    with open(config_settings[dcp.OutputSection.INITIAL_GUESS_TIMECOURSE], 'rb') as pfile:
        timecourses_by_type = pickle.load(pfile)
        avg_timecourse = timecourses_by_type[cell_type]

    initial_timevector_guess = np.linalg.solve(timecourse_basis @ timecourse_basis.T,
                                               timecourse_basis @ avg_timecourse)

    ##########################################################################
    # Begin model fitting for each cell
    relevant_cell_ids = cells_ordered.get_reference_cell_order(cell_type)

    fitted_models_dict = {}  # type: Dict[int, FittedGLM]

    model_fit_pbar = tqdm.tqdm(total=len(relevant_cell_ids))
    for idx, cell_id in enumerate(relevant_cell_ids):
        ### Get neighboring cells ######################################

        print(f"idx {idx}, cell_id {cell_id}")

        coupled_cells_subset = {}  # type: Dict[str, List[int]]
        all_coupled_cell_ids_ordered = [] # type: List[int]
        for coupled_cell_type in ct_order:
            max_coupled_distance_typed = cell_distances_by_type[coupled_cell_type]
            interaction_edges = pairwise_interactions.query_cell_interaction_edges(cell_id, coupled_cell_type)
            coupled_cell_ids = [x.dest_cell_id for x in interaction_edges if
                                x.additional_attributes['distance'] < max_coupled_distance_typed]
            coupled_cells_subset[coupled_cell_type] = coupled_cell_ids
            all_coupled_cell_ids_ordered.extend(coupled_cell_ids)

        ### Get cropping objects ########################################
        cell_ix = relev_cell_ids.index(cell_id)
        crop_bbox = bounding_boxes_by_type[cell_type][cell_ix] # type: CroppedSTABoundingBox
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
            INCLUDED_BLOCKS_BY_NAME,
            (crop_bounds_h, crop_bounds_w),
            ddu.PartitionType.TRAIN_PARTITION
        )

        training_wn_movie_blocks = preload_bind_wn_movie_patches(
            loaded_wn_movie,
            [(0, int(WN_FRAME_RATE * args.seconds_wn // wn_interval)), ],
            (sta_cropping_slice_h, sta_cropping_slice_w)
        )

        if args.debug:
            solver_schedule = (ProxSolverParameterGenerator(500, 250), ProxSolverParameterGenerator(500, 250))
        elif args.stupid:
            solver_schedule = (ProxSolverParameterGenerator(25, 25), ProxSolverParameterGenerator(25, 25))
        else:
            solver_schedule = (ProxSolverParameterGenerator(1000, 250), ProxSolverParameterGenerator(1000, 250))

        optim_fn = new_style_wn_regularized_precompute_convs_and_fit_glm

        loss, glm_fit_params = optim_fn(
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
            l21_group_sparse,  # group sparsity
            l1_spat_sparse,  # spatial sparsity
            2,
            1e-5,
            solver_schedule,
            device,
            jitter_spike_times=args.jitter,
            initial_guess_timecourse=initial_timevector_guess[None, :],
            trim_spikes_seq=coupling_basis.shape[1] - 1,
            log_verbose_ascent=True,
            movie_spike_dtype=torch.float16
        )

        coupling_w, feedback_w, spat_w, timecourse_w, bias = glm_fit_params

        spat_filters, time_filters, feedback_td, coupling_td = compute_full_filters(
            spat_w,
            spat_spline_basis.T,
            timecourse_w,
            timecourse_basis,
            feedback_w,
            feedback_basis,
            coupling_w,
            coupling_basis,
        )

        spat_filt_as_image_shape = spat_filters.reshape(hhh, www)

        fitting_params_to_store = {
            'l1_spat_sparse' : l1_spat_sparse,
            'l21_group_sparse' : l21_group_sparse,
            'wn_weight' : wn_model_weight,
            'spike_time_jitter': args.jitter
        }

        model_summary_parameters = FittedGLM(cell_id, spat_filt_as_image_shape, bias, timecourse_w, feedback_w,
                                             (coupling_w, np.array(all_coupled_cell_ids_ordered, dtype=np.int64)),
                                             fitting_params_to_store, loss, None)

        fitted_models_dict[cell_id] = model_summary_parameters

        model_fit_pbar.update(1)

    model_fit_pbar.close()

    fitted_model_family = FittedGLMFamily(
        fitted_models_dict,
        model_fit_hyperparameters.spatial_basis,
        timecourse_basis,
        feedback_basis,
        coupling_basis
    )

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(fitted_model_family, pfile)
