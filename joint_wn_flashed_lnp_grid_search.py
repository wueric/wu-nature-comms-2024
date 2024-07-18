import argparse
import pickle
from collections import namedtuple
from typing import List

import numpy as np
import torch
import tqdm
import visionloader as vl
from torch import autocast
from whitenoise import RandomNoiseFrameGenerator

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from basis_functions.spatial_basis_functions import make_spatial_basis_from_hyperparameters
from glm_precompute.flashed_glm_precompute import flashed_ns_precompute_spatial_basis, \
    flashed_ns_bin_spikes_precompute_timecourse_basis_conv2, \
    flashed_ns_bin_spikes_only
from glm_precompute.wn_glm_precompute import preapply_spatial_basis_to_wn
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.dataset_specific_hyperparams.jitter_lnp_hyperparameters import \
    CROPPED_FLASHED_WN_LNP_HYPERPARAMETERS_FN_BY_PIECE
from lib.dataset_specific_ttl_corrections.ttl_interval_constants import WN_FRAME_RATE, WN_N_FRAMES_PER_TRIGGER, \
    SAMPLE_RATE
from lib.dataset_specific_ttl_corrections.wn_ttl_structure import WhiteNoiseSynchroSection
from lnp_precompute.wn_lnp_precompute import frame_rate_wn_lnp_multidata_bin_spikes
from new_style_optim_encoder.separable_trial_lnp import new_style_LNP_joint_wn_flashed_ns_alternating_optim, \
    Flashed_PreappliedSpatialSingleCellLNPLoss
from optimization_encoder.ct_glm import fused_poisson_spiking_neg_ll_loss
from optimization_encoder.separable_trial_glm import ProxSolverParameterGenerator, \
    UnconstrainedFISTASolverParameterGenerator
from optimization_encoder.trial_glm import FittedLNP

N_BINS_BEFORE = 30  # 250 ms, or 30 frames at 120 Hz
N_BINS_AFTER = 18  # 150 ms, or 18 frames at 120 Hz
SAMPLES_PER_BIN = 8.3333333 * 20  # 120 Hz -> 8.333 ms period

L1_GRID = np.logspace(np.log10(1e-7), np.log10(1e-5), 5)
FIXED_WN_WEIGHT = 1e-2
N_ITER_OUTER = 2

# We only use a small number of good manually identified cells for the grid
# search. This is to improve the computational efficiency of the grid search
KNOWN_GOOD_CELLS_BY_REF = {
    ('2017-12-04-5', 'data005'): {
        'ON parasol': [449, 452, 686, 968],
        'OFF parasol': [162, 295, 471, 1169],
        'ON midget': [195, 208, 436, 1018],
        'OFF midget': [187, 231, 327]
    },
    ('2018-08-07-5', 'data000'): {
        'ON parasol': [1228, 478, 788, 1094],
        'OFF parasol': [157, 423, 754, 848],
        'ON midget': [134, 208, 436, 772],
        'OFF midget': [815, 808, 891, 727]
    },
    ('2019-11-07-0', 'data003'): {
        'ON parasol': [286, 638, 1345, 26],
        'OFF parasol': [269, 508, 633, 1099],
        'ON midget': [1070, 666, 1500, 202],
        'OFF midget': [11, 403, 1197, 854]
    },
    ('2018-03-01-0', 'data010'): {
        'ON parasol': [1160, 404, 123, 71],
        'OFF parasol': [475, 56, 177, 1398],
        'ON midget': [395, 831, 98, 544],
        'OFF midget': [1190, 40, 63, 522]
    },
    ('2018-11-12-5', 'data008'): {
        'ON parasol': [275, 549, 691, 1000],
        'OFF parasol': [2, 853, 499, 774],
        'ON midget': [273, 28, 120, 996],
        'OFF midget': [814, 42, 587, 128]
    },
    ('2017-11-29-0', 'data001'): {
        'ON parasol': [10, 1244, 718, 250],
        'OFF parasol': [560, 87, 1222, 319],
        'ON midget': [1123, 596, 74, 1286],
        'OFF midget': [1120, 1169, 1231, 871]
    }
}

Joint_WN_Flash_LNP_Hyperparams = namedtuple('Joint_WN_Flash_LNP_Hyperparams',
                                            ['l1_sparse', 'wn_weight'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Grid search for hyperparameters for the joint WN-flashed nscenes LNP fit')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('cell_type', type=str, help='particular cell type to fit')
    parser.add_argument('save_path', type=str, help='path to save grid search pickle file')
    parser.add_argument('-n', '--outer_iter', type=int, default=N_ITER_OUTER,
                        help='number of outer coordinate descent iterations to use')
    parser.add_argument('-m', '--seconds_wn', type=int, default=300,
                        help='number of seconds of white noise data to use')
    parser.add_argument('-j', '--jitter_time', type=float, default=0.0,
                        help='SD (units number of samples) to do Gaussian jitter of spike times')
    parser.add_argument('-w', '--wn_weight', type=float, default=1e-2,
                        help='Weight to place on the WN loss function')
    parser.add_argument('-hp', '--half_prec', action='store_true', default=False,
                        help='Use half-precision for linear filter operations at the input of GLM.')
    args = parser.parse_args()

    seconds_white_noise = args.seconds_wn
    cell_type = args.cell_type

    prec_type = torch.float16 if args.half_prec else torch.float32

    device = torch.device('cuda')

    config_settings = dcp.read_config_file(args.cfg_file)

    #### Now load the natural scenes dataset #################################
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)
    image_rescale_lambda_torch = du.make_torch_image_transform_lambda(image_rescale_low, image_rescale_high)

    ##########################################################################
    # Load precomputed cell matches, crops, etc.
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct

    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)

    #######################################################################
    # Load Vision dataset and frames for the reference dataset first
    ref_dataset_info = config_settings['ReferenceDataset'] # type: dcp.DatasetInfo

    piece_name, ref_datarun_name = dcp.awsify_piece_name_and_datarun_lookup_key(
        ref_dataset_info.path, ref_dataset_info.name)

    vision_ref_dataset = vl.load_vision_data(ref_dataset_info.path,
                                             ref_dataset_info.name,
                                             include_params=True,
                                             include_neurons=True)

    wn_ttl_times = vision_ref_dataset.get_ttl_times()

    wn_frame_generator = RandomNoiseFrameGenerator.construct_from_xml(ref_dataset_info.wn_xml)
    wn_interval = wn_frame_generator.refresh_interval  # TODO verify that this doesn't go crazy
    wn_rel_downsample = wn_frame_generator.stixel_height // (2 * nscenes_downsample_factor)

    wn_synchro_block = WhiteNoiseSynchroSection(wn_frame_generator,
                                                wn_ttl_times,
                                                WN_N_FRAMES_PER_TRIGGER // wn_interval,
                                                SAMPLE_RATE)

    loaded_wn_movie = ddu.LoadedWNMovies(ref_dataset_info.path,
                                         ref_dataset_info.name,
                                         vision_ref_dataset,
                                         wn_synchro_block)

    #######################################################################
    # Then load the flashed nscenes dataset
    nscenes_dataset_info_list = config_settings[dcp.NScenesFlashedDatasetSection.OUTPUT_KEY]

    create_test_dataset = (dcp.TestFlashedSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutFlashedSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestFlashedSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutFlashedSection.MOVIE_BLOCK_DESCRIPTOR]

    bin_width_time_ms = int(np.around(SAMPLES_PER_BIN / 20, decimals=0))
    stimulus_onset_time_length = int(np.around(100 / bin_width_time_ms, decimals=0))

    nscenes_dset_list = ddu.load_nscenes_dataset_and_timebin_blocks3(
        nscenes_dataset_info_list,
        SAMPLES_PER_BIN,
        N_BINS_BEFORE,
        N_BINS_AFTER,
        stimulus_onset_time_length,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
        downsample_factor=nscenes_downsample_factor,
        crop_w_low=crop_width_low,
        crop_w_high=crop_width_high,
        crop_h_low=crop_height_low,
        crop_h_high=crop_height_high
    )

    for item in nscenes_dset_list:
        item.load_frames_from_disk()

    ###### Get model hyperparameters ###########################################
    # FIXME we need a new hyperparameter dict
    ref_ds_lookup_key = dcp.generate_lookup_key_from_dataset_info(ref_dataset_info)
    model_fit_hyperparameters = \
        CROPPED_FLASHED_WN_LNP_HYPERPARAMETERS_FN_BY_PIECE[ref_ds_lookup_key]()[cell_type]
    timecourse_basis = model_fit_hyperparameters.timecourse_basis
    (n_spatbasis_h, n_spatbasis_w), spat_basis_type = model_fit_hyperparameters.spatial_basis
    n_bins_timefilt = timecourse_basis.shape[1]

    timecourse_basis_torch = torch.tensor(timecourse_basis, dtype=torch.float32, device=device)

    ###### Compute the STA timecourse, and use that as the initial guess for ###
    ###### the timecourse ######################################################
    # Load the timecourse initial guess
    relev_cell_ids = cells_ordered.get_reference_cell_order(cell_type)
    with open(config_settings[dcp.OutputSection.INITIAL_GUESS_TIMECOURSE], 'rb') as pfile:
        timecourses_by_type = pickle.load(pfile)
        avg_timecourse = timecourses_by_type[cell_type]

    initial_timevector_guess = np.linalg.solve(timecourse_basis @ timecourse_basis.T,
                                               timecourse_basis @ avg_timecourse)

    ##########################################################################
    # Do the fitting for each cell
    relevant_ref_cell_ids = KNOWN_GOOD_CELLS_BY_REF[(piece_name, ref_datarun_name)][args.cell_type]
    n_cells = len(relevant_ref_cell_ids)

    fitted_models_dict = {}
    test_loss_matrix = np.zeros((n_cells, L1_GRID.shape[0]))
    train_loss_matrix = np.zeros((n_cells, L1_GRID.shape[0]))

    model_fit_pbar = tqdm.tqdm(total=n_cells)
    for idx, cell_id in enumerate(relevant_ref_cell_ids):

        cell_ix = relev_cell_ids.index(cell_id)
        crop_bbox = bounding_boxes_by_type[cell_type][cell_ix]  # type: CroppedSTABoundingBox

        crop_slice_h, crop_slice_w = crop_bbox.make_precropped_sliceobj(
            crop_hlow=crop_height_low,
            crop_hhigh=crop_height_high,
            crop_wlow=crop_width_low,
            crop_whigh=crop_width_high,
            downsample_factor=nscenes_downsample_factor
        )

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

        # Construct the spline basis
        hhh = crop_bounds_h[1] - crop_bounds_h[0]
        www = crop_bounds_w[1] - crop_bounds_w[0]
        spat_spline_basis = make_spatial_basis_from_hyperparameters(crop_bounds_h, crop_bounds_w,
                                                                    n_spatbasis_h, n_spatbasis_w,
                                                                    spat_basis_type,
                                                                    nscenes_downsample_factor=nscenes_downsample_factor)

        spat_spline_basis_imshape = spat_spline_basis.T.reshape(-1, hhh, www)

        ##########################################################################
        # Bin WN spikes and precompute WN stimulus stuff
        training_wn_movie_blocks = ddu.preload_bind_wn_movie_patches_at_framerate(
            loaded_wn_movie,
            [(0, int(WN_FRAME_RATE * args.seconds_wn // wn_interval)), ],
            (sta_cropping_slice_h, sta_cropping_slice_w)
        )

        wn_center_spikes_frame_rate_torch = frame_rate_wn_lnp_multidata_bin_spikes(
            training_wn_movie_blocks,
            (cell_type, cell_id),
            device,
            jitter_time_amount=args.jitter_time,
            prec_dtype=prec_type,
            trim_spikes_seq=(n_bins_timefilt - 1)
        )[0]

        training_wn_movie_block = training_wn_movie_blocks[0]

        wn_movie_basis_applied_flat_torch = preapply_spatial_basis_to_wn(
            training_wn_movie_block.stimulus_frame_patches_wn_resolution,
            image_rescale_lambda_torch,
            spat_spline_basis_imshape,
            device,
            prec_dtype=prec_type,
        ).T.contiguous()

        torch.cuda.synchronize(device)

        #############################################################
        ### bin nscenes spikes and pre-compute nscenes stimulus stuff
        nscenes_patch_list = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                  ddu.PartitionType.TRAIN_PARTITION,
                                                                  crop_slice_h=crop_slice_h,
                                                                  crop_slice_w=crop_slice_w)

        ns_cell_spikes_torch = flashed_ns_bin_spikes_only(
            nscenes_patch_list,
            (cell_type, cell_id),
            cells_ordered,
            device,
            jitter_time_amount=args.jitter_time,
            prec_dtype=prec_type,
            trim_spikes_seq=(n_bins_timefilt - 1)
        )

        ns_stim_time_conv = flashed_ns_bin_spikes_precompute_timecourse_basis_conv2(
            nscenes_patch_list,
            timecourse_basis,
            device,
            prec_dtype=prec_type
        )

        ns_spat_basis_filt = flashed_ns_precompute_spatial_basis(
            nscenes_patch_list,
            spat_spline_basis.T,
            image_rescale_lambda_torch,
            device,
            prec_dtype=prec_type
        )

        ##############################################
        # bin test nscenes stuff as well
        test_nscenes_patch_list = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                       ddu.PartitionType.TEST_PARTITION,
                                                                       crop_slice_h=crop_slice_h,
                                                                       crop_slice_w=crop_slice_w)

        test_ns_cell_spikes_torch = flashed_ns_bin_spikes_only(
            test_nscenes_patch_list,
            (cell_type, cell_id),
            cells_ordered,
            device,
            jitter_time_amount=args.jitter_time,
            prec_dtype=prec_type,
            trim_spikes_seq=(n_bins_timefilt - 1)
        )

        test_ns_stim_time_conv = flashed_ns_bin_spikes_precompute_timecourse_basis_conv2(
            test_nscenes_patch_list,
            timecourse_basis,
            device,
            prec_dtype=prec_type
        )

        test_ns_spat_basis_filt = flashed_ns_precompute_spatial_basis(
            test_nscenes_patch_list,
            spat_spline_basis.T,
            image_rescale_lambda_torch,
            device,
            prec_dtype=prec_type
        )

        hyperparams_pbar = tqdm.tqdm(total=L1_GRID.shape[0])
        for l1_ix, l1_reg_value in enumerate(L1_GRID):
            solver_schedule = (ProxSolverParameterGenerator(1000, 500),
                               UnconstrainedFISTASolverParameterGenerator(1000, 500))

            loss, lnp_fit_params = new_style_LNP_joint_wn_flashed_ns_alternating_optim(
                ns_spat_basis_filt,
                ns_cell_spikes_torch,
                ns_stim_time_conv,
                wn_movie_basis_applied_flat_torch,
                wn_center_spikes_frame_rate_torch,
                timecourse_basis_torch,
                fused_poisson_spiking_neg_ll_loss,
                solver_schedule,
                args.outer_iter,
                device,
                l1_spat_sparse_lambda=l1_reg_value,
                initial_guess_timecourse=torch.tensor(initial_timevector_guess[None, :], dtype=torch.float32, device=device),
                outer_opt_verbose=True
            )

            spat_filt, timecourse_w, bias = lnp_fit_params

            spat_filt_np = spat_filt.detach().cpu().numpy()
            timecourse_w_np = timecourse_w.detach().cpu().numpy()
            bias_np = bias.detach().cpu().numpy()

            # shape (1, n_spat_basis) @ (n_spat_basis, n_pixels) -> (1, n_pixels)
            spat_filt_as_image_shape = (spat_filt_np @ spat_spline_basis.T).reshape(hhh, www)
            time_filters = timecourse_w_np @ timecourse_basis

            evaluation_model = Flashed_PreappliedSpatialSingleCellLNPLoss(
                spat_filt,
                timecourse_w,
                bias,
                fused_poisson_spiking_neg_ll_loss,
                dtype=torch.float32
            ).to(device)

            with torch.no_grad(), autocast('cuda'):
                mean_test_loss = evaluation_model(
                    test_ns_spat_basis_filt,
                    test_ns_stim_time_conv,
                    test_ns_cell_spikes_torch
                ).item()

            test_loss_matrix[idx, l1_ix] = mean_test_loss
            train_loss_matrix[idx, l1_ix] = loss

            print(
                f'cell_id {cell_id}, L1 {l1_reg_value}: opt_loss {loss}, ns_test_loss {mean_test_loss}')

            # cleanup GPU memory
            del evaluation_model

            additional_fit_params = {
                'fista_schedule': (ProxSolverParameterGenerator(500, 250), ProxSolverParameterGenerator(250, 250)),
                'outer_iter_num': N_ITER_OUTER
            }

            model_summary_parameters = FittedLNP(cell_id, spat_filt_as_image_shape, bias,
                                                 timecourse_w, additional_fit_params, loss, None)

            fitted_models_dict[(l1_reg_value,)] = model_summary_parameters
            hyperparams_pbar.update(1)

        hyperparams_pbar.close()

        model_fit_pbar.update(1)

        # cleanup GPU of everything
        del wn_center_spikes_frame_rate_torch, wn_movie_basis_applied_flat_torch
        del ns_cell_spikes_torch, ns_stim_time_conv, ns_spat_basis_filt
        del test_ns_cell_spikes_torch, test_ns_stim_time_conv, test_ns_spat_basis_filt

        torch.cuda.empty_cache()

    with open(args.save_path, 'wb') as pfile:
        pickle_obj = {
            'models': fitted_models_dict,
            'L1': L1_GRID,
            'test_loss': test_loss_matrix,
            'train_opt_loss': train_loss_matrix,
            'timecourse_basis': timecourse_basis,
            'spike_jitter': args.jitter_time
        }

        pickle.dump(pickle_obj, pfile)
