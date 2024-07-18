'''
The frame rate and bin size for this script is hardcoded to be 8.333 ms, corresponding
to a 120 Hz frame rate on the stimulus monitor.

This is because we choose to compute the LNP models at frame rate, which is a non-integer
multiple of the electrical sample rate, and because the TTL trigger design for the white
noise and the flashed stimulus are so different that there isn't a simple intuitive way
to do both at the same time at 120 Hz
'''

import argparse
import pickle
from typing import Dict, List
from enum import Enum

import numpy as np
import torch

import tqdm

import visionloader as vl
from whitenoise import RandomNoiseFrameGenerator

from basis_functions.spatial_basis_functions import make_spatial_basis_from_hyperparameters
from glm_precompute.flashed_glm_precompute import flashed_ns_bin_spikes_precompute_timecourse_basis_conv2, \
    flashed_ns_precompute_spatial_basis, flashed_ns_bin_spikes_only
from glm_precompute.wn_glm_precompute import preapply_spatial_basis_to_wn
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import CroppedSTABoundingBox
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
from lib.dataset_specific_hyperparams.jitter_lnp_hyperparameters import \
    CROPPED_FLASHED_WN_LNP_HYPERPARAMETERS_FN_BY_PIECE

from lib.dataset_specific_ttl_corrections.wn_ttl_structure import WhiteNoiseSynchroSection
from lib.dataset_specific_ttl_corrections.ttl_interval_constants import WN_FRAME_RATE, WN_N_FRAMES_PER_TRIGGER, \
    SAMPLE_RATE
from lnp_precompute.wn_lnp_precompute import frame_rate_wn_lnp_multidata_bin_spikes
from new_style_optim_encoder.separable_trial_lnp import new_style_LNP_joint_wn_flashed_ns_alternating_optim
from optimization_encoder.ct_glm import fused_poisson_spiking_neg_ll_loss
from optimization_encoder.separable_trial_glm import ProxSolverParameterGenerator, \
    UnconstrainedFISTASolverParameterGenerator
from optimization_encoder.trial_glm import FittedLNP, FittedLNPFamily

N_ITER_OUTER = 2

N_BINS_BEFORE = 30  # 250 ms, or 30 frames at 120 Hz
N_BINS_AFTER = 18  # 150 ms, or 18 frames at 120 Hz
SAMPLES_PER_BIN = 8.3333333 * 20  # 120 Hz -> 8.333 ms period


class SolverMode(Enum):
    FULL = 1
    STUPID = 2
    DEBUG = 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fit frame-rate LNPs jointly to flashed images and white noise')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('cell_type', type=str, help='particular cell type to fit')
    parser.add_argument('save_path', type=str, help='path to save grid search pickle file')
    parser.add_argument('-m', '--seconds_wn', type=int, default=300,
                        help='number of seconds of white noise data to use')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-s', '--stupid', action='store_true', default=False)
    parser.add_argument('-l', '--l1_sparse', type=float, default=1e-5,
                        help='hyperparameter for L1 spatial filter sparsity')
    parser.add_argument('-w', '--wn_weight', type=float, default=1e-2,
                        help='Weight to place on the WN loss function')
    parser.add_argument('-j', '--jitter', type=float, default=0.0,
                        help='Amount to jitter the spike times by, in samples; Default 0')
    parser.add_argument('-hp', '--half_prec', action='store_true', default=False,
                        help='Use half-precision for linear filter operations at the input of GLM.')
    args = parser.parse_args()

    prec_type = torch.float16 if args.half_prec else torch.float32

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    cell_type = args.cell_type
    if args.debug:
        solver_mode = SolverMode.DEBUG
    elif args.stupid:
        solver_mode = SolverMode.STUPID
    else:
        solver_mode = SolverMode.FULL

    # get additional config settings
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

    l1_spat_sparse = args.l1_sparse
    wn_model_weight = args.wn_weight

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

    fitted_models_dict = {}  # type: Dict[int, FittedLNP]
    model_fit_pbar = tqdm.tqdm(total=len(relev_cell_ids))
    for idx, cell_id in enumerate(relev_cell_ids):
        if solver_mode == SolverMode.DEBUG:
            solver_schedule = (ProxSolverParameterGenerator(500, 250),
                               UnconstrainedFISTASolverParameterGenerator(250, 250))
        elif solver_mode == SolverMode.STUPID:
            solver_schedule = (ProxSolverParameterGenerator(50, 25),
                               UnconstrainedFISTASolverParameterGenerator(50, 25))
        else:
            solver_schedule = (ProxSolverParameterGenerator(1000, 500),
                               UnconstrainedFISTASolverParameterGenerator(1000, 500))


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
            jitter_time_amount=args.jitter,
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
            jitter_time_amount=args.jitter,
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

        loss, lnp_fit_params = new_style_LNP_joint_wn_flashed_ns_alternating_optim(
            ns_spat_basis_filt,
            ns_cell_spikes_torch,
            ns_stim_time_conv,
            wn_movie_basis_applied_flat_torch,
            wn_center_spikes_frame_rate_torch,
            timecourse_basis_torch,
            fused_poisson_spiking_neg_ll_loss,
            solver_schedule,
            N_ITER_OUTER,
            device,
            l1_spat_sparse_lambda=l1_spat_sparse,
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

        fitting_params_to_store = {
            'l1_spat_sparse': l1_spat_sparse,
            'wn_weight': wn_model_weight
        }

        model_summary_parameters = FittedLNP(cell_id, spat_filt_as_image_shape, bias_np,
                                             timecourse_w_np, fitting_params_to_store, loss, None)

        fitted_models_dict[cell_id] = model_summary_parameters
        model_fit_pbar.update(1)

        del spat_filt, timecourse_w, bias

        # GPU cleanup
        del wn_center_spikes_frame_rate_torch, wn_movie_basis_applied_flat_torch
        del ns_stim_time_conv, ns_spat_basis_filt

        torch.cuda.empty_cache()

    model_fit_pbar.close()

    fitted_model_family = FittedLNPFamily(
        fitted_models_dict,
        model_fit_hyperparameters.spatial_basis,
        timecourse_basis,
    )

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(fitted_model_family, pfile)





