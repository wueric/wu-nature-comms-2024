import argparse
import pickle
from typing import Dict, List

import numpy as np
import torch
import tqdm
import visionloader as vl
from whitenoise import RandomNoiseFrameGenerator

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from basis_functions.spatial_basis_functions import make_spatial_basis_from_hyperparameters
from glm_precompute.flashed_glm_precompute import flashed_ns_precompute_spatial_basis, \
    flashed_ns_bin_spikes_precompute_timecourse_basis_conv2, flashed_ns_bin_spikes_precompute_feedback_convs2
from glm_precompute.wn_glm_precompute import preapply_spatial_basis_to_wn_and_temporally_upsample, \
    wn_bin_spikes_precompute_feedback_convs
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_specific_hyperparams.glm_hyperparameters import CROPPED_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE2
from lib.dataset_specific_ttl_corrections.ttl_interval_constants import WN_FRAME_RATE, WN_N_FRAMES_PER_TRIGGER, \
    SAMPLE_RATE
from lib.dataset_specific_ttl_corrections.wn_ttl_structure import WhiteNoiseSynchroSection
from new_style_optim_encoder.separable_fb_only_glm import new_style_joint_wn_flashed_fb_only_alternating_optim
from optimization_encoder.separable_trial_glm import ProxSolverParameterGenerator, \
    UnconstrainedFISTASolverParameterGenerator
from optimization_encoder.trial_glm import mean_bin_batch_bernoulli_spiking_neg_ll_loss, \
    make_batch_mean_binomial_spiking_neg_ll_loss, FittedFBOnlyGLM, \
    FittedFBOnlyGLMFamily, batch_poisson_spiking_neg_ll_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Grid search for hyperparameters for the joint WN-flashed nscenes LNBR (uncoupled) fit')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('cell_type', type=str, help='particular cell type to fit')
    parser.add_argument('save_path', type=str, help='path to save grid search pickle file')
    parser.add_argument('-n', '--outer_iter', type=int, default=2,
                        help='number of outer coordinate descent iterations to use')
    parser.add_argument('-m', '--seconds_wn', type=int, default=600,
                        help='number of seconds of white noise data to use')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-s', '--stupid', action='store_true', default=False)
    parser.add_argument('-l', '--l1_sparse', type=float, default=1e-6,
                        help='hyperparmeter for L1 spatial filter sparsity')
    parser.add_argument('-w', '--wn_weight', type=float, default=1e-2,
                        help='Weight to place on the WN loss function')
    parser.add_argument('-j', '--jitter', type=float, default=0.0,
                        help='Amount to jitter the spike times by, in samples; Default 0')
    parser.add_argument('-hp', '--half_prec', action='store_true', default=False,
                        help='Use half-precision for linear filter operations at the input of GLM.')
    parser.add_argument('-p', '--poisson_loss', action='store_true', default=False,
                        help='Use Poisson loss for training')
    args = parser.parse_args()

    seconds_white_noise = args.seconds_wn
    cell_type = args.cell_type

    prec_type = torch.float32
    if args.half_prec:
        prec_type = torch.float16

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

    ##########################################################################
    # Load the datasets; since we are doing joint fitting with both the flashed
    # images and the white noise datasets, we need to load both the reference
    # dataset and the natural scenes flashes datasets

    # Load Vision dataset and frames for the reference dataset first
    ref_dataset_info = config_settings['ReferenceDataset'] # type: dcp.DatasetInfo

    ###################################################################
    # Load the white noise reference dataset since we are joint fitting
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

    ###################################################################
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

    bin_width_time_ms = samples_per_bin // 20
    stimulus_onset_time_length = 100 // bin_width_time_ms

    nscenes_dset_list = ddu.load_nscenes_dataset_and_timebin_blocks3(
        nscenes_dataset_info_list,
        samples_per_bin,
        n_bins_before,
        n_bins_after,
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

    ref_lookup_key = dcp.generate_lookup_key_from_dataset_info(ref_dataset_info)
    model_fit_hyperparameters = \
        CROPPED_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE2[ref_lookup_key][bin_width_time_ms]()[cell_type]

    cell_distances_by_type = model_fit_hyperparameters.neighboring_cell_dist
    timecourse_basis = model_fit_hyperparameters.timecourse_basis
    feedback_basis = model_fit_hyperparameters.feedback_basis

    l1_spat_sparse = args.l1_sparse
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

    initial_timevector_guess_torch = torch.tensor(initial_timevector_guess,
                                                  dtype=torch.float32,
                                                  device=device)

    #########################################################################
    # Move the bases to GPU
    bump_basis_stimulus_torch = torch.tensor(timecourse_basis, dtype=torch.float32, device=device)
    bump_basis_feedback_torch = torch.tensor(feedback_basis, dtype=torch.float32, device=device)

    ##########################################################################
    # Begin model fitting for each cell
    relevant_cell_ids = cells_ordered.get_reference_cell_order(cell_type)

    fitted_models_dict = {}  # type: Dict[int, FittedFBOnlyGLM]

    model_fit_pbar = tqdm.tqdm(total=len(relevant_cell_ids))
    for idx, cell_id in enumerate(relevant_cell_ids):
        ### Get neighboring cells ######################################

        print(f"idx {idx}, cell_id {cell_id}")

        ### Get cropping objects ########################################
        cell_ix = relevant_cell_ids.index(cell_id)
        crop_bbox = bounding_boxes_by_type[cell_type][cell_ix]
        crop_slice_h, crop_slice_w = crop_bbox.make_precropped_sliceobj(
            crop_hlow=crop_height_low,
            crop_hhigh=crop_height_high,
            crop_wlow=crop_width_low,
            crop_whigh=crop_width_high,
            downsample_factor=nscenes_downsample_factor
        )

        crop_bounds_h, crop_bounds_w = crop_bbox.make_precropped_sliceobj(
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
                                                                    spat_basis_type,
                                                                    nscenes_downsample_factor=nscenes_downsample_factor)

        spat_spline_basis_imshape = spat_spline_basis.T.reshape(-1, hhh, www)

        sta_cropping_slice_h, sta_cropping_slice_w = crop_bbox.make_cropping_sliceobj(
            crop_hlow=crop_height_low,
            crop_hhigh=crop_height_high,
            crop_wlow=crop_width_low,
            crop_whigh=crop_width_high,
            downsample_factor=wn_rel_downsample * nscenes_downsample_factor
        )

        training_wn_movie_block = ddu.preload_bind_wn_movie_patches(
            loaded_wn_movie,
            [(0, int(WN_FRAME_RATE * args.seconds_wn // wn_interval)), ],
            (sta_cropping_slice_h, sta_cropping_slice_w)
        )[0]

        ################################################################
        # now do the same for the white noise
        wn_timebin = ddu.construct_wn_movie_timebins(training_wn_movie_block,
                                                           samples_per_bin)

        wn_center_spikes_torch, wn_feedback_conv_torch = wn_bin_spikes_precompute_feedback_convs(
            training_wn_movie_block,
            wn_timebin,
            (cell_type, cell_id),
            feedback_basis,
            device,
            jitter_time_amount=args.jitter,
            prec_dtype=prec_type,
            trim_spikes_seq=feedback_basis.shape[1] - 1
        )

        wn_movie_upsampled_flat_torch = preapply_spatial_basis_to_wn_and_temporally_upsample(
            training_wn_movie_block.stimulus_frame_patches_wn_resolution,
            training_wn_movie_block.timing_synchro,
            wn_timebin,
            image_rescale_lambda_torch,
            spat_spline_basis_imshape,
            device,
            prec_dtype=prec_type,
        )

        torch.cuda.synchronize(device)

        ### bin nscenes spikes and pre-compute nscenes stimulus stuff
        nscenes_patch_list = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                  ddu.PartitionType.TRAIN_PARTITION,
                                                                  crop_slice_h=crop_slice_h,
                                                                  crop_slice_w=crop_slice_w)

        ns_feedback_spikes_torch, ns_fb_conv = flashed_ns_bin_spikes_precompute_feedback_convs2(
            nscenes_patch_list,
            feedback_basis,
            (cell_type, cell_id),
            cells_ordered,
            device,
            jitter_time_amount=args.jitter,
            prec_dtype=prec_type,
            trim_spikes_seq=(feedback_basis.shape[1] - 1)
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

        ns_feedback_start_sample = bump_basis_feedback_torch.shape[1] - 1
        wn_feedback_start_sample = bump_basis_feedback_torch.shape[1] - 1

        if args.poisson_loss:
            loss_fn = batch_poisson_spiking_neg_ll_loss
        else:
            loss_fn = mean_bin_batch_bernoulli_spiking_neg_ll_loss if not use_binom_loss else \
                make_batch_mean_binomial_spiking_neg_ll_loss(n_bins_binom)

        if args.debug:
            solver_schedule = (
            ProxSolverParameterGenerator(500, 250), UnconstrainedFISTASolverParameterGenerator(250, 250))
        else:
            solver_schedule = (
            ProxSolverParameterGenerator(1000, 500), UnconstrainedFISTASolverParameterGenerator(1000, 500))

        if args.stupid:
            solver_schedule = (ProxSolverParameterGenerator(50, 25), UnconstrainedFISTASolverParameterGenerator(50, 25))

        torch.cuda.synchronize(device)

        wn_regularized_params = new_style_joint_wn_flashed_fb_only_alternating_optim(
            ns_spat_basis_filt,
            ns_feedback_spikes_torch,
            ns_stim_time_conv,
            ns_fb_conv,
            wn_movie_upsampled_flat_torch,
            wn_center_spikes_torch,
            wn_feedback_conv_torch,
            bump_basis_stimulus_torch,
            loss_fn,
            solver_schedule,
            n_iter_outer,
            device,
            weight_wn=wn_model_weight,
            l1_spat_sparse_lambda=l1_spat_sparse,
            initial_guess_timecourse=initial_timevector_guess_torch[None, :],
            outer_opt_verbose=True,
        )

        train_loss, (spat_filt, timecourse_w, feedback_w, bias) = wn_regularized_params

        spat_filt_np = spat_filt.detach().cpu().numpy()
        timecourse_w_np = timecourse_w.detach().cpu().numpy()
        feedback_w_np = feedback_w.detach().cpu().numpy()
        bias_np = bias.detach().cpu().numpy()

        # shape (1, n_spat_basis) @ (n_spat_basis, n_pixels) -> (1, n_pixels)
        spat_filt_as_image_shape = (spat_filt_np @ spat_spline_basis.T).reshape(hhh, www)
        time_filters = timecourse_w_np @ timecourse_basis
        feedback_td = feedback_w_np @ feedback_basis

        fitting_params_to_store = {
            'l1_spat_sparse': l1_spat_sparse,
            'wn_weight': wn_model_weight
        }

        model_summary_parameters = FittedFBOnlyGLM(cell_id, spat_filt_as_image_shape, bias_np, timecourse_w_np,
                                                   feedback_w_np, fitting_params_to_store, train_loss, None)

        fitted_models_dict[cell_id] = model_summary_parameters

        model_fit_pbar.update(1)

        del spat_filt, timecourse_w, feedback_w, bias

        # GPU cleanup
        del wn_center_spikes_torch
        del wn_movie_upsampled_flat_torch
        del ns_feedback_spikes_torch, ns_fb_conv, ns_stim_time_conv
        del ns_spat_basis_filt

        torch.cuda.empty_cache()

    model_fit_pbar.close()

    fitted_model_family = FittedFBOnlyGLMFamily(
        fitted_models_dict,
        model_fit_hyperparameters.spatial_basis,
        timecourse_basis,
        feedback_basis,
    )

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(fitted_model_family, pfile)
