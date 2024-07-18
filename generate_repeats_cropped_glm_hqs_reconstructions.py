import argparse
import pickle
from typing import List

import numpy as np
import torch

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from denoise_inverse_alg.glm_inverse_alg import reinflate_cropped_glm_model, reinflate_cropped_fb_only_glm_model, \
    noreduce_bernoulli_neg_LL
from generate_cropped_glm_hqs_reconstructions import make_glm_stim_time_component, GridSearchParams, \
    load_fitted_glm_families, LinearModelBinningRange, make_HQS_X_prob_solver_generator_fn, \
    make_HQS_Z_prob_solver_generator_fn
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from reconstruction_fns.flashed_reconstruction_toplevel_fns import batch_parallel_generate_flashed_hqs_reconstructions


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Generate (shuffled) repeat reconstructions for flashed images using LNBRC encoding and dCNN image prior")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('yaml_path', type=str, help='path to YAML specifying encoding model weights')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-n', '--noise_init', type=float, default=1e-3)
    parser.add_argument('-l', '--linear_init', type=str, default=None,
                        help='optional, path to linear model if using linear model to initialize HQS')
    parser.add_argument('-bb', '--before_lin_bins', type=int, default=0, help='start binning for linear')
    parser.add_argument('-aa', '--after_lin_bins', type=int, default=0, help='end binning for linear')
    parser.add_argument('-st', '--start', type=float, default=0.1, help='HQS rho first value')
    parser.add_argument('-en', '--end', type=float, default=10.0, help='HQS rho last value')
    parser.add_argument('-i', '--iter', type=int, default=25, help='HQS num outer iters')
    parser.add_argument('-lam', '--prior_lambda', type=float, default=0.2, help='HQS prior weight')
    parser.add_argument('-m', '--use_mask', action='store_true', default=False, help='use valid mask')
    parser.add_argument('-sh', '--shuffle', type=int, default=0,
                        help='number of shuffled repeats to generate; if 0 use real data')
    parser.add_argument('-f', '--feedback_only', action='store_true', default=False,
                        help='GLM model specified by model_cfg_path is feedback-only')

    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    reference_piece_path = config_settings['ReferenceDataset'].path

    ################################################################
    # Load the cell types and matching
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

    ################################################################
    # Load some of the model fit parameters
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)
        blurred_stas_by_type = pickle.load(pfile)

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

    ################################################################
    convex_hull_valid_mask = None
    if args.use_mask:
        # Compute the valid region mask
        print("Computing valid region mask")
        ref_lookup_key = dcp.awsify_piece_name_and_datarun_lookup_key(config_settings['ReferenceDataset'].path,
                                                                      config_settings['ReferenceDataset'].name)
        valid_mask = make_sig_stixel_loss_mask(
            ref_lookup_key,
            blurred_stas_by_type,
            crop_wlow=crop_width_low,
            crop_whigh=crop_width_high,
            crop_hlow=crop_height_low,
            crop_hhigh=crop_height_high,
            downsample_factor=nscenes_downsample_factor
        )

        convex_hull_valid_mask = compute_convex_hull_of_mask(valid_mask)

    #################################################################
    # Load the raw data
    n_bins_before = config_settings[dcp.TimebinningSection.NBINS_BEFORE_TRANS]
    n_bins_after = config_settings[dcp.TimebinningSection.NBINS_AFTER_TRANS]
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    # Load the natural scenes Vision datasets and determine what the
    # train and test partitions are
    nscenes_dataset_info_list = config_settings[dcp.NScenesFlashedDatasetSection.OUTPUT_KEY]

    create_test_dataset = (dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]

    bin_width_time_ms = int(np.around(samples_per_bin / 20, decimals=0))
    stimulus_onset_time_length = int(np.around(100 / bin_width_time_ms, decimals=0))

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

    ###############################################################
    nscenes_dset_list[0].load_repeat_frames()
    repeat_frames = image_rescale_lambda(nscenes_dset_list[0].repeat_frames_cached)

    n_repeat_frames, height, width = repeat_frames.shape

    data_repeats_response_vector = ddu.timebin_load_repeats_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        nscenes_dset_list,
    )

    if args.shuffle != 0:
        data_repeats_response_vector = ddu.shuffle_loaded_repeat_spikes(
            data_repeats_response_vector,
            args.shuffle
        )

    n_repeat_trials, _, n_cells, n_bins = data_repeats_response_vector.shape
    flat_data_repeats_response_vector = data_repeats_response_vector.reshape(-1, n_cells, n_bins)

    glm_stim_time_component = make_glm_stim_time_component(config_settings)

    fitted_glm_paths = parse_prefit_glm_paths(args.yaml_path)
    fitted_glm_families = load_fitted_glm_families(fitted_glm_paths)
    model_reinflation_fn = reinflate_cropped_fb_only_glm_model if args.feedback_only \
        else reinflate_cropped_glm_model

    packed_glm_tensors = model_reinflation_fn(
        fitted_glm_families,
        bounding_boxes_by_type,
        cells_ordered,
        height,
        width,
        downsample_factor=nscenes_downsample_factor,
        crop_width_low=crop_width_low,
        crop_width_high=crop_width_high,
        crop_height_low=crop_height_low,
        crop_height_high=crop_height_high
    )

    hyperparameters = GridSearchParams(args.start, args.end, args.prior_lambda, args.iter)

    # Set up linear model if we choose to use it for initialization
    linear_model_param = None
    if args.linear_init is not None:
        linear_decoder = torch.load(args.linear_init, map_location=device)
        linear_bin_cutoffs = LinearModelBinningRange(args.before_lin_bins, args.after_lin_bins)
        linear_model_param = (linear_decoder, linear_bin_cutoffs)

    target_reconstructions_flat = batch_parallel_generate_flashed_hqs_reconstructions(
        flat_data_repeats_response_vector,
        (image_rescale_low, image_rescale_high),
        packed_glm_tensors,
        noreduce_bernoulli_neg_LL,
        glm_stim_time_component,
        hyperparameters,
        make_HQS_X_prob_solver_generator_fn,
        make_HQS_Z_prob_solver_generator_fn,
        16,
        device,
        initialize_noise_level=args.noise_init,
        initialize_linear_model=linear_model_param,
        valid_region_mask=convex_hull_valid_mask
    )

    repeat_reconstructions = target_reconstructions_flat.reshape(n_repeat_trials, n_repeat_frames, height, width)

    with open(args.save_path, 'wb') as pfile:
        save_data = {
            'ground_truth': repeat_frames,
            'repeats' : repeat_reconstructions,
            'shuffle': args.shuffle
        }

        pickle.dump(save_data, pfile)

    print('done')
