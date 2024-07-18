import argparse
import pickle
from typing import List, Iterator

import numpy as np
import torch

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from denoise_inverse_alg.hqs_alg import HQS_ParameterizedSolveFn, BatchParallel_HQS_XGenerator, \
    BatchParallel_DirectSolve_HQS_ZGenerator
from denoise_inverse_alg.poisson_inverse_alg import reinflate_cropped_lnp_model, poisson_loss_noreduce
from eval_fns.eval import MaskedMSELoss, MS_SSIM, SSIM
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from optimization_encoder.trial_glm import load_fitted_lnp_families
from reconstruction_fns.flashed_reconstruction_toplevel_fns import batch_parallel_flashed_hqs_grid_search, \
    LinearModelBinningRange

BATCH_SIZE = 16
HQS_MAX_ITER = 25
GRID_SEARCH_TEST_N_IMAGES = 80

N_BINS_BEFORE = 30  # 250 ms, or 30 frames at 120 Hz
N_BINS_AFTER = 18  # 150 ms, or 18 frames at 120 Hz
SAMPLES_PER_BIN = 8.3333333 * 20  # 120 Hz -> 8.333 ms period


def make_HQS_X_prob_solver_generator_fn() -> Iterator[HQS_ParameterizedSolveFn]:
    return BatchParallel_HQS_XGenerator(first_niter=300,subsequent_niter=300)


def make_HQS_Z_prob_solver_generator_fn() -> Iterator[HQS_ParameterizedSolveFn]:
    return BatchParallel_DirectSolve_HQS_ZGenerator()



if __name__ == '__main__':

    parser = argparse.ArgumentParser("Grid search for flashed reconstruction, using LNP encoding and dCNN image prior.")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('model_cfg_path', type=str, help='path to YAML specifying where the GLM fits are')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-l', '--linear_init', type=str, default=None,
                        help='optional, path to linear model if using linear model to initialize HQS')
    parser.add_argument('-bb', '--before_lin_bins', type=int, default=0, help='start binning for linear')
    parser.add_argument('-aa', '--after_lin_bins', type=int, default=0, help='end binning for linear')
    parser.add_argument('-n', '--use_noncausal', action='store_true', default=False)
    parser.add_argument('-ls', '--lambda_start', nargs=3,
                        help='Grid search parameters for HQS prox start. Start, end, n_grid')
    parser.add_argument('-le', '--lambda_end', nargs=3,
                        help='Grid search parameters for HQS prox end. Start, end, n_grid')
    parser.add_argument('-p', '--prior', type=float, nargs='+',
                        help='Grid search parameters for prior lambda. Specify weights explicitly')
    parser.add_argument('-it', '--max_iter', type=int, default=25, help='Number of HQS iterations')
    parser.add_argument('-j', '--jitter', type=float, default=0.0,
                        help='jitter sample standard deviation, in units of electrical samples')
    parser.add_argument('-m', '--use_mask', action='store_true', default=False,
                        help='use valid mask for NN prior')
    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    grid_search_lambda_start = np.logspace(float(args.lambda_start[0]), float(args.lambda_start[1]),
                                           int(args.lambda_start[2]))
    grid_search_lambda_end = np.logspace(float(args.lambda_end[0]), float(args.lambda_end[1]),
                                         int(args.lambda_end[2]))
    grid_search_prior = np.array(args.prior)

    bin_width_time_ms = int(np.around(SAMPLES_PER_BIN / 20, decimals=0))
    stimulus_onset_time_length = int(np.around(100 / bin_width_time_ms, decimals=0))

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

    #################################################################
    # Load the raw data
    n_bins_before = config_settings[dcp.TimebinningSection.NBINS_BEFORE_TRANS]
    n_bins_after = config_settings[dcp.TimebinningSection.NBINS_AFTER_TRANS]
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    # Load the natural scenes Vision datasets and determine what the
    # train and test partitions are
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
        crop_h_low=crop_height_low,
        crop_h_high=crop_height_high,
        crop_w_low=crop_width_low,
        crop_w_high=crop_width_high
    )

    for item in nscenes_dset_list:
        item.load_frames_from_disk()

    ###############################################################
    # Load the stimulus frames
    nscenes_test_patch_list = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                   ddu.PartitionType.TEST_PARTITION)

    test_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(nscenes_test_patch_list))

    n_test_frames = test_frames.shape[0]

    _, height, width = test_frames.shape

    # use a separate script to do GLM reconstructions once we get those going
    test_glm_response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        nscenes_test_patch_list,
        jitter_time_amount=args.jitter
    )

    #################################################################
    # Load the LNPs
    fitted_lnp_paths = parse_prefit_glm_paths(args.yaml_path)
    fitted_lnp_families = load_fitted_lnp_families(fitted_lnp_paths)
    packed_lnp_tensors = reinflate_cropped_lnp_model(
        fitted_lnp_families,
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

    ################################################################
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

    invalid_mask = ~convex_hull_valid_mask

    valid_mask_float_torch = torch.tensor(convex_hull_valid_mask, dtype=torch.float32, device=device)
    inverse_mask_float_torch = torch.tensor(invalid_mask, dtype=torch.float32, device=device)

    ################################################################
    # Construct the loss evaluation modules
    masked_mse_module = MaskedMSELoss(convex_hull_valid_mask).to(device)

    ms_ssim_module = MS_SSIM(channel=1,
                             data_range=1.0,
                             win_size=9,
                             weights=[0.07105472, 0.45297383, 0.47597145],
                             not_valid_mask=inverse_mask_float_torch).to(device)

    ssim_module = SSIM(channel=1,
                       data_range=1.0,
                       not_valid_mask=inverse_mask_float_torch).to(device)

    ################################################################
    # Set up linear model if we choose to use it for initialization
    linear_model_param = None
    if args.linear_init is not None:
        linear_decoder = torch.load(args.linear_init, map_location=device)
        linear_bin_cutoffs = LinearModelBinningRange(args.before_lin_bins, args.after_lin_bins)

        linear_model_param = (linear_decoder, linear_bin_cutoffs)

    ################################################################
    # Do the grid search
    grid_search_output_dict = batch_parallel_flashed_hqs_grid_search(
        test_glm_response_vector[:GRID_SEARCH_TEST_N_IMAGES, ...],
        test_frames[:GRID_SEARCH_TEST_N_IMAGES, ...],
        (image_rescale_low, image_rescale_high),
        BATCH_SIZE,
        packed_lnp_tensors,
        poisson_loss_noreduce,
        nscenes_dset_list[0].stimulus_time_component,
        masked_mse_module,
        ssim_module,
        ms_ssim_module,
        grid_search_lambda_start,
        grid_search_lambda_end,
        grid_search_prior,
        args.max_iter,
        make_HQS_X_prob_solver_generator_fn,
        make_HQS_Z_prob_solver_generator_fn,
        device,
        initialize_linear_model=linear_model_param,
        valid_region_mask=convex_hull_valid_mask if args.use_mask else None
    )

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(grid_search_output_dict, pfile)
