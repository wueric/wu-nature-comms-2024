import argparse
import pickle
from collections import namedtuple
from typing import Dict, List, Any, Iterator

import numpy as np
import torch

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from denoise_inverse_alg.hqs_alg import BatchParallel_HQS_XGenerator, \
    BatchParallel_DirectSolve_HQS_ZGenerator, HQS_ParameterizedSolveFn
from denoise_inverse_alg.poisson_inverse_alg import reinflate_cropped_lnp_model, poisson_loss_noreduce
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from optimization_encoder.trial_glm import load_fitted_lnp_families
from reconstruction_fns.flashed_reconstruction_toplevel_fns import batch_parallel_generate_flashed_hqs_reconstructions

N_BINS_BEFORE = 30  # 250 ms, or 30 frames at 120 Hz
N_BINS_AFTER = 18  # 150 ms, or 18 frames at 120 Hz
SAMPLES_PER_BIN = 8.3333333 * 20  # 120 Hz -> 8.333 ms period


GridSearchParams = namedtuple('GridSearchParams', ['lambda_start', 'lambda_end', 'prior_weight', 'max_iter'])


LinearModelBinningRange = namedtuple('LinearModelBinningRange', ['start_cut', 'end_cut'])


def make_HQS_X_prob_solver_generator_fn() -> Iterator[HQS_ParameterizedSolveFn]:
    return BatchParallel_HQS_XGenerator(first_niter=300, subsequent_niter=300)


def make_HQS_Z_prob_solver_generator_fn() -> Iterator[HQS_ParameterizedSolveFn]:
    return BatchParallel_DirectSolve_HQS_ZGenerator()


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Generate flashed image reconstructions using the LNP encoding model and dCNN prior")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('yaml_path', type=str, help='path to YAML specifying encoding model weights')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-n', '--noise_init', type=float, default=1e-3)
    parser.add_argument('-l', '--linear_init', type=str, default=None,
                        help='optional, path to linear model if using linear model to initialize HQS')
    parser.add_argument('-bb', '--before_lin_bins', type=int, default=0, help='start binning for linear')
    parser.add_argument('-aa', '--after_lin_bins', type=int, default=0, help='end binning for linear')
    parser.add_argument('-hh', '--heldout', action='store_true', default=False, help='generate heldout images')
    parser.add_argument('-st', '--start', type=float, default=0.1, help='HQS rho first value')
    parser.add_argument('-en', '--end', type=float, default=10.0, help='HQS rho last value')
    parser.add_argument('-i', '--iter', type=int, default=25, help='HQS num outer iters')
    parser.add_argument('-lam', '--prior_lambda', type=float, default=0.2, help='HQS prior weight')
    parser.add_argument('-m', '--use_mask', action='store_true', default=False, help='use valid mask')
    parser.add_argument('-b', '--batch', type=int, default=16,
                        help='maximum batch size')

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

    ###############################################################
    # Load and optionally downsample/crop the stimulus frames
    test_heldout_nscenes_patch_list = ddu.preload_bind_get_flashed_patches(
        nscenes_dset_list,
        ddu.PartitionType.HELDOUT_PARTITION if args.heldout else ddu.PartitionType.TEST_PARTITION
    )

    test_heldout_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(
        test_heldout_nscenes_patch_list))

    _, height, width = test_heldout_frames.shape

    ################################################################
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

    # we only do reconstructions from single-bin data here
    # use a separate script to do GLM reconstructions once we get those going
    response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        test_heldout_nscenes_patch_list,
    )

    hyperparameters = GridSearchParams(args.start, args.end, args.prior_lambda, args.iter)

    # Set up linear model if we choose to use it for initialization
    linear_model_param = None
    if args.linear_init is not None:
        linear_decoder = torch.load(args.linear_init, map_location=device)
        linear_bin_cutoffs = LinearModelBinningRange(args.before_lin_bins, args.after_lin_bins)
        linear_model_param = (linear_decoder, linear_bin_cutoffs)

    target_reconstructions = batch_parallel_generate_flashed_hqs_reconstructions(
        response_vector,
        (image_rescale_low, image_rescale_high),
        packed_lnp_tensors,
        poisson_loss_noreduce,
        nscenes_dset_list[0].stimulus_time_component,
        hyperparameters,
        make_HQS_X_prob_solver_generator_fn,
        make_HQS_Z_prob_solver_generator_fn,
        args.batch,
        device,
        initialize_noise_level=args.noise_init,
        initialize_linear_model=linear_model_param,
        valid_region_mask=convex_hull_valid_mask
    )

    with open(args.save_path, 'wb') as pfile:
        save_data = {
            'ground_truth': test_heldout_frames,
            'glm_cropped': target_reconstructions
        }

        pickle.dump(save_data, pfile)

    print('done')
