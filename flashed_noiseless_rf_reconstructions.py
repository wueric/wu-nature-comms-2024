import argparse
import pickle
from collections import namedtuple
from typing import Dict, List, Any, Iterator

import numpy as np
import torch

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from denoise_inverse_alg.glm_inverse_alg import reinflate_cropped_glm_model
from denoise_inverse_alg.hqs_alg import BatchParallel_DirectSolve_HQS_ZGenerator, HQS_ParameterizedSolveFn, \
    BatchParallel_DirectSolve_HQS_XGenerator
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from optimization_encoder.trial_glm import load_fitted_glm_families
from reconstruction_fns.flashed_reconstruction_toplevel_fns import \
    batch_parallel_generated_noiseless_linear_hqs_reconstructions

GridSearchParams = namedtuple('GridSearchParams', ['lambda_start', 'lambda_end', 'prior_weight', 'max_iter'])


def retrieve_flashed_stimulus_frames(config_settings: Dict[str, Any],
                                     partition: ddu.PartitionType) -> np.ndarray:
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    n_bins_before = config_settings[dcp.TimebinningSection.NBINS_BEFORE_TRANS]
    n_bins_after = config_settings[dcp.TimebinningSection.NBINS_AFTER_TRANS]
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

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

    for item in nscenes_dset_list:
        item.load_frames_from_disk()

    ###############################################################
    # Load and optionally downsample/crop the stimulus frames
    test_heldout_nscenes_patch_list = ddu.preload_bind_get_flashed_patches(
        nscenes_dset_list,
        partition
    )

    test_heldout_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(
        test_heldout_nscenes_patch_list))

    return test_heldout_frames


def retrieve_eye_movements_single_frames(config_settings: Dict[str, Any],
                                         partition: ddu.PartitionType,
                                         cells_ordered: OrderedMatchedCellsStruct) -> np.ndarray:
    ################################################################
    # Load some of the model fit parameters
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

    create_test_dataset = (dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]

    print(heldout_dataset_movie_blocks)

    nscenes_dataset_info_list = config_settings[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]

    loaded_synchronized_datasets = ddu.load_jittered_nscenes_dataset_and_timebin(
        nscenes_dataset_info_list,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
    )

    jitter_dataloader = ddu.JitteredMovieBatchDataloader(
        loaded_synchronized_datasets,
        cells_ordered,
        partition,
        samples_per_bin,
        image_rescale_lambda=image_rescale_lambda,
        crop_w_ix=(crop_width_low, 320 - crop_width_low),  # FIXME
    )

    target_image_buffer = []
    for low in range(0, len(jitter_dataloader)):
        _, target_frames, _q, _r, _c = jitter_dataloader[low]
        target_image_buffer.append(target_frames[0, ...])

    # shape (n_images, height, width)
    return np.stack(target_image_buffer, axis=0)


def apply_linear_projections_gpu(linear_filters: np.ndarray,
                                 images: np.ndarray,
                                 device: torch.device,
                                 dtype: torch.dtype = torch.float32) -> np.ndarray:
    '''
    linear_filters: shape (n_cells, height, width)
    images: shape (n_images, height, width)

    returns shape (n_images, n_cells)
    '''

    n_images, height, width = images.shape
    n_cells = linear_filters.shape[0]
    with torch.no_grad():
        # shape (n_cells, height, width)
        linear_filters_torch = torch.tensor(linear_filters, dtype=dtype,
                                            device=device)

        # shape (n_images, height, width)
        images_torch = torch.tensor(images, dtype=dtype,
                                    device=device)

        # shape (n_images, height * width)
        images_flat_torch = images_torch.reshape(n_images, -1)

        # shape (n_cells, height * width)
        linear_filters_flat_torch = linear_filters_torch.reshape(n_cells, -1)

        # shape (n_images, height * width) @ (height * width, n_cells)
        projected = images_flat_torch @ linear_filters_flat_torch.T

        return projected.cpu().numpy()


def make_linear_noiseless_HQS_X_prob_solver_generator_fn() -> Iterator[HQS_ParameterizedSolveFn]:
    return BatchParallel_DirectSolve_HQS_XGenerator()


def make_linear_noiseless_HQS_Z_prob_solver_generator_fn() -> Iterator[HQS_ParameterizedSolveFn]:
    return BatchParallel_DirectSolve_HQS_ZGenerator()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        "Mass-generate noiseless reconstructions, using LNBRC linear filters normalized to 1")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('yaml_path', type=str, help='path to YAML specifying encoding model weights')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-n', '--noise_init', type=float, default=1e-3)
    parser.add_argument('-hh', '--heldout', action='store_true', default=False, help='generate heldout images')
    parser.add_argument('-st', '--start', type=float, default=0.1, help='HQS rho first value')
    parser.add_argument('-en', '--end', type=float, default=10.0, help='HQS rho last value')
    parser.add_argument('-i', '--iter', type=int, default=25, help='HQS num outer iters')
    parser.add_argument('-lam', '--prior_lambda', type=float, default=0.2, help='HQS prior weight')
    parser.add_argument('-m', '--use_mask', action='store_true', default=False, help='use valid mask')
    parser.add_argument('-b', '--batch', type=int, default=16,
                        help='maximum batch size')
    parser.add_argument('-r', '--full_resolution', action='store_true', default=False,
                        help='Set if the GLMs pointed to by the .yaml file are full resolution')
    parser.add_argument('-f', '--fixational', action='store_true', default=False,
                        help='is fixational eye movements stimulus')

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

    #####################################################################
    # load the stimulus
    # different for flashed or jittered
    if args.fixational:
        target_frames = retrieve_eye_movements_single_frames(
            config_settings,
            ddu.PartitionType.HELDOUT_PARTITION if args.heldout else ddu.PartitionType.TEST_PARTITION,
            cells_ordered
        )
    else:
        target_frames = retrieve_flashed_stimulus_frames(
            config_settings,
            ddu.PartitionType.HELDOUT_PARTITION if args.heldout else ddu.PartitionType.TEST_PARTITION
        )

    n_images, height, width = target_frames.shape

    #####################################################################
    # get the models, extract the linear filters, renormalize them to 1
    fitted_glm_paths = parse_prefit_glm_paths(args.yaml_path)
    fitted_glm_families = load_fitted_glm_families(fitted_glm_paths)

    packed_glm_tensors = reinflate_cropped_glm_model(
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

    linear_filters = packed_glm_tensors.spatial_filters.reshape(-1, height * width)
    linear_filters = linear_filters / np.linalg.norm(linear_filters, axis=-1, keepdims=True)
    linear_filters = linear_filters.reshape(-1, height, width)

    #####################################################################
    # perform the linear projections
    # shape (n_images, n_cells)
    linearly_projected = apply_linear_projections_gpu(linear_filters,
                                                      target_frames,
                                                      device)

    ######################################################################
    # perform the reconstructions
    hyperparameters = GridSearchParams(args.start, args.end, args.prior_lambda, args.iter)

    reconstructions = batch_parallel_generated_noiseless_linear_hqs_reconstructions(
        linearly_projected,
        (image_rescale_low, image_rescale_high),
        linear_filters,
        hyperparameters,
        make_linear_noiseless_HQS_X_prob_solver_generator_fn,
        make_linear_noiseless_HQS_Z_prob_solver_generator_fn,
        args.batch,
        device,
        initialize_noise_level=args.noise_init,
        valid_region_mask=convex_hull_valid_mask
    )

    with open(args.save_path, 'wb') as pfile:
        save_data = {
            'ground_truth': target_frames,
            'linear_noiseless': reconstructions
        }

        pickle.dump(save_data, pfile)

    print('done')

