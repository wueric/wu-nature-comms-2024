import argparse
import pickle
from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from convex_optim_base.unconstrained_optim import batch_parallel_unconstrained_solve, FistaSolverParams
from denoise_inverse_alg.glm_inverse_alg import BatchParallelPatchGaussian1FPriorGLMReconstruction, \
    reinflate_cropped_glm_model
from denoise_inverse_alg.glm_inverse_alg import PackedGLMTensors, \
    make_full_res_packed_glm_tensors, noreduce_bernoulli_neg_LL
from eval_fns.eval import MaskedMSELoss, MS_SSIM, SSIM
from generate_cropped_glm_hqs_reconstructions import make_glm_stim_time_component
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from optimization_encoder.trial_glm import load_fitted_glm_families
from reconstruction_fns.grid_search_types import GridSearchReconstructions
from simple_priors.gaussian_prior import make_zca_gaussian_prior_matrix


GridSearchParams1F = namedtuple('GridSearchParams1F', ['prior_weight'])


BATCH_SIZE = 8
GRID_SEARCH_TEST_N_IMAGES = 80


def image_rescale_0_1(images_min1_max1: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        images_rescaled = torch.clamp((images_min1_max1 + 1.0) / 2.0,
                                      min=0.0, max=1.0)
        return images_rescaled


def batch_parallel_1F_glm_grid_search(example_spikes: np.ndarray,
                                      example_stimuli: np.ndarray,
                                      batch_size: int,
                                      packed_glm_tensors: PackedGLMTensors,
                                      glm_time_component: np.ndarray,
                                      prior_weights: List[float],
                                      patch_dimensions: Tuple[int, int],
                                      mse_module: MaskedMSELoss,
                                      ssim_module: SSIM,
                                      ms_ssim_module: MS_SSIM,
                                      device: torch.device,
                                      patch_stride: int = 2) \
        -> Dict[GridSearchParams1F, GridSearchReconstructions]:
    n_examples, n_cells, n_bins_observed = example_spikes.shape
    _, height, width = packed_glm_tensors.spatial_filters.shape

    assert n_examples % batch_size == 0, 'number of example images must be multiple of batch size'

    example_stimuli_torch = torch.tensor(example_stimuli, dtype=torch.float32, device=device)
    example_spikes_torch = torch.tensor(example_spikes, dtype=torch.float32, device=device)

    rescaled_example_stimuli = image_rescale_0_1(example_stimuli_torch)
    del example_stimuli_torch

    p_height, p_width = patch_dimensions

    gaussian_zca_mat = make_zca_gaussian_prior_matrix(patch_dimensions,
                                                      dc_multiple=1.0)
    gaussian_zca_mat_imshape = (gaussian_zca_mat.reshape((p_height, p_width, p_height, p_width)))

    grid_search_pbar = tqdm.tqdm(total=len(prior_weights))
    ret_dict = {}
    for lambda_1f in prior_weights:

        output_dict_key = GridSearchParams1F(lambda_1f)

        batch_gaussian_problem = BatchParallelPatchGaussian1FPriorGLMReconstruction(
            batch_size,
            gaussian_zca_mat_imshape,
            lambda_1f,
            packed_glm_tensors,
            glm_time_component,
            noreduce_bernoulli_neg_LL,
            patch_stride=patch_stride
        ).to(device)

        ################################################################################
        output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
        pbar = tqdm.tqdm(total=n_examples)

        for low in range(0, n_examples, batch_size):
            high = low + batch_size

            glm_trial_spikes_torch = example_spikes_torch[low:high, ...]
            batch_gaussian_problem.precompute_gensig_components(glm_trial_spikes_torch)

            _ = batch_parallel_unconstrained_solve(
                batch_gaussian_problem,
                FistaSolverParams(
                    initial_learning_rate=1.0,
                    max_iter=250,
                    converge_epsilon=1e-6,
                    backtracking_beta=0.5
                ),
                verbose=False,
                observed_spikes=glm_trial_spikes_torch,
            )

            reconstructed_image = batch_gaussian_problem.get_reconstructed_image()
            output_image_buffer_np[low:high, :, :] = reconstructed_image

            pbar.update(batch_size)

        pbar.close()

        output_image_buffer = torch.tensor(output_image_buffer_np, dtype=torch.float32, device=device)

        # now compute SSIM, MS-SSIM, and masked MSE
        reconstruction_rescaled = image_rescale_0_1(output_image_buffer)
        masked_mse = torch.mean(mse_module(reconstruction_rescaled, rescaled_example_stimuli)).item()
        ssim_val = ssim_module(rescaled_example_stimuli[:, None, :, :],
                               reconstruction_rescaled[:, None, :, :]).item()
        ms_ssim_val = ms_ssim_module(rescaled_example_stimuli[:, None, :, :],
                                     reconstruction_rescaled[:, None, :, :]).item()

        ret_dict[output_dict_key] = GridSearchReconstructions(
            example_stimuli,
            output_image_buffer_np,
            masked_mse,
            ssim_val,
            ms_ssim_val)

        print(f"{output_dict_key}, MSE {masked_mse}, SSIM {ssim_val}, MS-SSIM {ms_ssim_val}")

        del output_image_buffer, batch_gaussian_problem, reconstruction_rescaled
        grid_search_pbar.update(1)

    return ret_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Grid search for hyperparameters for flashed image reconstruction using LNBRC encoding and 1/F Gaussian image prior")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('yaml_path', type=str, help='path to YAML specifying encoding model weights')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-p', '--prior', type=float, nargs='+',
                        help='Grid search parameters for prior lambda. Specify weights explicitly')
    parser.add_argument('-d', '--patchdim', type=int, default=64, help='Gaussian prior patch dimension')
    parser.add_argument('-s', '--patchstride', type=int, default=1, help='Gaussian prior patch dimension')
    parser.add_argument('-r', '--full_resolution', action='store_true', default=False,
                        help='Set if the GLMs pointed to by the .yaml file are full resolution')

    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

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
        ddu.PartitionType.TEST_PARTITION
    )

    test_heldout_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(
        test_heldout_nscenes_patch_list))

    _, height, width = test_heldout_frames.shape

    glm_stim_time_component = make_glm_stim_time_component(config_settings)

    #######################################################
    fitted_glm_paths = parse_prefit_glm_paths(args.yaml_path)
    fitted_glm_families = load_fitted_glm_families(fitted_glm_paths)

    if not args.full_resolution:

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
    else:
        packed_glm_tensors = make_full_res_packed_glm_tensors(
            cells_ordered,
            fitted_glm_families
        )

    # we only do reconstructions from single-bin data here
    # use a separate script to do GLM reconstructions once we get those going
    response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        test_heldout_nscenes_patch_list,
    )

    gaussian_zca_mat = make_zca_gaussian_prior_matrix((args.patchdim, args.patchdim),
                                                      dc_multiple=1.0)
    gaussian_zca_mat_imshape = (gaussian_zca_mat.reshape((args.patchdim, args.patchdim,
                                                          args.patchdim, args.patchdim)))

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
    # Do the grid search
    grid_search_output_dict = batch_parallel_1F_glm_grid_search(
        response_vector[:GRID_SEARCH_TEST_N_IMAGES, ...],
        test_heldout_frames[:GRID_SEARCH_TEST_N_IMAGES, ...],
        BATCH_SIZE,
        packed_glm_tensors,
        glm_stim_time_component,
        list(args.prior),
        (args.patchdim, args.patchdim),
        masked_mse_module,
        ssim_module,
        ms_ssim_module,
        device,
        patch_stride=args.patchstride
    )

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(grid_search_output_dict, pfile)
