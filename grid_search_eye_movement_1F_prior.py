import argparse
import pickle
from collections import namedtuple
from typing import List, Callable, Sequence, Optional

import numpy as np
import torch
import tqdm

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from convex_optim_base.unconstrained_optim import FistaSolverParams
from lib.data_utils.dynamic_data_util import JitteredMovieBatchDataloader
from dejitter_recons.estimate_image import compute_ground_truth_eye_movements, \
    noreduce_nomask_batch_bin_bernoulli_neg_LL, construct_magic_rescale_const, \
    estimate_1F_image_with_fixed_eye_movements
from dejitter_recons.joint_em_estimation import batched_time_mask_history_and_frame_transition
from denoise_inverse_alg.glm_inverse_alg import PackedGLMTensors, reinflate_cropped_glm_model
from eval_fns.eval import MaskedMSELoss, SSIM, MS_SSIM
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from optimization_encoder.trial_glm import load_fitted_glm_families
from simple_priors.gaussian_prior import make_zca_gaussian_prior_matrix

GridSearchParams1F = namedtuple('GridSearchParams1F', ['prior_weight'])
GridSearchReconstructions = namedtuple('GridSearchReconstructions',
                                       ['ground_truth', 'reconstructions', 'mse', 'ssim', 'ms_ssim'])

BATCH_SIZE = 8
GRID_SEARCH_TEST_N_IMAGES = 80


def batch_parallel_1F_eye_movement_grid_search(
        jittered_movie_dataloader: JitteredMovieBatchDataloader,
        indices_to_get: Sequence[int],
        batch_size: int,
        samples_per_bin: int,
        packed_glm_tensors: PackedGLMTensors,
        reconstruction_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        image_to_metric_callable: Callable[[torch.Tensor], torch.Tensor],
        patch_zca_matrix: np.ndarray,
        mse_module: MaskedMSELoss,
        ssim_module: SSIM,
        ms_ssim_module: MS_SSIM,
        grid_prior: np.ndarray,
        device: torch.device,
        fista_solver_params: FistaSolverParams = FistaSolverParams(
            initial_learning_rate=1.0,
            max_iter=250,
            converge_epsilon=1e-6,
            backtracking_beta=0.5),
        patch_prior_stride: int = 1,
        use_computed_eye_movements: bool = True,
        integration_num_samples: Optional[int] = None):
    _, height, width = packed_glm_tensors.spatial_filters.shape

    n_examples = len(indices_to_get)
    assert n_examples % batch_size == 0, 'number of example images must be multiple of batch size'

    use_subset_time_integration = (integration_num_samples is not None)

    grid_search_pbar = tqdm.tqdm(total=grid_prior.shape[0])

    ret_dict = {}
    for ix, prior_weight in enumerate(grid_prior):

        output_dict_key = GridSearchParams1F(prior_weight)

        output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
        example_stimuli_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
        pbar = tqdm.tqdm(total=n_examples)

        for low in range(0, n_examples, batch_size):
            high = low + batch_size
            things_to_get = indices_to_get[low:high]

            history_f, target_f, f_transitions, spike_bins, binned_spikes = jittered_movie_dataloader[things_to_get]

            n_frames_history = history_f.shape[1]

            eye_movements = np.zeros((batch_size, n_frames_history, 2), dtype=np.int64)
            if use_computed_eye_movements:
                for ii in range(batch_size):
                    jitter_coords, diff_norms = compute_ground_truth_eye_movements(target_f[ii, ...], 15, device)
                    eye_movements[ii, ...] = jitter_coords

            if use_subset_time_integration:
                history_f, f_transitions, binned_spikes, spike_bins = batched_time_mask_history_and_frame_transition(
                    history_f,
                    f_transitions,
                    spike_bins,
                    binned_spikes,
                    integration_num_samples
                )

                n_target_eye_movements = f_transitions.shape[1] - n_frames_history - 1
                eye_movements = eye_movements[:, :n_target_eye_movements, :]

            magic_const_to_use = construct_magic_rescale_const(integration_num_samples / samples_per_bin) if \
                use_subset_time_integration else construct_magic_rescale_const(500.0)
            estim_image = estimate_1F_image_with_fixed_eye_movements(
                packed_glm_tensors,
                patch_zca_matrix,
                history_f,
                f_transitions,
                binned_spikes,
                spike_bins,
                eye_movements,
                np.ones((batch_size, binned_spikes.shape[2])),
                reconstruction_loss_fn,
                prior_weight,
                device,
                solver_verbose=True,
                fista_solver_params=fista_solver_params,
                patch_stride=patch_prior_stride,
                magic_rescale_const=magic_const_to_use
            )

            output_image_buffer_np[low:high, ...] = estim_image
            example_stimuli_buffer_np[low:high, ...] = target_f[:, 0, :, :]
            pbar.update(batch_size)

        pbar.close()
        output_image_buffer = torch.tensor(output_image_buffer_np, dtype=torch.float32, device=device)
        example_stimuli_buffer = torch.tensor(example_stimuli_buffer_np, dtype=torch.float32, device=device)

        # now compute SSIM, MS-SSIM, and masked MSE
        reconstruction_rescaled = image_to_metric_callable(output_image_buffer)
        rescaled_example_stimuli = image_to_metric_callable(example_stimuli_buffer)
        masked_mse = torch.mean(mse_module(reconstruction_rescaled, rescaled_example_stimuli)).item()
        ssim_val = ssim_module(rescaled_example_stimuli[:, None, :, :],
                               reconstruction_rescaled[:, None, :, :]).item()
        ms_ssim_val = ms_ssim_module(rescaled_example_stimuli[:, None, :, :],
                                     reconstruction_rescaled[:, None, :, :]).item()

        ret_dict[output_dict_key] = GridSearchReconstructions(
            example_stimuli_buffer_np,
            output_image_buffer_np,
            masked_mse,
            ssim_val,
            ms_ssim_val)

        print(f"{output_dict_key}, MSE {masked_mse}, SSIM {ssim_val}, MS-SSIM {ms_ssim_val}")

        del output_image_buffer, rescaled_example_stimuli
        grid_search_pbar.update(1)

    return ret_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Grid search HQS hyperparameters for eye movements reconstructions, using LNBRC encoding and 1/F Gaussian image prior")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('model_cfg_path', type=str, help='path to YAML specifying where the GLM fits are')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-p', '--prior', type=float, nargs='+',
                        help='Grid search parameters for prior lambda. Specify weights explicitly')
    parser.add_argument('-e', '--eye_movements', action='store_true', default=False,
                        help='Use exact computed eye movements. Default guess 0')
    parser.add_argument('-mS', '--max_samples', type=int, default=None,
                        help='maximum time samples after initial stimulus presentation to use')
    parser.add_argument('-j', '--jitter_time', type=float, default=0.0,
                        help='number of electrical samples to jitter spikes in time by (SD of Gaussian)')
    parser.add_argument('-d', '--patchdim', type=int, default=64, help='Gaussian prior patch dimension')
    parser.add_argument('-s', '--patchstride', type=int, default=1, help='Gaussian prior patch dimension')
    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    grid_search_prior = np.array(args.prior)

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
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)
        blurred_stas_by_type = pickle.load(pfile)

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)
    image_to_metric_lambda = du.make_torch_transform_to_recons_metric_lambda(image_rescale_low, image_rescale_high)

    ################################################################
    # Load the natural scenes Vision datasets and determine what the
    # train and test partitions are
    nscenes_dataset_info_list = config_settings['NScenesMovieDatasets']

    create_test_dataset = (dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]

    loaded_synchronized_datasets = ddu.load_jittered_nscenes_dataset_and_timebin(
        nscenes_dataset_info_list,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
    )

    jitter_dataloader = JitteredMovieBatchDataloader(
        loaded_synchronized_datasets,
        cells_ordered,
        ddu.PartitionType.TEST_PARTITION,
        samples_per_bin,
        crop_w_ix=(32, 320 - 32),  # FIXME,
        image_rescale_lambda=image_rescale_lambda,
        time_jitter_spikes=args.jitter_time
    )

    #####################################################
    # Load the GLMs
    fitted_glm_paths = parse_prefit_glm_paths(args.model_cfg_path)
    fitted_glm_families = load_fitted_glm_families(fitted_glm_paths)
    packed_glm_tensors = reinflate_cropped_glm_model(
        fitted_glm_families,
        bounding_boxes_by_type,
        cells_ordered,
        160, 256,  # FIXME
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

    gaussian_zca_mat = make_zca_gaussian_prior_matrix((args.patchdim, args.patchdim),
                                                      dc_multiple=1.0)
    gaussian_zca_mat_imshape = (gaussian_zca_mat.reshape((args.patchdim, args.patchdim, args.patchdim, args.patchdim)))


    grid_search_output_dict = batch_parallel_1F_eye_movement_grid_search(
        jitter_dataloader,
        np.r_[0:GRID_SEARCH_TEST_N_IMAGES],
        BATCH_SIZE,
        samples_per_bin,
        packed_glm_tensors,
        noreduce_nomask_batch_bin_bernoulli_neg_LL,
        image_to_metric_lambda,
        gaussian_zca_mat_imshape,
        masked_mse_module,
        ssim_module,
        ms_ssim_module,
        grid_search_prior,
        device,
        FistaSolverParams(
            initial_learning_rate=1.0,
            max_iter=250,
            converge_epsilon=1e-6,
            backtracking_beta=0.5),
        patch_prior_stride=args.patchstride,
        use_computed_eye_movements=args.eye_movements,
        integration_num_samples=args.max_samples
    )

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(grid_search_output_dict, pfile)
