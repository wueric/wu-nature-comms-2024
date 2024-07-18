import argparse
import itertools
import pickle
from collections import namedtuple
from typing import List, Callable, Sequence

import numpy as np
import torch
import tqdm

import gaussian_denoiser.denoiser_wrappers as denoiser_wrappers
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.dynamic_data_util import FrameRateJitteredMovieBatchDataloader
from dejitter_recons.estimate_image import compute_ground_truth_eye_movements, \
    noreduce_nomask_batch_bin_bernoulli_neg_LL, estimate_frame_rate_lnp_with_fixed_eye_movements
from denoise_inverse_alg.hqs_alg import BatchParallel_DirectSolve_HQS_ZGenerator, \
    BatchParallel_Adam_HQS_XGenerator, AdamOptimParams, ScheduleVal, make_logspaced_rho_schedule
from denoise_inverse_alg.poisson_inverse_alg import reinflate_cropped_lnp_model, PackedLNPTensors
from eval_fns.eval import MaskedMSELoss, SSIM, MS_SSIM
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from optimization_encoder.trial_glm import load_fitted_lnp_families

GridSearchParams = namedtuple('GridSearchParams', ['lambda_start', 'lambda_end', 'prior_weight'])
GridSearchReconstructions = namedtuple('GridSearchReconstructions',
                                       ['ground_truth', 'reconstructions', 'mse', 'ssim', 'ms_ssim'])

BATCH_SIZE = 8
HQS_MAX_ITER = 10
GRID_SEARCH_TEST_N_IMAGES = 32


def batch_parallel_frame_rate_LNP_jittered_movie_hqs_grid_search(
        frame_rate_jitter_dataset: FrameRateJitteredMovieBatchDataloader,
        indices_to_get: Sequence[int],
        batch_size: int,
        packed_lnp_tensors: PackedLNPTensors,
        valid_region_mask: np.ndarray,
        reconstruction_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        image_to_metric_callable: Callable[[torch.Tensor], torch.Tensor],
        mse_module: MaskedMSELoss,
        ssim_module: SSIM,
        ms_ssim_module: MS_SSIM,
        grid_lambda_start: np.ndarray,
        grid_lambda_end: np.ndarray,
        grid_prior: np.ndarray,
        device: torch.device,
        use_computed_eye_movements: bool = True,
        max_iter=HQS_MAX_ITER):

    _, height, width = packed_lnp_tensors.spatial_filters.shape

    n_examples = len(indices_to_get)
    assert n_examples % batch_size == 0, 'number of example images must be multiple of batch size'

    ################################################################################
    # first load the unblind denoiser
    unblind_denoiser_model = denoiser_wrappers.load_masked_drunet_unblind_denoiser(device)
    unblind_denoiser_callable = denoiser_wrappers.make_unblind_apply_dpir_denoiser_with_mask(
        unblind_denoiser_model,
        (-1.0, 1.0), (0.0, 255))

    grid_search_pbar = tqdm.tqdm(
        total=grid_prior.shape[0] * grid_lambda_start.shape[0] * grid_lambda_end.shape[0])

    ret_dict = {}
    for ix, grid_params in enumerated_product(grid_prior, grid_lambda_start, grid_lambda_end):
        prior_weight, lambda_start, lambda_end = grid_params

        output_dict_key = GridSearchParams(lambda_start, lambda_end, prior_weight)
        schedule_rho = make_logspaced_rho_schedule(ScheduleVal(lambda_start, lambda_end, prior_weight, max_iter))

        output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
        example_stimuli_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
        pbar = tqdm.tqdm(total=n_examples)
        for low in range(0, n_examples, batch_size):
            high = low + batch_size
            things_to_get = indices_to_get[low:high]

            history_f, target_f, binned_spikes = frame_rate_jitter_dataset[things_to_get]

            n_frames_history = history_f.shape[1]
            n_frames_target = target_f.shape[1]
            n_frames_total = n_frames_target + n_frames_history

            eye_movements = np.zeros((batch_size, n_frames_target, 2), dtype=np.int64)
            if use_computed_eye_movements:
                for ii in range(batch_size):
                    jitter_coords, diff_norms = compute_ground_truth_eye_movements(target_f[ii, ...], 15, device)
                    eye_movements[ii, ...] = jitter_coords

            adam_solver_gen = BatchParallel_Adam_HQS_XGenerator(
                [AdamOptimParams(50, 1e-1), AdamOptimParams(25, 1e-1), AdamOptimParams(25, 1e-1)],
                default_params=AdamOptimParams(25, 5e-2))

            z_solver_gen = BatchParallel_DirectSolve_HQS_ZGenerator()

            estim_image = estimate_frame_rate_lnp_with_fixed_eye_movements(
                packed_lnp_tensors,
                unblind_denoiser_callable,
                history_f,
                n_frames_total,
                binned_spikes,
                valid_region_mask,
                eye_movements,
                np.ones((batch_size, binned_spikes.shape[2])),
                reconstruction_loss_fn,
                schedule_rho,
                prior_weight,
                adam_solver_gen,
                z_solver_gen,
                device,
                return_intermediate_images=False,
                solver_verbose=False,
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



def enumerated_product(*args):
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Grid search HQS hyperparameters for eye movements reconstructions, using LNP encoding and dCNN image prior")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('model_cfg_path', type=str, help='path to YAML specifying where the GLM fits are')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-ls', '--lambda_start', nargs=3,
                        help='Grid search parameters for HQS prox start. Start, end, n_grid')
    parser.add_argument('-le', '--lambda_end', nargs=3,
                        help='Grid search parameters for HQS prox end. Start, end, n_grid')
    parser.add_argument('-p', '--prior', type=float, nargs='+',
                        help='Grid search parameters for prior lambda. Specify weights explicitly')
    parser.add_argument('-e', '--eye_movements', action='store_true', default=False,
                        help='Use exact computed eye movements. Default guess 0')
    parser.add_argument('-i', '--max_iter', type=int, default=5,
                        help='HQS maximum iterations')
    parser.add_argument('-j', '--jitter_time', type=float, default=0.0,
                        help='number of electrical samples to jitter spikes in time by (SD of Gaussian)')
    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    grid_search_lambda_start = np.logspace(float(args.lambda_start[0]), float(args.lambda_start[1]),
                                           int(args.lambda_start[2]))
    grid_search_lambda_end = np.logspace(float(args.lambda_end[0]), float(args.lambda_end[1]),
                                         int(args.lambda_end[2]))
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

    jitter_dataloader = FrameRateJitteredMovieBatchDataloader(
        loaded_synchronized_datasets,
        cells_ordered,
        ddu.PartitionType.TEST_PARTITION,
        image_rescale_lambda=image_rescale_lambda,
        crop_w_ix=(crop_width_low, 320 - crop_width_low),  # FIXME
        time_jitter_spikes=args.jitter_time
    )

    print(len(jitter_dataloader))

    # FIXME see if we can get a better way to do this
    height, width = 160, (320 - crop_width_low - crop_width_high)

    ###########################################################################
    # Load the GLM model parameters
    fitted_lnp_paths = parse_prefit_glm_paths(args.model_cfg_path)
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
    # Do the grid search
    grid_search_output_dict = batch_parallel_frame_rate_LNP_jittered_movie_hqs_grid_search(
        jitter_dataloader,
        np.r_[0:GRID_SEARCH_TEST_N_IMAGES],
        BATCH_SIZE,
        packed_lnp_tensors,
        convex_hull_valid_mask,
        noreduce_nomask_batch_bin_bernoulli_neg_LL,
        image_to_metric_lambda,
        masked_mse_module,
        ssim_module,
        ms_ssim_module,
        grid_search_lambda_start,
        grid_search_lambda_end,
        grid_search_prior,
        device,
        use_computed_eye_movements=args.eye_movements,
        max_iter=args.max_iter,
    )

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(grid_search_output_dict, pfile)
