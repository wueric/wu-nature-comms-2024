import argparse
import pickle
from typing import List, Tuple, Optional, Callable, Union

import numpy as np
import torch
import tqdm

import gaussian_denoiser.denoiser_wrappers as denoiser_wrappers
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.dynamic_data_util import RepeatsJitteredMovieDataloader, ShuffledRepeatsJitteredMovieDataloader
from dejitter_recons.estimate_image import noreduce_nomask_batch_bin_bernoulli_neg_LL
from dejitter_recons.joint_em_estimation import non_online_joint_em_estimation2, create_gaussian_multinomial
from denoise_inverse_alg.glm_inverse_alg import PackedGLMTensors, \
    reinflate_cropped_glm_model, reinflate_cropped_fb_only_glm_model, FeedbackOnlyPackedGLMTensors
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask

from generate_joint_eye_movements_reconstructions import load_fitted_glm_families, make_get_iterators


def data_repeats_generate_joint_eye_movement_trajectory_reconstructions(
        packed_glm_tensors: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors],
        repeats_jitter_dataset: RepeatsJitteredMovieDataloader,
        bin_width: int,
        bin_width_delay: int,
        n_particles: int,
        gaussian_multinomial,
        prior_weight_lambda: float,
        eye_movement_weight_lambda: float,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        full_hqs_get_iterators: Callable,
        update_hqs_get_iterators: Callable,
        valid_region_mask: np.ndarray,
        device: torch.device,
        init_noise_sigma: Optional[float] = None,
        em_inner_opt_verbose=False,
        throwaway_log_prob=-6,
        compute_image_every_n=10) \
        -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[np.ndarray, np.ndarray]]]]:
    _, height, width = packed_glm_tensors.spatial_filters.shape

    ################################################################################
    # first load the unblind denoiser
    unblind_denoiser_model = denoiser_wrappers.load_masked_drunet_unblind_denoiser(device)
    unblind_denoiser_callable = denoiser_wrappers.make_unblind_apply_dpir_denoiser_with_mask(
        unblind_denoiser_model,
        (-1.0, 1.0), (0.0, 255))
    ################################################################################

    n_stimuli, n_repeats = len(repeats_jitter_dataset), repeats_jitter_dataset.num_repeats

    output_image_buffer = np.zeros((n_stimuli, n_repeats, height, width), dtype=np.float32)
    ground_truth_buffer = np.zeros((n_stimuli, height, width), dtype=np.float32)
    traj_weights_buffer = []  # type: List[List[Tuple[np.ndarray, np.ndarray]]]

    pbar = tqdm.tqdm(total=n_stimuli * n_repeats)
    for i in range(n_stimuli):
        traj_weights_buffer.append([])
        for j in range(n_repeats):

            history_frames, target_frames, snippet_transitions, spike_bins, binned_spikes = repeats_jitter_dataset[i, j]

            if init_noise_sigma is not None:
                image_init_guess = (np.random.randn(height, width) * init_noise_sigma).astype(np.float32)
            else:
                image_init_guess = np.zeros((height, width), dtype=np.float32)

            image, traj, weights = non_online_joint_em_estimation2(
                packed_glm_tensors,
                unblind_denoiser_callable,
                history_frames,
                snippet_transitions,
                binned_spikes,
                spike_bins,
                valid_region_mask,
                bin_width,
                bin_width_delay,  # FIXME check units; it's right
                n_particles,  # n_particles
                gaussian_multinomial,
                loss_fn,
                prior_weight_lambda,
                full_hqs_get_iterators,
                update_hqs_get_iterators,
                device,
                image_init_guess=image_init_guess,
                em_inner_opt_verbose=em_inner_opt_verbose,
                return_intermediate_em_results=False,
                throwaway_log_prob=throwaway_log_prob,
                likelihood_scale=eye_movement_weight_lambda,
                compute_image_every_n=compute_image_every_n
            )

            ground_truth_buffer[i, ...] = target_frames[0, ...]
            output_image_buffer[i, j, ...] = image
            traj_weights_buffer[-1].append((traj, weights))

            pbar.update(1)
    pbar.close()

    return ground_truth_buffer, output_image_buffer, traj_weights_buffer


def shuffled_repeats_generate_joint_eye_movement_trajectory_reconstructions(
        packed_glm_tensors: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors],
        shuff_repeats_jitter_dataset: ShuffledRepeatsJitteredMovieDataloader,
        n_shuffle_repeats: int,
        bin_width: int,
        bin_width_delay: int,
        n_particles: int,
        gaussian_multinomial,
        prior_weight_lambda: float,
        eye_movement_weight_lambda: float,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        full_hqs_get_iterators: Callable,
        update_hqs_get_iterators: Callable,
        valid_region_mask: np.ndarray,
        device: torch.device,
        init_noise_sigma: Optional[float] = None,
        em_inner_opt_verbose=False,
        throwaway_log_prob=-6,
        compute_image_every_n=10) \
        -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[np.ndarray, np.ndarray]]]]:
    _, height, width = packed_glm_tensors.spatial_filters.shape

    ################################################################################
    # first load the unblind denoiser
    unblind_denoiser_model = denoiser_wrappers.load_masked_drunet_unblind_denoiser(device)
    unblind_denoiser_callable = denoiser_wrappers.make_unblind_apply_dpir_denoiser_with_mask(
        unblind_denoiser_model,
        (-1.0, 1.0), (0.0, 255))
    ################################################################################

    n_stimuli = len(shuff_repeats_jitter_dataset)

    output_image_buffer = np.zeros((n_stimuli, n_shuffle_repeats, height, width), dtype=np.float32)
    ground_truth_buffer = np.zeros((n_stimuli, height, width), dtype=np.float32)
    traj_weights_buffer = []  # type: List[List[Tuple[np.ndarray, np.ndarray]]]

    pbar = tqdm.tqdm(total=n_stimuli * n_shuffle_repeats)
    for i in range(n_stimuli):
        traj_weights_buffer.append([])
        history_frames, target_frames, shuf_snippet_transitions, shuf_spike_bins, shuf_binned_spikes = \
            shuff_repeats_jitter_dataset[i]

        for j in range(n_shuffle_repeats):

            if init_noise_sigma is not None:
                image_init_guess = (np.random.randn(height, width) * init_noise_sigma).astype(np.float32)
            else:
                image_init_guess = np.zeros((height, width), dtype=np.float32)

            image, traj, weights = non_online_joint_em_estimation2(
                packed_glm_tensors,
                unblind_denoiser_callable,
                history_frames,
                shuf_snippet_transitions[j, ...],
                shuf_binned_spikes[j, ...],
                shuf_spike_bins[j, ...],
                valid_region_mask,
                bin_width,
                bin_width_delay,  # FIXME check units; it's right
                n_particles,  # n_particles
                gaussian_multinomial,
                loss_fn,
                prior_weight_lambda,
                full_hqs_get_iterators,
                update_hqs_get_iterators,
                device,
                image_init_guess=image_init_guess,
                em_inner_opt_verbose=em_inner_opt_verbose,
                return_intermediate_em_results=False,
                throwaway_log_prob=throwaway_log_prob,
                likelihood_scale=eye_movement_weight_lambda,
                compute_image_every_n=compute_image_every_n
            )

            ground_truth_buffer[i, ...] = target_frames[0, ...]
            output_image_buffer[i, j, ...] = image
            traj_weights_buffer[-1].append((traj, weights))

            pbar.update(1)
    pbar.close()

    return ground_truth_buffer, output_image_buffer, traj_weights_buffer


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate repeat reconstructions for the joint estimation of eye movements and image using LNBRC encoding and dCNN image prior")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('yaml_path', type=str, help='path to YAML specifying encoding model weights')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-st', '--start', type=float, default=0.1, help='HQS rho first value')
    parser.add_argument('-en', '--end', type=float, default=10.0, help='HQS rho last value')
    parser.add_argument('-i', '--iter', type=int, default=5, help='HQS num outer iters')
    parser.add_argument('-lam', '--prior_lambda', type=float, default=0.2, help='HQS prior weight')
    parser.add_argument('-eye', '--eye_movements', type=float, default=1.0, help='Eye movements term weight')
    parser.add_argument('-n', '--noise_init', type=float, default=None)
    parser.add_argument('-p', '--num_particles', type=int, default=10)
    parser.add_argument('-re', '--reestim_every', type=int, default=10)
    parser.add_argument('-sh', '--shuffle', type=int, default=0,
                        help='number of shuffled repeats to generate.' + \
                             ' If 0, uses real data')
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

    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)
        blurred_stas_by_type = pickle.load(pfile)

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

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

    #################################################################
    # Load the raw data
    create_test_dataset = (dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]

    nscenes_dataset_info_list = config_settings[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]

    loaded_synchronized_datasets = ddu.load_jittered_nscenes_dataset_and_timebin(
        nscenes_dataset_info_list,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
    )

    # FIXME see if we can get a better way to do this
    height, width = 160, (320 - crop_width_low - crop_width_high)

    ###########################################################################
    # Load the GLM model parameters
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

    gaussian_multinomial = create_gaussian_multinomial(1.2, 2)

    if args.shuffle == 0:

        repeat_jitter_dataloader = RepeatsJitteredMovieDataloader(
            loaded_synchronized_datasets,
            cells_ordered,
            samples_per_bin,
            crop_w_ix=(crop_width_low, 320 - crop_width_low),  # FIXME
            image_rescale_lambda=image_rescale_lambda,
        )

        ground_truth, reconstructions, eye_movement_trajectories = data_repeats_generate_joint_eye_movement_trajectory_reconstructions(
            packed_glm_tensors,
            repeat_jitter_dataloader,
            samples_per_bin,
            30 * samples_per_bin,
            args.num_particles,
            gaussian_multinomial,
            args.prior_lambda,
            args.eye_movements,
            noreduce_nomask_batch_bin_bernoulli_neg_LL,
            make_get_iterators(args.start, args.end, args.iter),
            make_get_iterators(args.end, args.end, 1),
            convex_hull_valid_mask,
            device,
            init_noise_sigma=args.noise_init,
            em_inner_opt_verbose=False,
            throwaway_log_prob=-6,
            compute_image_every_n=args.reestim_every
        )

    else:

        shuffle_jitter_dataloader = ShuffledRepeatsJitteredMovieDataloader(
            loaded_synchronized_datasets,
            cells_ordered,
            samples_per_bin,
            crop_w_ix=(crop_width_low, 320 - crop_width_low),  # FIXME
            image_rescale_lambda=image_rescale_lambda,
            n_shuffle_at_a_time=args.shuffle
        )

        ground_truth, reconstructions, eye_movement_trajectories = shuffled_repeats_generate_joint_eye_movement_trajectory_reconstructions(
            packed_glm_tensors,
            shuffle_jitter_dataloader,
            args.shuffle,
            samples_per_bin,
            30 * samples_per_bin,
            args.num_particles,
            gaussian_multinomial,
            args.prior_lambda,
            args.eye_movements,
            noreduce_nomask_batch_bin_bernoulli_neg_LL,
            make_get_iterators(args.start, args.end, args.iter),
            make_get_iterators(args.end, args.end, 1),
            convex_hull_valid_mask,
            device,
            init_noise_sigma=args.noise_init,
            em_inner_opt_verbose=False,
            throwaway_log_prob=-6,
            compute_image_every_n=args.reestim_every
        )

    with open(args.save_path, 'wb') as pfile:
        save_data = {
            'ground_truth': ground_truth,
            'joint_estim': reconstructions,
            'traj': eye_movement_trajectories,
            'repeat_type': 'shuffle' if args.shuffle != 0 else 'raw'
        }

        pickle.dump(save_data, pfile)

    print('done')
