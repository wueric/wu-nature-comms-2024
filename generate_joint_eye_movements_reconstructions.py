import argparse
import pickle
from typing import List, Tuple, Optional, Callable

import numpy as np
import torch
import tqdm

import gaussian_denoiser.denoiser_wrappers as denoiser_wrappers
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.dynamic_data_util import JitteredMovieBatchDataloader
from dejitter_recons.estimate_image import noreduce_nomask_batch_bin_bernoulli_neg_LL
from dejitter_recons.joint_em_estimation import non_online_joint_em_estimation2, create_gaussian_multinomial
from denoise_inverse_alg.glm_inverse_alg import PackedGLMTensors, \
    reinflate_cropped_glm_model, reinflate_cropped_fb_only_glm_model
from denoise_inverse_alg.hqs_alg import BatchParallel_DirectSolve_HQS_ZGenerator, Adam_HQS_XGenerator, \
    AdamOptimParams
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths, parse_mixnmatch_path_yaml
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from optimization_encoder.trial_glm import load_fitted_glm_families


# FIXME when we get another dataset going, make this dataset specific
def make_get_iterators(
        rho_start: float,
        rho_end: float,
        max_iter: int) -> Callable:

    def get_iterators():
        basic_rho_sched = np.logspace(np.log10(rho_start), np.log10(rho_end), max_iter)

        adam_solver_gen = Adam_HQS_XGenerator(
            [AdamOptimParams(25, 1e-1), AdamOptimParams(25, 1e-1), AdamOptimParams(25, 1e-1)],
            default_params=AdamOptimParams(10, 5e-2))

        z_solver_gen = BatchParallel_DirectSolve_HQS_ZGenerator()

        return basic_rho_sched, adam_solver_gen, z_solver_gen

    return get_iterators


def generate_joint_eye_movement_trajectory_reconstructions(
        packed_glm_tensors: PackedGLMTensors,
        jitter_dataset: JitteredMovieBatchDataloader,
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
        init_noise_sigma: Optional[float]=None,
        em_inner_opt_verbose=False,
        throwaway_log_prob=-6,
        compute_image_every_n=10) \
        -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:

    _, height, width = packed_glm_tensors.spatial_filters.shape

    ################################################################################
    # first load the unblind denoiser
    unblind_denoiser_model = denoiser_wrappers.load_masked_drunet_unblind_denoiser(device)
    unblind_denoiser_callable = denoiser_wrappers.make_unblind_apply_dpir_denoiser_with_mask(
        unblind_denoiser_model,
        (-1.0, 1.0), (0.0, 255))
    ################################################################################


    output_image_buffer = np.zeros((len(jitter_dataset), height, width), dtype=np.float32)
    ground_truth_buffer = np.zeros((len(jitter_dataset), height, width), dtype=np.float32)
    traj_weights_buffer = [] # type: List[Tuple[np.ndarray, np.ndarray]]

    pbar = tqdm.tqdm(total=len(jitter_dataset))
    for i in range(len(jitter_dataset)):
        history_frames, target_frames, snippet_transitions, spike_bins, binned_spikes = jitter_dataset[i]

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
        output_image_buffer[i, ...] = image
        traj_weights_buffer.append((traj, weights))

        pbar.update(1)
    pbar.close()

    return ground_truth_buffer, output_image_buffer, traj_weights_buffer



if __name__ == '__main__':

    parser = argparse.ArgumentParser("Generate reconstructions for the joint estimation of eye movements and image, using LNBRC (coupled) encoding model and dCNN image prior.")
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
    parser.add_argument('-hh', '--heldout', action='store_true', default=False, help='generate heldout images')
    parser.add_argument('-j', '--jitter', type=float, default=0.0, help='Spike time jitter SD, units electrical samples')
    parser.add_argument('-f', '--feedback_only', action='store_true', default=False,
                        help='GLM model specified by model_cfg_path is feedback-only')
    parser.add_argument('--mixnmatch', action='store_true', default=False,
                        help='Mix and match spike perturbation levels')

    args = parser.parse_args()

    if args.mixnmatch:
        assert args.jitter == 0, 'jitter must be zero for mix-and-match'

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
    #####################################################
    # We have to parse the YAML first before getting spikes, since we need to figure out
    # ahead of time whether we're doing mix-n-match and what the jitter
    # should be if we are
    # Load the GLMs
    jitter_amount = args.jitter
    if args.mixnmatch:
        fitted_glm_paths, jitter_amount_dict = parse_mixnmatch_path_yaml(args.yaml_path)
        jitter_amount = ddu.construct_spike_jitter_amount_by_cell_id(jitter_amount_dict,
                                                                     cells_ordered)
    else:
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

    jitter_dataloader = JitteredMovieBatchDataloader(
        loaded_synchronized_datasets,
        cells_ordered,
        ddu.PartitionType.HELDOUT_PARTITION if args.heldout \
            else ddu.PartitionType.TEST_PARTITION,
        samples_per_bin,
        image_rescale_lambda=image_rescale_lambda,
        crop_w_ix=(crop_width_low, 320 - crop_width_low), # FIXME
        time_jitter_spikes=jitter_amount
    )

    gaussian_multinomial = create_gaussian_multinomial(1.2, 2)

    ground_truth, reconstructions, eye_movement_trajectories = generate_joint_eye_movement_trajectory_reconstructions(
        packed_glm_tensors,
        jitter_dataloader,
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
            'ground_truth': ground_truth ,
            'joint_estim': reconstructions,
            'traj': eye_movement_trajectories
        }

        pickle.dump(save_data, pfile)

    print('done')
