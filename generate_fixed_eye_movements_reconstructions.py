import argparse
import pickle
from typing import List, Tuple, Callable, Optional

import numpy as np
import torch
import tqdm

import gaussian_denoiser.denoiser_wrappers as denoiser_wrappers
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.dynamic_data_util import JitteredMovieBatchDataloader
from dejitter_recons.estimate_image import noreduce_nomask_batch_bin_bernoulli_neg_LL, construct_likelihood_masks, \
    compute_ground_truth_eye_movements, estimate_image_with_fixed_eye_movements
from denoise_inverse_alg.glm_inverse_alg import PackedGLMTensors, \
    reinflate_cropped_glm_model, reinflate_cropped_fb_only_glm_model
from denoise_inverse_alg.hqs_alg import BatchParallel_DirectSolve_HQS_ZGenerator, AdamOptimParams, \
    BatchParallel_Adam_HQS_XGenerator
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths, parse_mixnmatch_path_yaml
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from optimization_encoder.trial_glm import load_fitted_glm_families


def make_get_iterators(
        rho_start: float,
        rho_end: float,
        max_iter: int) -> Callable:

    def get_iterators():
        basic_rho_sched = np.logspace(np.log10(rho_start), np.log10(rho_end), max_iter)

        adam_solver_gen = BatchParallel_Adam_HQS_XGenerator(
            [AdamOptimParams(25, 1e-1), AdamOptimParams(25, 1e-1), AdamOptimParams(25, 1e-1)],
            default_params=AdamOptimParams(10, 5e-2))

        z_solver_gen = BatchParallel_DirectSolve_HQS_ZGenerator()

        return basic_rho_sched, adam_solver_gen, z_solver_gen

    return get_iterators


def batch_generate_known_eye_movement_trajectory_reconstructions(
        packed_glm_tensors: PackedGLMTensors,
        jitter_dataset: JitteredMovieBatchDataloader,
        bin_width: int,
        bin_width_delay: int,
        prior_weight_lambda: float,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        full_hqs_get_iterators: Callable,
        valid_region_mask: np.ndarray,
        batch_size: int,
        device: torch.device,
        use_exact_eye_movements: bool = True,
        predetermined_eye_movements: Optional[np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    '''

    :param packed_glm_tensors:
    :param jitter_dataset:
    :param bin_width:
    :param bin_width_delay:
    :param prior_weight_lambda:
    :param loss_fn:
    :param full_hqs_get_iterators:
    :param valid_region_mask:
    :param batch_size:
    :param device:
    :param use_exact_eye_movements: bool, default True. When True, computes the exact eye movements
        from the stimulus movies, and then uses those eye movements to generate the images.

        Otherwise, if False, and predetermined_eye_movements is None, sets the eye movements to zero
        when generating images

        Otherwise, if False, and predetermined_eye_movements has a value, takes the eye movements from
            predetermined_eye_movements when generating images

    :param predetermined_eye_movements: Optional[np.ndarray], shape (len(jitter_dataset), n_frames_per_stimulus, 2)
        manual eye movements. Shape must match len(jitter_dataset) otherwise will crash

        If want to use the eye movements in here, use_exact_eye_movements must be False
    :return:
    '''

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

    if not use_exact_eye_movements and predetermined_eye_movements is not None:
        assert predetermined_eye_movements.shape[0] == len(jitter_dataset), \
            'externally set eye movements must match the provided dataset'

    eye_movements_buffer = []

    pbar = tqdm.tqdm(total=len(jitter_dataset))
    for low in range(0, len(jitter_dataset), batch_size):
        high = min(len(jitter_dataset), low + batch_size)

        history_frames, target_frames, snippet_transitions, spike_bins, binned_spikes = jitter_dataset[low:high]

        batch_likelihood_masks = construct_likelihood_masks(
            history_frames.shape[1],
            snippet_transitions,
            spike_bins,
            bin_width_delay,
            bin_width)

        eye_movements = np.zeros((target_frames.shape[0], target_frames.shape[1], 2), dtype=np.int64)
        if use_exact_eye_movements:
            for i in range(target_frames.shape[0]):
                jitter_coords, diff_norms = compute_ground_truth_eye_movements(target_frames[i, ...], 15, device)
                eye_movements[i, ...] = jitter_coords
        else:
            if predetermined_eye_movements is not None:
                for jj, kk in enumerate(range(low, high)):
                    eye_movements[jj, ...] = predetermined_eye_movements[kk, ...]
        eye_movements_buffer.append(eye_movements)

        basic_rho_sched, adam_solver_gen, z_solver_gen = full_hqs_get_iterators()

        estim_image, intermediates = estimate_image_with_fixed_eye_movements(
            packed_glm_tensors,
            unblind_denoiser_callable,
            history_frames,
            snippet_transitions,
            binned_spikes,
            spike_bins,
            valid_region_mask,
            eye_movements,
            batch_likelihood_masks[-1],
            loss_fn,
            basic_rho_sched,
            prior_weight_lambda,
            adam_solver_gen,
            z_solver_gen,
            device,
            return_intermediate_images=True,
            solver_verbose=False
        )

        ground_truth_buffer[low:high, ...] = target_frames[:, 0, ...]
        output_image_buffer[low:high, ...] = estim_image

        pbar.update((high - low))
    pbar.close()

    return ground_truth_buffer, output_image_buffer, eye_movements_buffer



if __name__ == '__main__':

    parser = argparse.ArgumentParser("Generate reconstructions for the eye movements stimulus where the eye movements are either known a priori or ignored entirely, using LNBRC encoding and dCNN prior.")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('yaml_path', type=str, help='path to YAML specifying encoding model weights')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-st', '--start', type=float, default=0.1, help='HQS rho first value')
    parser.add_argument('-en', '--end', type=float, default=10.0, help='HQS rho last value')
    parser.add_argument('-i', '--iter', type=int, default=5, help='HQS num outer iters')
    parser.add_argument('-lam', '--prior_lambda', type=float, default=0.2, help='HQS prior weight')
    parser.add_argument('-ee', '--exact_eye_movements', action='store_true', default=False,
                        help='Use exact eye movements')
    parser.add_argument('-b', '--batch', type=int, default=8, help='Maximum batch size')
    parser.add_argument('-hh', '--heldout', action='store_true', default=False, help='generate heldout images')
    parser.add_argument('-j', '--jitter', type=float, default=0.0, help='Time jitter SD, units electrical samples')
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

    print(create_test_dataset, create_heldout_dataset)

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

    # FIXME see if we can get a better way to do this
    height, width = 160, (320 - crop_width_low - crop_width_high)

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

    print(len(jitter_dataloader))

    ground_truth, reconstructions, assumed_eye_movements = batch_generate_known_eye_movement_trajectory_reconstructions(
        packed_glm_tensors,
        jitter_dataloader,
        samples_per_bin,
        30 * samples_per_bin,
        args.prior_lambda,
        noreduce_nomask_batch_bin_bernoulli_neg_LL,
        make_get_iterators(args.start, args.end, args.iter),
        convex_hull_valid_mask,
        args.batch,
        device,
        use_exact_eye_movements=(args.exact_eye_movements)
    )

    with open(args.save_path, 'wb') as pfile:
        save_data = {
            'ground_truth': ground_truth ,
            'joint_estim': reconstructions,
            'assumed_eye_movements': assumed_eye_movements
        }

        pickle.dump(save_data, pfile)

    print('done')
