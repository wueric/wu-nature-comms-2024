import argparse
import itertools
import pickle
from collections import namedtuple
from typing import List, Callable, Sequence, Tuple, Iterator, Union

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
from denoise_inverse_alg.glm_inverse_alg import PackedGLMTensors, reinflate_cropped_glm_model, \
    FeedbackOnlyPackedGLMTensors, reinflate_cropped_fb_only_glm_model
from denoise_inverse_alg.hqs_alg import BatchParallel_DirectSolve_HQS_ZGenerator, \
    AdamOptimParams, Adam_HQS_XGenerator
from eval_fns.eval import MaskedMSELoss, SSIM, MS_SSIM
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths, parse_mixnmatch_path_yaml
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from optimization_encoder.trial_glm import load_fitted_glm_families

EyeMovementGridSearchParams = namedtuple('EyeMovementGridSearchParams', ['eye_movement_weight'])
GridSearchReconstructions = namedtuple('GridSearchReconstructions',
                                       ['ground_truth', 'reconstructions', 'mse', 'ssim', 'ms_ssim'])

GRID_SEARCH_TEST_N_IMAGES = 80

BIN_WIDTH = 20


def enumerated_product(*args):
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))


def make_get_iterators(rho_start: float,
                       rho_end: float,
                       max_iter: int) \
        -> Callable[[], Tuple[Iterator, Iterator, Iterator]]:
    def get_iterators():
        basic_rho_sched = np.logspace(np.log10(rho_start), np.log10(rho_end), max_iter)

        adam_solver_gen = Adam_HQS_XGenerator(
            [AdamOptimParams(25, 1e-1), AdamOptimParams(25, 1e-1), AdamOptimParams(25, 1e-1)],
            default_params=AdamOptimParams(10, 5e-2))

        z_solver_gen = BatchParallel_DirectSolve_HQS_ZGenerator()

        return basic_rho_sched, adam_solver_gen, z_solver_gen

    return get_iterators


def make_do_less_work_update_iter(rho_start: float,
                                  rho_end: float,
                                  max_iter: int) \
        -> Callable[[], Tuple[Iterator, Iterator, Iterator]]:
    def do_less_work_update_iter():
        basic_rho_sched = np.logspace(np.log10(rho_start), np.log10(rho_end), max_iter)

        adam_solver_gen = Adam_HQS_XGenerator(
            [AdamOptimParams(25, 1e-1), ],
            default_params=AdamOptimParams(10, 5e-2))

        z_solver_gen = BatchParallel_DirectSolve_HQS_ZGenerator()

        return basic_rho_sched, adam_solver_gen, z_solver_gen

    return do_less_work_update_iter


def single_eye_movments_map_em_grid_search(
        jittered_movie_dataloader: JitteredMovieBatchDataloader,
        indices_to_get: Sequence[int],
        packed_glm_tensors: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors],
        valid_region_mask: np.ndarray,
        reconstruction_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        image_to_metric_callable: Callable[[torch.Tensor], torch.Tensor],
        mse_module: MaskedMSELoss,
        ssim_module: SSIM,
        ms_ssim_module: MS_SSIM,
        prior_weight_lambda: float,
        grid_eye_movements_weight: np.ndarray,
        make_iterators_callable: Callable[[], Tuple[Iterator, Iterator, Iterator]],
        make_do_less_work_iterators_callable: Callable[[], Tuple[Iterator, Iterator, Iterator]],
        device: torch.device):
    _, height, width = packed_glm_tensors.spatial_filters.shape

    n_examples = len(indices_to_get)

    ################################################################################
    # first load the unblind denoiser
    unblind_denoiser_model = denoiser_wrappers.load_masked_drunet_unblind_denoiser(device)
    unblind_denoiser_callable = denoiser_wrappers.make_unblind_apply_dpir_denoiser_with_mask(
        unblind_denoiser_model,
        (-1.0, 1.0), (0.0, 255))

    grid_search_pbar = tqdm.tqdm(
        total=grid_eye_movements_weight.shape[0])

    gaussian_multinomial = create_gaussian_multinomial(1.2, 2)

    ret_dict = {}
    for ix, grid_params in enumerated_product(grid_eye_movements_weight):

        eye_mvmt_weight, = grid_params
        n_particles = 10

        output_dict_key = EyeMovementGridSearchParams(eye_mvmt_weight)

        output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
        example_stimuli_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
        pbar = tqdm.tqdm(total=n_examples)
        for store_to, read_from in enumerate(indices_to_get):
            to_get = indices_to_get[read_from]

            history_f, target_f, f_transitions, spike_bins, binned_spikes = jittered_movie_dataloader[to_get]

            image2, traj2, weights2 = non_online_joint_em_estimation2(
                packed_glm_tensors,
                unblind_denoiser_callable,
                history_f,
                f_transitions,
                binned_spikes,
                spike_bins,
                valid_region_mask,
                BIN_WIDTH,
                BIN_WIDTH * 30,  # FIXME check units; it's right
                n_particles,  # n_particles
                gaussian_multinomial,
                reconstruction_loss_fn,
                prior_weight_lambda,
                make_iterators_callable,
                make_do_less_work_iterators_callable,
                device,
                image_init_guess=np.zeros((160, 256), dtype=np.float32),
                em_inner_opt_verbose=False,
                return_intermediate_em_results=False,
                throwaway_log_prob=-6,
                likelihood_scale=eye_mvmt_weight,  # tried 1.0
                compute_image_every_n=10
            )

            output_image_buffer_np[store_to, ...] = image2
            example_stimuli_buffer_np[store_to, ...] = target_f[0, :, :]
            pbar.update(1)

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

    parser = argparse.ArgumentParser("Grid search HQS hyperparameters for eye movements reconstructions")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('model_cfg_path', type=str, help='path to YAML specifying where the GLM fits are')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-e', '--eye_movements', type=float, nargs='+',
                        help='Grid search parameters for eye movements weight. Specify weights explicitly')
    parser.add_argument('-j', '--jitter', type=float, default=0.0, help='time jitter SD (units of electrical samples)')
    parser.add_argument('-i', '--n_iter', type=int, default=5, help='number of HQS outer iterations')
    parser.add_argument('-lam', '--prior_lambda', type=float, default=0.15, help='Lambda weight on the prior')
    parser.add_argument('-st', '--rho_start', type=float, default=1.0, help='starting value for rho, log-spaced')
    parser.add_argument('-en', '--rho_end', type=float, default=10.0, help='ending value for rho, log-spaced')
    parser.add_argument('-f', '--feedback_only', action='store_true', default=False,
                        help='GLM model specified by model_cfg_path is feedback-only')
    parser.add_argument('--mixnmatch', action='store_true', default=False,
                        help='Mix and match spike perturbation levels')

    args = parser.parse_args()

    if args.mixnmatch:
        assert args.jitter == 0, 'jitter must be zero for mix-and-match'

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    grid_search_eye_movements = np.array(args.eye_movements)

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

    #####################################################
    # We have to parse the YAML first before getting spikes, since we need to figure out
    # ahead of time whether we're doing mix-n-match and what the jitter
    # should be if we are
    # Load the GLMs
    jitter_amount = args.jitter
    if args.mixnmatch:
        fitted_glm_paths, jitter_amount_dict = parse_mixnmatch_path_yaml(args.model_cfg_path)
        jitter_amount = ddu.construct_spike_jitter_amount_by_cell_id(jitter_amount_dict,
                                                                     cells_ordered)
    else:
        fitted_glm_paths = parse_prefit_glm_paths(args.model_cfg_path)

    fitted_glm_families = load_fitted_glm_families(fitted_glm_paths)

    model_reinflation_fn = reinflate_cropped_fb_only_glm_model if args.feedback_only \
        else reinflate_cropped_glm_model

    packed_glm_tensors = model_reinflation_fn(
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
    # Load the natural scenes Vision datasets and determine what the
    # train and test partitions are
    nscenes_dataset_info_list = config_settings[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]

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
        # do_trigger_interpolation=True
    )

    jitter_dataloader = JitteredMovieBatchDataloader(
        loaded_synchronized_datasets,
        cells_ordered,
        ddu.PartitionType.TEST_PARTITION,
        samples_per_bin,
        crop_w_ix=(32, 320 - 32),  # FIXME,
        image_rescale_lambda=image_rescale_lambda,
        time_jitter_spikes=jitter_amount
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

    invalid_mask = ~convex_hull_valid_mask

    grid_search_output_dict = single_eye_movments_map_em_grid_search(
        jitter_dataloader,
        np.r_[0:GRID_SEARCH_TEST_N_IMAGES],
        packed_glm_tensors,
        convex_hull_valid_mask,
        noreduce_nomask_batch_bin_bernoulli_neg_LL,
        image_to_metric_lambda,
        masked_mse_module,
        ssim_module,
        ms_ssim_module,
        args.prior_lambda,
        grid_search_eye_movements,
        make_get_iterators(args.rho_start, args.rho_end, args.n_iter),
        make_do_less_work_update_iter(args.rho_end, args.rho_end, 1),
        device
    )

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(grid_search_output_dict, pfile)
