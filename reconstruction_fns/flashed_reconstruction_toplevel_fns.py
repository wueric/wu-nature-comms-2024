import itertools
from collections import namedtuple
from typing import Union, Callable, Tuple, Dict, Optional, Iterator

import numpy as np
import torch
import torch.nn as nn
import tqdm

from denoise_inverse_alg.glm_inverse_alg import PackedGLMTensors, FlashedModelRequiresPrecomputation, \
    BatchKnownSeparable_TrialGLM_ProxProblem, FeedbackOnlyPackedGLMTensors, \
    MixNMatch_BatchKnownSeparable_Trial_GLM_ProxProblem
from denoise_inverse_alg.hqs_alg import BatchParallel_HQS_X_Problem, ScheduleVal, iter_rho_fixed_prior_hqs_solve, \
    construct_create_batched_denoiser_HQS_Z_problem_fn, HQS_ParameterizedSolveFn, make_logspaced_rho_schedule
from denoise_inverse_alg.noiseless_inverse_alg import BatchNoiselessLinear_HQS_XProb
from denoise_inverse_alg.poisson_inverse_alg import PackedLNPTensors, BatchFlashedFrameRatePoissonProxProblem
from eval_fns.eval import MaskedMSELoss, MS_SSIM, SSIM
from linear_decoding_models.linear_decoding_models import ClosedFormLinearModel
from reconstruction_fns.grid_search_types import GridSearchParams, GridSearchReconstructions

LinearModelBinningRange = namedtuple('LinearModelBinningRange', ['start_cut', 'end_cut'])


def enumerated_product(*args):
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))


def image_rescale_0_1(images_min1_max1: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        images_rescaled = torch.clamp((images_min1_max1 + 1.0) / 2.0,
                                      min=0.0, max=1.0)
        return images_rescaled


def create_reconstruction_Batched_HQS_X_problem(
        packed_model_tensors: Union[PackedGLMTensors, PackedLNPTensors, FeedbackOnlyPackedGLMTensors],
        flashed_time_component: np.ndarray,
        reconstruction_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_size: int) -> Union[nn.Module,
                                  BatchParallel_HQS_X_Problem,
                                  FlashedModelRequiresPrecomputation]:
    if isinstance(packed_model_tensors, PackedLNPTensors):

        unblind_denoise_hqs_x_prob = BatchFlashedFrameRatePoissonProxProblem(
            batch_size,
            packed_model_tensors.spatial_filters,
            packed_model_tensors.timecourse_filters,
            packed_model_tensors.bias.squeeze(1),
            flashed_time_component,
            reconstruction_loss_fn,
        )

    else:

        unblind_denoise_hqs_x_prob = BatchKnownSeparable_TrialGLM_ProxProblem(
            batch_size,
            packed_model_tensors,
            flashed_time_component,
            reconstruction_loss_fn,
        )

    return unblind_denoise_hqs_x_prob


def batch_parallel_flashed_hqs_grid_search(
        example_spikes: np.ndarray,
        example_stimuli: np.ndarray,
        image_range_tuple: Tuple[float, float],
        batch_size: int,
        packed_model_tensors: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors, PackedLNPTensors],
        reconstruction_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        stimulus_time_component: np.ndarray,
        mse_module: MaskedMSELoss,
        ssim_module: SSIM,
        ms_ssim_module: MS_SSIM,
        grid_lambda_start: np.ndarray,
        grid_lambda_end: np.ndarray,
        grid_prior: np.ndarray,
        max_iter: int,
        make_x_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        make_z_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        device: torch.device,
        initialize_linear_model: Optional[Tuple[ClosedFormLinearModel, LinearModelBinningRange]] = None,
        valid_region_mask: Optional[np.ndarray] = None) -> Dict[GridSearchParams, GridSearchReconstructions]:
    '''

    :param example_spikes:
    :param example_stimuli:
    :param batch_size:
    :param packed_model_tensors:
    :param reconstruction_loss_fn:
    :param glm_time_component:
    :param mse_module:
    :param ssim_module:
    :param ms_ssim_module:
    :param grid_lambda_start:
    :param grid_lambda_end:
    :param grid_prior:
    :param max_iter:
    :param device:
    :param initialize_linear_model:
    :param valid_region_mask:
    :return:
    '''

    n_examples, n_cells, n_bins_observed = example_spikes.shape
    _, height, width = packed_model_tensors.spatial_filters.shape
    print(height, width)

    assert n_examples % batch_size == 0, 'number of example images must be multiple of batch size'

    example_stimuli_torch = torch.tensor(example_stimuli, dtype=torch.float32, device=device)
    example_spikes_torch = torch.tensor(example_spikes, dtype=torch.float32, device=device)

    rescaled_example_stimuli = image_rescale_0_1(example_stimuli_torch)
    del example_stimuli_torch

    ################################################################################
    create_HQS_Z_prob_fn = construct_create_batched_denoiser_HQS_Z_problem_fn(
        (height, width),
        image_range_tuple,
        device,
        valid_region_mask=valid_region_mask
    )

    grid_search_pbar = tqdm.tqdm(
        total=grid_prior.shape[0] * grid_lambda_start.shape[0] * grid_lambda_end.shape[0])

    ret_dict = {}
    for ix, grid_params in enumerated_product(grid_prior, grid_lambda_start, grid_lambda_end):

        prior_weight, lambda_start, lambda_end = grid_params

        output_dict_key = GridSearchParams(lambda_start, lambda_end, prior_weight, max_iter)

        schedule_rho = make_logspaced_rho_schedule(ScheduleVal(lambda_start, lambda_end, prior_weight, max_iter))

        ################################################################################
        # make the models
        unblind_denoise_hqs_x_prob = create_reconstruction_Batched_HQS_X_problem(
            packed_model_tensors,
            stimulus_time_component,
            reconstruction_loss_fn,
            batch_size
        ).to(device)

        unblind_denoise_hqs_z_prob = create_HQS_Z_prob_fn(batch_size)

        ################################################################################
        output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
        pbar = tqdm.tqdm(total=n_examples)

        for low in range(0, n_examples, batch_size):
            high = low + batch_size
            glm_trial_spikes_torch = example_spikes_torch[low:high, ...]

            unblind_denoise_hqs_x_prob_solver_iter = make_x_prob_solver_generator_fn()
            unblind_denoise_hqs_z_prob_solver_iter = make_z_prob_solver_generator_fn()

            # if we want to use the linear reconstruction as an intermediate, do a linear reconstruction
            initialize_z_tensor = None
            if initialize_linear_model is not None:
                linear_model, linear_bin_range = initialize_linear_model
                linear_low, linear_high = linear_bin_range.start_cut, glm_trial_spikes_torch.shape[
                    1] - linear_bin_range.end_cut
                with torch.no_grad():
                    summed_spike_for_linear = torch.sum(glm_trial_spikes_torch[:, :, linear_low:linear_high],
                                                        dim=2)
                    initialize_z_tensor = linear_model(summed_spike_for_linear)

            unblind_denoise_hqs_x_prob.reinitialize_variables(initialized_z_const=initialize_z_tensor)
            unblind_denoise_hqs_x_prob.precompute_gensig_components(glm_trial_spikes_torch)
            unblind_denoise_hqs_z_prob.reinitialize_variables()

            _ = iter_rho_fixed_prior_hqs_solve(
                unblind_denoise_hqs_x_prob,
                iter(unblind_denoise_hqs_x_prob_solver_iter),
                unblind_denoise_hqs_z_prob,
                iter(unblind_denoise_hqs_z_prob_solver_iter),
                iter(schedule_rho),
                prior_weight,
                verbose=False,
                save_intermediates=False,
                observed_spikes=glm_trial_spikes_torch
            )

            denoise_hqs_reconstructed_image = unblind_denoise_hqs_x_prob.get_reconstructed_image()
            output_image_buffer_np[low:high, :, :] = denoise_hqs_reconstructed_image

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

        del output_image_buffer, unblind_denoise_hqs_x_prob, unblind_denoise_hqs_z_prob, reconstruction_rescaled
        grid_search_pbar.update(1)

    return ret_dict


def batch_parallel_generate_flashed_hqs_reconstructions(
        example_spikes: np.ndarray,
        image_range_tuple: Tuple[float, float],
        packed_model_tensors: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors, PackedLNPTensors],
        reconstruction_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        glm_time_component: np.ndarray,
        reconstruction_hyperparams: GridSearchParams,
        make_x_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        make_z_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        max_batch_size: int,
        device: torch.device,
        initialize_noise_level: float = 1e-3,
        initialize_linear_model: Optional[
            Tuple[ClosedFormLinearModel, LinearModelBinningRange]] = None,
        valid_region_mask: Optional[np.ndarray] = None) \
        -> np.ndarray:
    '''

    :param example_spikes:
    :param packed_glm_tensors:
    :param glm_time_component:
    :param device:
    :return:
    '''

    n_examples, n_cells, n_bins_observed = example_spikes.shape
    _, height, width = packed_model_tensors.spatial_filters.shape

    example_spikes_torch = torch.tensor(example_spikes, dtype=torch.float32, device=device)

    #################################################################################
    # get the hyperparameters
    lambda_start, lambda_end = reconstruction_hyperparams.lambda_start, reconstruction_hyperparams.lambda_end
    prior_weight = reconstruction_hyperparams.prior_weight
    max_iter = reconstruction_hyperparams.max_iter
    schedule_rho = make_logspaced_rho_schedule(ScheduleVal(lambda_start, lambda_end, prior_weight, max_iter))

    #################################################################################
    create_HQS_Z_prob_fn = construct_create_batched_denoiser_HQS_Z_problem_fn(
        (height, width),
        image_range_tuple,
        device,
        valid_region_mask=valid_region_mask
    )

    # run the first N-1 iterations
    output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
    pbar = tqdm.tqdm(total=n_examples)
    for low in range(0, n_examples, max_batch_size):
        high = min(low + max_batch_size, n_examples)
        eff_batch_size = high - low

        ################################################################################
        # make the models and transfer to device
        unblind_denoise_hqs_x_prob = create_reconstruction_Batched_HQS_X_problem(
            packed_model_tensors,
            glm_time_component,
            reconstruction_loss_fn,
            eff_batch_size
        ).to(device)

        unblind_denoise_hqs_z_prob = create_HQS_Z_prob_fn(eff_batch_size)
        ################################################################################

        glm_trial_spikes_torch = example_spikes_torch[low:high, ...]

        unblind_denoise_hqs_x_prob_solver_iter = make_x_prob_solver_generator_fn()
        unblind_denoise_hqs_z_prob_solver_iter = make_z_prob_solver_generator_fn()

        # if we want to use the linear reconstruction as an intermediate, do a linear reconstruction
        if initialize_linear_model is not None:
            linear_model, linear_bin_range = initialize_linear_model
            linear_low, linear_high = linear_bin_range.start_cut, glm_trial_spikes_torch.shape[
                1] - linear_bin_range.end_cut
            with torch.no_grad():
                summed_spike_for_linear = torch.sum(glm_trial_spikes_torch[:, :, linear_low:linear_high],
                                                    dim=2)
                initialize_z_tensor = linear_model(summed_spike_for_linear)
        else:
            initialize_z_tensor = torch.randn((glm_trial_spikes_torch.shape[0], height, width),
                                              dtype=torch.float32) * initialize_noise_level

        unblind_denoise_hqs_x_prob.reinitialize_variables(initialized_z_const=initialize_z_tensor)
        unblind_denoise_hqs_x_prob.precompute_gensig_components(glm_trial_spikes_torch)
        unblind_denoise_hqs_z_prob.reinitialize_variables()

        _ = iter_rho_fixed_prior_hqs_solve(
            unblind_denoise_hqs_x_prob,
            iter(unblind_denoise_hqs_x_prob_solver_iter),
            unblind_denoise_hqs_z_prob,
            iter(unblind_denoise_hqs_z_prob_solver_iter),
            iter(schedule_rho),
            prior_weight,
            verbose=False,
            save_intermediates=False,
            observed_spikes=glm_trial_spikes_torch
        )

        denoise_hqs_reconstructed_image = unblind_denoise_hqs_x_prob.get_reconstructed_image()
        output_image_buffer_np[low:high, :, :] = denoise_hqs_reconstructed_image

        del unblind_denoise_hqs_x_prob, unblind_denoise_hqs_z_prob

        pbar.update(max_batch_size)

    return output_image_buffer_np


def batch_parallel_generated_noiseless_linear_hqs_reconstructions(
        linear_projections: np.ndarray,
        image_range_tuple: Tuple[float, float],
        linear_projection_tensors: np.ndarray,
        reconstruction_hyperparams: GridSearchParams,
        make_x_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        make_z_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        max_batch_size: int,
        device: torch.device,
        initialize_noise_level: float = 1e-3,
        valid_region_mask: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    linear_projections: shape (n_examples, n_cells)
    image_range_tuple: min and max value for the images
    linear_projection_tensors: shape (n_cells, height, width), linear projection filters for each cell
    '''

    n_examples, n_cells = linear_projections.shape
    _, height, width = linear_projection_tensors.shape

    linear_projections_torch = torch.tensor(linear_projections, dtype=torch.float32, device=device)

    #################################################################################
    # get the hyperparameters
    lambda_start, lambda_end = reconstruction_hyperparams.lambda_start, reconstruction_hyperparams.lambda_end
    prior_weight = reconstruction_hyperparams.prior_weight
    max_iter = reconstruction_hyperparams.max_iter
    schedule_rho = make_logspaced_rho_schedule(ScheduleVal(lambda_start, lambda_end, prior_weight, max_iter))

    #################################################################################
    create_HQS_Z_prob_fn = construct_create_batched_denoiser_HQS_Z_problem_fn(
        (height, width),
        image_range_tuple,
        device,
        valid_region_mask=valid_region_mask
    )

    # run the first N-1 iterations
    output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
    pbar = tqdm.tqdm(total=n_examples)
    for low in range(0, n_examples, max_batch_size):
        high = min(low + max_batch_size, n_examples)
        eff_batch_size = high - low

        ################################################
        # make the problem and transfer to device
        hqs_x_problem = BatchNoiselessLinear_HQS_XProb(
            eff_batch_size,
            linear_projection_tensors,
            dtype=torch.float32
        ).to(device)

        hqs_z_problem = create_HQS_Z_prob_fn(eff_batch_size)
        ##################################################

        unblind_denoise_hqs_x_prob_solver_iter = make_x_prob_solver_generator_fn()
        unblind_denoise_hqs_z_prob_solver_iter = make_z_prob_solver_generator_fn()

        initialize_z_tensor = torch.randn((eff_batch_size, height, width),
                                          dtype=torch.float32) * initialize_noise_level

        hqs_x_problem.reinitialize_variables(initialized_z_const=initialize_z_tensor)
        hqs_z_problem.reinitialize_variables()

        _ = iter_rho_fixed_prior_hqs_solve(
            hqs_x_problem,
            iter(unblind_denoise_hqs_x_prob_solver_iter),
            hqs_z_problem,
            iter(unblind_denoise_hqs_z_prob_solver_iter),
            iter(schedule_rho),
            prior_weight,
            verbose=False,
            save_intermediates=False,
            projections=linear_projections_torch[low:high, ...]
        )

        denoise_hqs_reconstructed_image = hqs_x_problem.get_reconstructed_image()
        output_image_buffer_np[low:high, :, :] = denoise_hqs_reconstructed_image

        del hqs_x_problem, hqs_z_problem

        pbar.update(max_batch_size)

    return output_image_buffer_np


def batch_parallel_noiseless_linear_hqs_grid_search(
        linear_projections: np.ndarray,
        orig_stimuli: np.ndarray,
        image_range_tuple: Tuple[float, float],
        linear_projection_tensors: np.ndarray,
        batch_size: int,
        mse_module: MaskedMSELoss,
        ssim_module: SSIM,
        ms_ssim_module: MS_SSIM,
        grid_lambda_start: np.ndarray,
        grid_lambda_end: np.ndarray,
        grid_prior: np.ndarray,
        max_iter: int,
        make_x_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        make_z_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        device: torch.device,
        initialize_noise_level: float = 1e-3,
        valid_region_mask: Optional[np.ndarray] = None) -> Dict[GridSearchParams, GridSearchReconstructions]:

    '''
    linear_projections: shape (n_examples, n_cells)
    image_range_tuple: min and max value for the images
    linear_projection_tensors: shape (n_cells, height, width), linear projection filters for each cell
    '''

    n_cells, height, width = linear_projection_tensors.shape
    n_examples = linear_projections.shape[0]

    linear_projections_torch = torch.tensor(linear_projections,
                                            dtype=torch.float32, device=device)

    example_stimuli_torch = torch.tensor(orig_stimuli, dtype=torch.float32, device=device)
    rescaled_example_stimuli = image_rescale_0_1(example_stimuli_torch)
    del example_stimuli_torch

    ################################################################################
    create_HQS_Z_prob_fn = construct_create_batched_denoiser_HQS_Z_problem_fn(
        (height, width),
        image_range_tuple,
        device,
        valid_region_mask=valid_region_mask
    )

    grid_search_pbar = tqdm.tqdm(
        total=grid_prior.shape[0] * grid_lambda_start.shape[0] * grid_lambda_end.shape[0])

    ret_dict = {}

    for ix, grid_params in enumerated_product(grid_prior, grid_lambda_start, grid_lambda_end):
        prior_weight, lambda_start, lambda_end = grid_params

        output_dict_key = GridSearchParams(lambda_start, lambda_end, prior_weight, max_iter)
        schedule_rho = make_logspaced_rho_schedule(ScheduleVal(lambda_start, lambda_end, prior_weight, max_iter))

        ################################################
        # make the problem and transfer to device
        hqs_x_problem = BatchNoiselessLinear_HQS_XProb(
            batch_size,
            linear_projection_tensors,
            dtype=torch.float32
        ).to(device)

        hqs_z_problem = create_HQS_Z_prob_fn(batch_size)

        ################################################################################
        output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
        pbar = tqdm.tqdm(total=n_examples)

        for low in range(0, n_examples, batch_size):
            high = low + batch_size

            unblind_denoise_hqs_x_prob_solver_iter = make_x_prob_solver_generator_fn()
            unblind_denoise_hqs_z_prob_solver_iter = make_z_prob_solver_generator_fn()

            initialize_z_tensor = torch.randn((batch_size, height, width),
                                              dtype=torch.float32) * initialize_noise_level

            hqs_x_problem.reinitialize_variables(initialized_z_const=initialize_z_tensor)
            hqs_z_problem.reinitialize_variables()

            _ = iter_rho_fixed_prior_hqs_solve(
                hqs_x_problem,
                iter(unblind_denoise_hqs_x_prob_solver_iter),
                hqs_z_problem,
                iter(unblind_denoise_hqs_z_prob_solver_iter),
                iter(schedule_rho),
                prior_weight,
                verbose=False,
                save_intermediates=False,
                projections=linear_projections_torch[low:high, ...]
            )

            denoise_hqs_reconstructed_image = hqs_x_problem.get_reconstructed_image()
            output_image_buffer_np[low:high, :, :] = denoise_hqs_reconstructed_image

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
            orig_stimuli,
            output_image_buffer_np,
            masked_mse,
            ssim_val,
            ms_ssim_val)

        print(f"{output_dict_key}, MSE {masked_mse}, SSIM {ssim_val}, MS-SSIM {ms_ssim_val}")

        del output_image_buffer, hqs_x_problem, hqs_z_problem, reconstruction_rescaled
        grid_search_pbar.update(1)

    return ret_dict


def mixnmatch_batch_parallel_flashed_hqs_grid_search(
        example_keyed_spikes: Dict[str, np.ndarray],
        example_stimuli: np.ndarray,
        image_range_tuple: Tuple[float, float],
        batch_size: int,
        typed_packed_model_tensors: Dict[str, PackedGLMTensors],
        reconstruction_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        stimulus_time_component: np.ndarray,
        mse_module: MaskedMSELoss,
        ssim_module: SSIM,
        ms_ssim_module: MS_SSIM,
        grid_lambda_start: np.ndarray,
        grid_lambda_end: np.ndarray,
        grid_prior: np.ndarray,
        max_iter: int,
        make_x_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        make_z_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        device: torch.device,
        initialize_noise_level: float = 1e-3,
        valid_region_mask: Optional[np.ndarray] = None) -> Dict[GridSearchParams, GridSearchReconstructions]:
    n_examples, height, width = example_stimuli.shape

    assert n_examples % batch_size == 0, 'number of example images must be multiple of batch size'

    example_stimuli_torch = torch.tensor(example_stimuli, dtype=torch.float32, device=device)

    rescaled_example_stimuli = image_rescale_0_1(example_stimuli_torch)
    del example_stimuli_torch

    ################################################################################
    create_HQS_Z_prob_fn = construct_create_batched_denoiser_HQS_Z_problem_fn(
        (height, width),
        image_range_tuple,
        device,
        valid_region_mask=valid_region_mask
    )

    grid_search_pbar = tqdm.tqdm(
        total=grid_prior.shape[0] * grid_lambda_start.shape[0] * grid_lambda_end.shape[0])

    ret_dict = {}
    for ix, grid_params in enumerated_product(grid_prior, grid_lambda_start, grid_lambda_end):

        prior_weight, lambda_start, lambda_end = grid_params

        output_dict_key = GridSearchParams(lambda_start, lambda_end, prior_weight, max_iter)

        schedule_rho = make_logspaced_rho_schedule(ScheduleVal(lambda_start, lambda_end, prior_weight, max_iter))

        unblind_denoise_hqs_x_prob = MixNMatch_BatchKnownSeparable_Trial_GLM_ProxProblem(
            batch_size,
            typed_packed_model_tensors,
            stimulus_time_component,
            reconstruction_loss_fn,
            dtype=torch.float32
        ).to(device)

        unblind_denoise_hqs_z_prob = create_HQS_Z_prob_fn(batch_size)

        ################################################################################
        output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
        pbar = tqdm.tqdm(total=n_examples)

        for low in range(0, n_examples, batch_size):
            high = low + batch_size

            # put the spikes in a dict, and onto GPU
            typed_trial_spikes_torch = {
                key: torch.tensor(val[low:high, ...], dtype=torch.float32, device=device)
                for key, val in example_keyed_spikes.items()
            }

            unblind_denoise_hqs_x_prob_solver_iter = make_x_prob_solver_generator_fn()
            unblind_denoise_hqs_z_prob_solver_iter = make_z_prob_solver_generator_fn()

            initialize_z_tensor = torch.randn((batch_size, height, width),
                                              dtype=torch.float32) * initialize_noise_level

            unblind_denoise_hqs_x_prob.reinitialize_variables(initialized_z_const=None)
            unblind_denoise_hqs_x_prob.precompute_gensig_components(typed_trial_spikes_torch)
            unblind_denoise_hqs_z_prob.reinitialize_variables()

            _ = iter_rho_fixed_prior_hqs_solve(
                unblind_denoise_hqs_x_prob,
                iter(unblind_denoise_hqs_x_prob_solver_iter),
                unblind_denoise_hqs_z_prob,
                iter(unblind_denoise_hqs_z_prob_solver_iter),
                iter(schedule_rho),
                prior_weight,
                verbose=False,
                save_intermediates=False,
                observed_spikes=typed_trial_spikes_torch
            )

            denoise_hqs_reconstructed_image = unblind_denoise_hqs_x_prob.get_reconstructed_image()
            output_image_buffer_np[low:high, :, :] = denoise_hqs_reconstructed_image

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

        del output_image_buffer, unblind_denoise_hqs_x_prob, unblind_denoise_hqs_z_prob, reconstruction_rescaled
        grid_search_pbar.update(1)

    return ret_dict


def mixnmatch_batch_parallel_generate_flashed_hqs_reconstructions(
        example_keyed_spikes: Dict[str, np.ndarray],
        image_range_tuple: Tuple[float, float],
        typed_packed_model_tensors: Dict[str, PackedGLMTensors],
        reconstruction_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        glm_time_component: np.ndarray,
        reconstruction_hyperparams: GridSearchParams,
        make_x_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        make_z_prob_solver_generator_fn: Callable[[], Iterator[HQS_ParameterizedSolveFn]],
        max_batch_size: int,
        device: torch.device,
        initialize_noise_level: float = 1e-3,
        valid_region_mask: Optional[np.ndarray] = None) \
        -> np.ndarray:
    _first_key = list(example_keyed_spikes.keys())[0]
    _first_model = typed_packed_model_tensors[_first_key]
    _first_spikes = example_keyed_spikes[_first_key]

    n_examples, n_cells, n_bins_observed = _first_spikes.shape
    _, height, width = _first_model.spatial_filters.shape

    #################################################################################
    # get the hyperparameters
    lambda_start, lambda_end = reconstruction_hyperparams.lambda_start, reconstruction_hyperparams.lambda_end
    prior_weight = reconstruction_hyperparams.prior_weight
    max_iter = reconstruction_hyperparams.max_iter
    schedule_rho = make_logspaced_rho_schedule(ScheduleVal(lambda_start, lambda_end, prior_weight, max_iter))

    #################################################################################
    create_HQS_Z_prob_fn = construct_create_batched_denoiser_HQS_Z_problem_fn(
        (height, width),
        image_range_tuple,
        device,
        valid_region_mask=valid_region_mask
    )

    # run the first N-1 iterations
    output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
    pbar = tqdm.tqdm(total=n_examples)
    for low in range(0, n_examples, max_batch_size):
        high = min(low + max_batch_size, n_examples)
        eff_batch_size = high - low

        # put the spikes in a dict, and onto GPU
        typed_trial_spikes_torch = {
            key: torch.tensor(val[low:high, ...], dtype=torch.float32, device=device)
            for key, val in example_keyed_spikes.items()
        }

        ################################################################################
        # make the models and transfer to device
        unblind_denoise_hqs_x_prob = MixNMatch_BatchKnownSeparable_Trial_GLM_ProxProblem(
            eff_batch_size,
            typed_packed_model_tensors,
            glm_time_component,
            reconstruction_loss_fn,
            dtype=torch.float32
        ).to(device)

        unblind_denoise_hqs_z_prob = create_HQS_Z_prob_fn(eff_batch_size)
        ################################################################################

        unblind_denoise_hqs_x_prob_solver_iter = make_x_prob_solver_generator_fn()
        unblind_denoise_hqs_z_prob_solver_iter = make_z_prob_solver_generator_fn()

        initialize_z_tensor = torch.randn((eff_batch_size, height, width),
                                          dtype=torch.float32) * initialize_noise_level

        unblind_denoise_hqs_x_prob.reinitialize_variables(initialized_z_const=initialize_z_tensor)
        unblind_denoise_hqs_x_prob.precompute_gensig_components(typed_trial_spikes_torch)
        unblind_denoise_hqs_z_prob.reinitialize_variables()

        _ = iter_rho_fixed_prior_hqs_solve(
            unblind_denoise_hqs_x_prob,
            iter(unblind_denoise_hqs_x_prob_solver_iter),
            unblind_denoise_hqs_z_prob,
            iter(unblind_denoise_hqs_z_prob_solver_iter),
            iter(schedule_rho),
            prior_weight,
            verbose=False,
            save_intermediates=False,
            observed_spikes=typed_trial_spikes_torch
        )

        denoise_hqs_reconstructed_image = unblind_denoise_hqs_x_prob.get_reconstructed_image()
        output_image_buffer_np[low:high, :, :] = denoise_hqs_reconstructed_image

        del unblind_denoise_hqs_x_prob, unblind_denoise_hqs_z_prob

        pbar.update(max_batch_size)

    return output_image_buffer_np
