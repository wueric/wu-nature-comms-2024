import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Callable, Tuple, List, Iterator, Optional, Union

import tqdm

from movie_upsampling import batch_compute_interval_overlaps, EMJitterFrame, SharedClockTimeUpsampleTransposeFlat, \
    SingleEMJitterFrame

from convex_optim_base.optim_base import SingleUnconstrainedProblem
from dejitter_recons.dejitter_glm_components import precompute_coupling_exp_args, precompute_feedback_exp_args
from denoise_inverse_alg.glm_inverse_alg import PackedGLMTensors, FeedbackOnlyPackedGLMTensors
from denoise_inverse_alg.hqs_alg import HQS_X_Problem, iter_rho_fixed_prior_hqs_solve, \
    MaskedUnblindDenoiserPrior_HQS_ZProb, Adam_HQS_XGenerator, DirectSolve_HQS_ZGenerator, AdamOptimParams, \
    HQS_ParameterizedSolveFn


def construct_magic_rescale_const(time_length_to_use: Union[float, int]):
    return 400.0 / (750.0 - 500.0 + time_length_to_use)


MAGIC_LOSS_RESCALE_CONST2 = construct_magic_rescale_const(500.0)


def create_gaussian_multinomial(sigma,
                                max_dist_to_test: int) -> Tuple[np.ndarray, np.ndarray]:
    shifts = -np.r_[-max_dist_to_test:max_dist_to_test + 1]
    mg_y, mg_x = np.meshgrid(shifts, shifts)

    _mg_x_flat = mg_x.reshape(-1)
    _mg_y_flat = mg_y.reshape(-1)
    # shape (loss_eval_batch, 2)
    _mg_xy_flat = np.stack([_mg_x_flat, _mg_y_flat], axis=1)
    dist_penalty = -np.sum(_mg_xy_flat * _mg_xy_flat, axis=1) / (0.5 * sigma * sigma)

    eval_at = np.exp(dist_penalty)

    normalized = eval_at / np.sum(eval_at)

    return normalized, _mg_xy_flat


class FBOnlyJointImageEyePositionLikelihood(nn.Module):
    '''
    Feedback-only version of the GLM

    Assumes batch size 1 for spikes and history

    This has no state variables but is supposed to be differentiable
        with respect to the image
    '''

    def __init__(self,
                 fb_glm_model_params: FeedbackOnlyPackedGLMTensors,
                 n_history_frames: int,
                 frame_transition_times: np.ndarray,
                 spike_bin_edges: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32,
                 magic_rescale_constant: float = MAGIC_LOSS_RESCALE_CONST2):
        super().__init__()
        self.n_bins_total = spike_bin_edges.shape[0] - 1
        self.n_frames = frame_transition_times.shape[0] - 1
        self.magic_rescale_constant = magic_rescale_constant

        stacked_spatial_filters = fb_glm_model_params.spatial_filters
        stacked_timecourse_filters = fb_glm_model_params.timecourse_filters
        stacked_feedback_filters = fb_glm_model_params.feedback_filters
        stacked_bias = fb_glm_model_params.bias

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_pixels = self.height * self.width
        self.n_history_frames = n_history_frames
        self.n_bins_filter = stacked_timecourse_filters.shape[1]

        self.spiking_loss_fn = spiking_loss_fn

        # precompute all of the quantities needed to upsample in time and do backprop
        forward_sel, forward_weights, backward_sel, backward_weights = batch_compute_interval_overlaps(
            frame_transition_times[None, :], spike_bin_edges[None, :])

        # store these quantities as buffers
        self.register_buffer('forward_sel', torch.tensor(forward_sel.squeeze(0), dtype=torch.long))
        self.register_buffer('backward_sel', torch.tensor(backward_sel.squeeze(0), dtype=torch.long))
        self.register_buffer('forward_weights', torch.tensor(forward_weights.squeeze(0), dtype=dtype))
        self.register_buffer('backward_weights', torch.tensor(backward_weights.squeeze(0), dtype=dtype))

        # store GLM quantities as buffers as well
        # shape (n_cells, n_pixels)
        stacked_flat_spat_filters = stacked_spatial_filters.reshape(self.n_cells, -1)
        self.register_buffer('stacked_flat_spat_filters', torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_timecourse_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_timecourse_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_timecourse_filters', torch.tensor(stacked_timecourse_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_feedback_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_feedback_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_feedback_filters', torch.tensor(stacked_feedback_filters, dtype=dtype))

        # shape (n_cells, 1)
        assert stacked_bias.shape == (self.n_cells, 1), f'stacked_bias must have shape {(self.n_cells, 1)}'
        self.register_buffer('stacked_bias', torch.tensor(stacked_bias, dtype=dtype))

        # Create buffers for pre-computed quantities (stuff that doesn't for a fixed spike train)
        self.register_buffer('precomputed_feedback_gensig',
                             torch.zeros((self.n_cells, self.n_bins_total - self.n_bins_filter + 1),
                                         dtype=dtype))
        self.register_buffer('precomputed_history_frames',
                             torch.zeros((n_history_frames, self.n_cells), dtype=dtype))

        self.upsampler = SharedClockTimeUpsampleTransposeFlat.apply
        self.grid_jitterer = SingleEMJitterFrame.apply

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        '''

        :param observed_spikes: shape (n_cells, n_bins)
        :return:
        '''
        with torch.no_grad():
            feedback_component = precompute_feedback_exp_args(
                self.stacked_feedback_filters,
                observed_spikes[None, :, :]
            ).squeeze(0)
            precomputed = feedback_component + self.stacked_bias
            self.precomputed_feedback_gensig.data[:] = precomputed.data[:]

    def precompute_history_frames(self, history_frames: torch.Tensor) -> None:
        '''

        :param history_frames: shape (n_history_frames, height, width)
        :return:
        '''
        with torch.no_grad():
            # shape (n_history_frames, n_pix)
            history_frames_flat = history_frames.reshape(self.n_history_frames, self.n_pixels)

            # shape (1, n_cells, n_pixels) @ (n_history_frames, n_pix, 1)
            # -> (n_history_frames, n_cells, 1) -> (n_history_frames, n_cells)
            spat_filt_applied = (
                    self.stacked_flat_spat_filters[None, :, :] @ history_frames_flat[:, :, None]).squeeze(2)
            self.precomputed_history_frames.data[:] = spat_filt_applied.data[:]

    def forward(self,
                single_image: torch.Tensor,
                observed_spikes: torch.Tensor,
                eye_movements: torch.Tensor,
                time_mask: torch.Tensor):
        '''

        :param single_image: shape (height, width)
        :param observed_spikes: shape (n_cells, n_bins)
        :param eye_movements: shape (n_grid, n_jittered_frames, 2), int64_t, each eye movement
            is associated with the corresponding frame transition time provided in the constructor
        :param time_mask: shape (n_bins - n_bins_filter, )
        :return: shape (n_grid, )
        '''

        n_grid = eye_movements.shape[0]

        grid_jittered_frames = self.grid_jitterer(single_image, eye_movements)

        # apply the spatial filters
        # shape (n_grid, n_jittered_frames, n_pix)
        grid_jittered_frames_flat = grid_jittered_frames.reshape(n_grid, -1, self.n_pixels)

        # shape (1, 1, n_cells, n_pix) @ (n_grid, n_jittered_frames, n_pix, 1)
        # -> (n_grid, n_jittered_frames, n_cells, 1) -> (n_grid, n_jittered_frames, n_cells)
        grid_jittered_spat_applied = (
                self.stacked_flat_spat_filters[None, None, :, :] @ grid_jittered_frames_flat[:, :, :, None]).squeeze(3)

        # shape (n_grid, n_frames_total, n_cells)
        spat_filt_applied = torch.concat([self.precomputed_history_frames[None, :, :].expand(n_grid, -1, -1),
                                          grid_jittered_spat_applied], dim=1)

        # upsample, can use the normal upsampler
        # (4) upsample the whole thing
        upsampled_movie = self.upsampler(spat_filt_applied, self.forward_sel, self.forward_weights,
                                         self.backward_sel, self.backward_weights)

        time_filt_applied = F.conv1d(upsampled_movie,
                                     self.stacked_timecourse_filters[:, None, :],
                                     groups=self.n_cells)

        gen_sig = time_filt_applied + self.precomputed_feedback_gensig

        # shape (n_grid, n_cells, n_bins_observed - n_bins_filter + 1)
        spiking_loss = self.spiking_loss_fn(gen_sig[:, :, :-1],
                                            observed_spikes[None, :, self.n_bins_filter:])

        # shape (n_grid, n_cells, n_bins_observed - n_bins_filter + 1)
        spiking_loss_masked = spiking_loss * time_mask[None, None, :]

        # we need to fix the loss scaling here: the reasoning behind this is as follows:
        # since according to Bayes we should gain more certainty when we observe more
        # data, we should not divide out by the total number bins.
        # Instead, to get the scaling that we want, we will sum over the bins
        # and then divide by a constant that achieves the correct scaling when all of the
        # bins are observed
        # shape (n_grid, )
        sum_loss = torch.sum(spiking_loss_masked, dim=(1, 2)) * self.magic_rescale_constant

        return sum_loss


class JointImageEyePositionLikelihood(nn.Module):
    '''
    Assumes batch size 1 for spikes and history

    This has no state variables but is supposed to be differentiable
        with respect to the image;

    This also requires a slightly different custom jitter routine
        than what we have been using. This is because
        we need to be able to backprop over all of the possible
        eye positions into the same image...
    '''

    def __init__(self,
                 glm_model_params: PackedGLMTensors,
                 n_history_frames: int,
                 frame_transition_times: np.ndarray,
                 spike_bin_edges: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32,
                 magic_rescale_constant: float = MAGIC_LOSS_RESCALE_CONST2):
        super().__init__()

        self.n_bins_total = spike_bin_edges.shape[0] - 1
        self.n_frames = frame_transition_times.shape[0] - 1
        self.magic_rescale_constant = magic_rescale_constant

        stacked_spatial_filters = glm_model_params.spatial_filters
        stacked_timecourse_filters = glm_model_params.timecourse_filters
        stacked_feedback_filters = glm_model_params.feedback_filters
        stacked_coupling_filters = glm_model_params.coupling_filters
        stacked_bias = glm_model_params.bias
        coupling_idx_sel = glm_model_params.coupling_indices

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_pixels = self.height * self.width
        self.n_history_frames = n_history_frames
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.max_coupled_cells = stacked_coupling_filters.shape[1]

        self.spiking_loss_fn = spiking_loss_fn

        # precompute all of the quantities needed to upsample in time and do backprop
        forward_sel, forward_weights, backward_sel, backward_weights = batch_compute_interval_overlaps(
            frame_transition_times[None, :], spike_bin_edges[None, :])

        # store these quantities as buffers
        self.register_buffer('forward_sel', torch.tensor(forward_sel.squeeze(0), dtype=torch.long))
        self.register_buffer('backward_sel', torch.tensor(backward_sel.squeeze(0), dtype=torch.long))
        self.register_buffer('forward_weights', torch.tensor(forward_weights.squeeze(0), dtype=dtype))
        self.register_buffer('backward_weights', torch.tensor(backward_weights.squeeze(0), dtype=dtype))

        # store GLM quantities as buffers as well
        # shape (n_cells, n_pixels)
        stacked_flat_spat_filters = stacked_spatial_filters.reshape(self.n_cells, -1)
        self.register_buffer('stacked_flat_spat_filters', torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_timecourse_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_timecourse_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_timecourse_filters', torch.tensor(stacked_timecourse_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_feedback_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_feedback_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_feedback_filters', torch.tensor(stacked_feedback_filters, dtype=dtype))

        # shape (n_cells, max_coupled_cells, n_bins_filter)
        assert stacked_coupling_filters.shape == (self.n_cells, self.max_coupled_cells, self.n_bins_filter), \
            f'stacked_coupling_filters must have shape {(self.n_cells, self.max_coupled_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_coupling_filters', torch.tensor(stacked_coupling_filters, dtype=dtype))

        # shape (n_cells, 1)
        assert stacked_bias.shape == (self.n_cells, 1), f'stacked_bias must have shape {(self.n_cells, 1)}'
        self.register_buffer('stacked_bias', torch.tensor(stacked_bias, dtype=dtype))

        # shape (n_cells, max_coupled_cells), integer LongTensor
        assert coupling_idx_sel.shape == (self.n_cells, self.max_coupled_cells), \
            f'coupling_idx_sel must have shape {(self.n_cells, self.max_coupled_cells)}'
        self.register_buffer('coupled_sel', torch.tensor(coupling_idx_sel, dtype=torch.long))

        # Create buffers for pre-computed quantities (stuff that doesn't for a fixed spike train)
        self.register_buffer('precomputed_feedback_coupling_gensig',
                             torch.zeros((self.n_cells, self.n_bins_total - self.n_bins_filter + 1),
                                         dtype=dtype))
        self.register_buffer('precomputed_history_frames',
                             torch.zeros((n_history_frames, self.n_cells), dtype=dtype))

        self.upsampler = SharedClockTimeUpsampleTransposeFlat.apply
        self.grid_jitterer = SingleEMJitterFrame.apply

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        '''

        :param observed_spikes: shape (n_cells, n_bins)
        :return:
        '''
        with torch.no_grad():
            coupling_component = precompute_coupling_exp_args(
                self.stacked_coupling_filters,
                self.coupled_sel,
                observed_spikes[None, :, :]
            ).squeeze(0)

            feedback_component = precompute_feedback_exp_args(
                self.stacked_feedback_filters,
                observed_spikes[None, :, :]
            ).squeeze(0)

            precomputed = coupling_component + feedback_component + self.stacked_bias
            self.precomputed_feedback_coupling_gensig.data[:] = precomputed.data[:]

    def precompute_history_frames(self, history_frames: torch.Tensor) -> None:
        '''

        :param history_frames: shape (n_history_frames, height, width)
        :return:
        '''
        with torch.no_grad():
            # shape (n_history_frames, n_pix)
            history_frames_flat = history_frames.reshape(self.n_history_frames, self.n_pixels)

            # shape (1, n_cells, n_pixels) @ (n_history_frames, n_pix, 1)
            # -> (n_history_frames, n_cells, 1) -> (n_history_frames, n_cells)
            spat_filt_applied = (
                    self.stacked_flat_spat_filters[None, :, :] @ history_frames_flat[:, :, None]).squeeze(2)
            self.precomputed_history_frames.data[:] = spat_filt_applied.data[:]

    def forward(self,
                single_image: torch.Tensor,
                observed_spikes: torch.Tensor,
                eye_movements: torch.Tensor,
                time_mask: torch.Tensor):
        '''

        :param single_image: shape (height, width)
        :param observed_spikes: shape (n_cells, n_bins)
        :param eye_movements: shape (n_grid, n_jittered_frames, 2), int64_t, each eye movement
            is associated with the corresponding frame transition time provided in the constructor
        :param time_mask: shape (n_bins - n_bins_filter, )
        :return: shape (n_grid, )
        '''

        n_grid = eye_movements.shape[0]

        grid_jittered_frames = self.grid_jitterer(single_image, eye_movements)

        # apply the spatial filters
        # shape (n_grid, n_jittered_frames, n_pix)
        grid_jittered_frames_flat = grid_jittered_frames.reshape(n_grid, -1, self.n_pixels)

        # shape (1, 1, n_cells, n_pix) @ (n_grid, n_jittered_frames, n_pix, 1)
        # -> (n_grid, n_jittered_frames, n_cells, 1) -> (n_grid, n_jittered_frames, n_cells)
        grid_jittered_spat_applied = (
                self.stacked_flat_spat_filters[None, None, :, :] @ grid_jittered_frames_flat[:, :, :, None]).squeeze(3)

        # shape (n_grid, n_frames_total, n_cells)
        spat_filt_applied = torch.concat([self.precomputed_history_frames[None, :, :].expand(n_grid, -1, -1),
                                          grid_jittered_spat_applied], dim=1)

        # upsample, can use the normal upsampler
        # (4) upsample the whole thing
        upsampled_movie = self.upsampler(spat_filt_applied, self.forward_sel, self.forward_weights,
                                         self.backward_sel, self.backward_weights)

        time_filt_applied = F.conv1d(upsampled_movie,
                                     self.stacked_timecourse_filters[:, None, :],
                                     groups=self.n_cells)

        gen_sig = time_filt_applied + self.precomputed_feedback_coupling_gensig

        # shape (n_grid, n_cells, n_bins_observed - n_bins_filter + 1)
        spiking_loss = self.spiking_loss_fn(gen_sig[:, :, :-1],
                                            observed_spikes[None, :, self.n_bins_filter:])

        # shape (n_grid, n_cells, n_bins_observed - n_bins_filter + 1)
        spiking_loss_masked = spiking_loss * time_mask[None, None, :]

        # we need to fix the loss scaling here: the reasoning behind this is as follows:
        # since according to Bayes we should gain more certainty when we observe more
        # data, we should not divide out by the total number bins.
        # Instead, to get the scaling that we want, we will sum over the bins
        # and then divide by a constant that achieves the correct scaling when all of the
        # bins are observed
        # shape (n_grid, )
        sum_loss = torch.sum(spiking_loss_masked, dim=(1, 2)) * self.magic_rescale_constant

        return sum_loss


class EMProxProblem(SingleUnconstrainedProblem,
                    HQS_X_Problem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'
    EYE_MOVEMENTS_KWARGS = 'eye_movements'
    PROB_TABLE_KWARGS = 'prob_table'
    TIME_MASK_KWARGS = 'time_mask'

    def __init__(self,
                 model_params: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors],
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 n_history_frames: int,
                 frame_transition_times: np.ndarray,
                 spike_bin_edges: np.ndarray,
                 rho: float,
                 dtype: torch.dtype = torch.float32,
                 magic_rescale_constant: float = MAGIC_LOSS_RESCALE_CONST2):
        super().__init__()

        stacked_spatial_filters = model_params.spatial_filters
        self.rho = rho

        if isinstance(model_params, PackedGLMTensors):
            self.prox_likelihood = JointImageEyePositionLikelihood(
                model_params,
                n_history_frames, frame_transition_times, spike_bin_edges,
                spiking_loss_fn, dtype=dtype, magic_rescale_constant=magic_rescale_constant
            )
        else:
            self.prox_likelihood = FBOnlyJointImageEyePositionLikelihood(
                model_params,
                n_history_frames, frame_transition_times, spike_bin_edges,
                spiking_loss_fn, dtype=dtype, magic_rescale_constant=magic_rescale_constant
            )

        self.height, self.width = stacked_spatial_filters.shape[1], stacked_spatial_filters.shape[2]

        # Create a buffer for the z const for HQS
        self.register_buffer('z_const_tensor', torch.zeros((self.height, self.width), dtype=dtype))

        self.image = nn.Parameter(torch.empty((self.height, self.width), dtype=dtype),
                                  requires_grad=True)
        nn.init.uniform_(self.image, -1e-2, 1e-2)

    def set_init_guess(self, guess: torch.Tensor) -> None:
        self.image.data[:] = guess.data[:]

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data[:]

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        self.prox_likelihood.precompute_gensig_components(observed_spikes)

    def precompute_history_frames(self, history_frames: torch.Tensor) -> None:
        self.prox_likelihood.precompute_history_frames(history_frames)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        # shape (height, width)
        image = args[self.IMAGE_IDX_ARGS]

        # shape (n_cells, n_bins)
        observed_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (n_grid, n_frames_target, 2)
        eye_movements = kwargs[self.EYE_MOVEMENTS_KWARGS]

        # shape (n_grid, )
        prob_table = kwargs[self.PROB_TABLE_KWARGS]

        # shape (n_bins - n_bins_filter)
        time_mask = kwargs[self.TIME_MASK_KWARGS]

        # shape (n_grid, )
        log_likelihood = self.prox_likelihood(image, observed_spikes,
                                              eye_movements, time_mask)

        weighted = (log_likelihood[None, :] @ prob_table[:, None]).squeeze()

        diff = image - self.z_const_tensor
        prox_penalty = 0.5 * self.rho * torch.sum(diff * diff)

        return weighted + prox_penalty

    def forward(self, **kwargs):
        return self._eval_smooth_loss(self.image, **kwargs)

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def get_output_image(self) -> torch.Tensor:
        return self.image.detach().clone()

    def get_reconstructed_image(self) -> np.ndarray:
        return self.image.detach().cpu().numpy()


def make_adam_em_hqs_iters(rho_sched: np.ndarray,
                           init_x_prob_iters: List[AdamOptimParams],
                           default_x_prob_iters: AdamOptimParams) \
        -> Tuple[Iterator[float], Iterator[HQS_ParameterizedSolveFn], Iterator[HQS_ParameterizedSolveFn]]:
    x_generator = Adam_HQS_XGenerator(init_x_prob_iters,
                                      default_x_prob_iters)
    z_generator = DirectSolve_HQS_ZGenerator()
    return rho_sched, x_generator, z_generator


Get_iter_fn = Callable[
    [], Tuple[Iterator[float], Iterator[HQS_ParameterizedSolveFn], Iterator[HQS_ParameterizedSolveFn]]]


class Stateless_EM_SIR_ParticleFilter(nn.Module):
    '''
    Wrapper for SIR particle filter that does not keep internal state

    This is useful for a faster version of the particle filter where we reduce
        the number of trajectories that go into the EM optimization using the
        following tricks:

        (1) Merging particles with the same trajectory: a lot of the time
            we will have particles with the same trajectory, in which case
            they can be merged to reduce the number of distinct trajectories
        (2) Getting rid of particles with extremely low probabilities

    This module does not keep any state variables; this is because the number of
        particles at any given time may vary because of the above tricks and
        so tracking state explicitly here is silly.

    Non-differentiable for obvious reaons, no forward method

    Uses multinomial for proposal distribution, since the jitter model
        is acutally a discretized integer random walk

    Has a copy of the encoding (likelihood) model so that we can compute
        particle weight updates easily
    '''

    def __init__(self,
                 model_params: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors],
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 n_history_frames: int,
                 frame_transition_times: np.ndarray,
                 spike_bin_edges: np.ndarray,
                 proposal_shift_distribution: Tuple[np.ndarray, np.ndarray],
                 likelihood_scale: float,
                 dtype: torch.dtype = torch.float32,
                 magic_rescale_constant: float = MAGIC_LOSS_RESCALE_CONST2):
        super().__init__()

        self.n_bins_total = spike_bin_edges.shape[0] - 1
        self.n_frames = frame_transition_times.shape[0] - 1
        self.n_eye_movements = self.n_frames - n_history_frames
        self.likelihood_scale = likelihood_scale

        self.dtype = dtype

        if isinstance(model_params, PackedGLMTensors):
            self.likelihood_model = JointImageEyePositionLikelihood(
                model_params,
                n_history_frames, frame_transition_times, spike_bin_edges,
                spiking_loss_fn, dtype=dtype, magic_rescale_constant=magic_rescale_constant
            )
        else:
            self.likelihood_model = FBOnlyJointImageEyePositionLikelihood(
                model_params,
                n_history_frames, frame_transition_times, spike_bin_edges,
                spiking_loss_fn, dtype=dtype, magic_rescale_constant=magic_rescale_constant
            )

        # store the probability table
        jitter_probs, jitter_offset_coords = proposal_shift_distribution
        # shape (n_jitter_pos, )
        self.register_buffer('jitter_prob_table', torch.tensor(jitter_probs, dtype=torch.float32))

        # shape (n_jitter_pos, 2)
        self.register_buffer('jitter_offset_coords', torch.tensor(jitter_offset_coords, dtype=torch.long))

    def propose_compute_weights(self,
                                particle_trajs: torch.Tensor,
                                log_particle_weights: torch.Tensor,
                                estimated_image: torch.Tensor,
                                prev_estimated_image: torch.Tensor,
                                observed_spikes: torch.Tensor,
                                frame_ix: int,
                                time_mask: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        :param particle_trajs: shape (n_particles, n_jitter_frames, 2)
        :param log_particle_weights shape (n_particles, )
        :param estimated_image: shape (height, width)
        :param prev_estimated_image: shape (height, width)
        :param observed_spikes: shape (n_observed_cells, n_bins)
        :param frame_ix: int
        :param time_mask:
        :return:
        '''

        with torch.no_grad():
            n_particles = particle_trajs.shape[0]

            previous_log_likelihood_weight = -self.likelihood_model(prev_estimated_image,
                                                                    observed_spikes,
                                                                    particle_trajs,
                                                                    time_mask)

            # shape (n_particles, ), type int64
            proposal_samples = torch.multinomial(self.jitter_prob_table, n_particles,
                                                 replacement=True)

            # shape (n_particles, 2), type int64
            selected_coordinates = self.jitter_offset_coords[proposal_samples, :]

            cloned_history = particle_trajs.detach().clone()

            prev_position = particle_trajs[:, frame_ix - 1, :]
            shift = prev_position + selected_coordinates
            cloned_history[:, frame_ix:, :] = shift[:, None, :]

            # now compute the un-normalized weight. In this case the weight is
            # simply the likelihood, since we use p(w_t \mid w_{t-1}) as the proposal
            # distribution
            # shape (n_particles, )
            log_likelihood_weight = -self.likelihood_model(estimated_image,
                                                           observed_spikes,
                                                           cloned_history,
                                                           time_mask)

            # shape (n_particles, )
            to_softmax = (-previous_log_likelihood_weight + log_likelihood_weight +
                          log_particle_weights) * self.likelihood_scale

            # to_softmax = (log_likelihood_weight + log_particle_weights) * self.likelihood_scale

            # shape (n_particles, )
            normalized_weight = F.softmax(to_softmax, dim=0)

            # shape (n_particles, )
            log_prob = torch.log(normalized_weight)

            return cloned_history, log_prob

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        self.likelihood_model.precompute_gensig_components(observed_spikes)

    def precompute_history_frames(self, history_frames: torch.Tensor) -> None:
        self.likelihood_model.precompute_history_frames(history_frames)


def resample_particles(particle_trajs: torch.Tensor,
                       log_particle_weights: torch.Tensor,
                       output_particle_num: int) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    :param particle_trajs: shape (n_particles_orig, n_jittered_frames, 2)
    :param log_particle_weights: shape (n_particles_orig, )
    :return:
    '''

    def first_nonzero(x, dim=0):
        nonz = (x > 0)
        values, indices = ((nonz.cumsum(dim) == 1) & nonz).max(dim)
        return indices

    device = particle_trajs.device
    dtype = log_particle_weights.dtype

    with torch.no_grad():
        random_offset = torch.rand((1,),
                                   dtype=dtype, device=device) / output_particle_num

        # shape (n_particles_orig, )
        sample_ix = (torch.arange(0, output_particle_num,
                                  dtype=dtype, device=device) / output_particle_num) + random_offset

        # shape (output_particle_num, )
        part_weights = torch.exp(log_particle_weights)
        cumsum = torch.cumsum(part_weights, dim=0)

        # shape (n_particles, )
        # resample_ix = torch.argmax(cumsum[None, :] >= sample_ix[:, None], dim=1)
        resample_ix = first_nonzero(cumsum[None, :] >= sample_ix[:, None], dim=1)

        resampled_particles = particle_trajs[resample_ix, :, :].contiguous()
        log_particle_weights = torch.zeros((output_particle_num,), device=device,
                                           dtype=log_particle_weights.dtype) - np.log(output_particle_num)

        return resampled_particles, log_particle_weights


def combine_particles_with_same_traj(particle_trajs: torch.Tensor,
                                     log_particle_weights: torch.Tensor,
                                     eps: float = 1e-9) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Combines particles that have the same trajectory, and sums their probabilities

    Combined numpy/torch routine, on CPU and GPU

    Overall routine that this is used for is kind of slow anyway,
        so performance is fine

    :param particle_trajs: shape (n_orig_particles, n_jittered_frames, 2)
    :param log_particle_weights: shape (n_orig_particles, )
    :return: shape (n_combined_particles, n_jittered_frames, 2)
        and shape (n_combined_particles, )
    '''

    def traj_tupleify(traj: np.ndarray):
        '''
        :param traj: shape (n_jittered_frames, 2)
        :return:
        '''
        # super sketch, but works
        return tuple(map(tuple, traj))

    device = particle_trajs.device
    dtype = log_particle_weights.dtype

    trajs_np = particle_trajs.detach().cpu().numpy()
    probs_np = np.exp(log_particle_weights.detach().cpu().numpy())

    n_particles_orig, n_jittered_frames, two = trajs_np.shape

    reweight_dict = {}
    for i in range(n_particles_orig):
        traj_tuple = traj_tupleify(trajs_np[i, :, :])
        if traj_tuple not in reweight_dict:
            reweight_dict[traj_tuple] = eps
        reweight_dict[traj_tuple] += probs_np[i]

    n_output_particles = len(reweight_dict)
    output_traj_np = np.zeros((n_output_particles, n_jittered_frames, two),
                              dtype=np.int64)
    output_probs_np = np.zeros((n_output_particles,), dtype=np.float64)
    for i, (traj_tuple, weight) in enumerate(reweight_dict.items()):
        traj_np = np.array(traj_tuple, dtype=np.int64)
        output_traj_np[i, :, :] = traj_np
        output_probs_np[i] = weight

    output_log_probs = np.log(output_probs_np)

    return (
        torch.tensor(output_traj_np, dtype=torch.long, device=device),
        torch.tensor(output_log_probs, dtype=dtype, device=device)
    )


def trim_low_probability_particles(particle_trajs: torch.Tensor,
                                   log_particle_weights: torch.Tensor,
                                   log_prob_cutoff: float) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''

    :param particle_trajs:
    :param log_particle_weights:
    :param log_prob_cutoff:
    :return:
    '''

    with torch.no_grad():
        good_particle_sel = log_particle_weights > log_prob_cutoff

        sel_log_prob = log_particle_weights[good_particle_sel]
        reweight_log_prob = torch.log(torch.softmax(sel_log_prob, dim=0))

        good_traj = particle_trajs[good_particle_sel, :, :].contiguous()
        return good_traj, reweight_log_prob


def trim_particle_sample_only(
        em_particle_sampler: Stateless_EM_SIR_ParticleFilter,
        observed_spikes: torch.Tensor,
        frame_ix_and_mask: Tuple[int, torch.Tensor],
        particles_weights: Tuple[torch.Tensor, torch.Tensor],
        image_guess: torch.Tensor,
        particle_number: int = 25,
        throwaway_log_prob: float = -6,
        do_resampling: bool = True) \
        -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''

    :param em_particle_sampler:
    :param observed_spikes:
    :param frame_ix_and_mask:
    :param particles_weights:
    :param image_guess:
    :param particle_number:
    :param throwaway_log_prob:
    :return:
    '''

    frame_ix, time_mask = frame_ix_and_mask

    prev_particle_trajs, prev_log_particle_weights = particles_weights

    particle_trajs, particle_log_prob = em_particle_sampler.propose_compute_weights(
        prev_particle_trajs,
        prev_log_particle_weights,
        image_guess,
        image_guess,
        observed_spikes,
        frame_ix,
        time_mask
    )

    # combine particles with the same trajectory
    same_traj_combined_particles, same_traj_combined_log_prob = combine_particles_with_same_traj(
        particle_trajs, particle_log_prob)

    # eliminate particles with too low probability
    trimmed_traj_particles, trimmed_log_prob = trim_low_probability_particles(
        same_traj_combined_particles, same_traj_combined_log_prob, throwaway_log_prob
    )

    # if this is the (user-specified) final iteration of the algorithm, we don't
    # want to resample, since resampled trajectories are harder to analyze
    if do_resampling:
        # now resample the particles
        resampled_trajs, resampled_log_probs = resample_particles(
            trimmed_traj_particles,
            trimmed_log_prob,
            particle_number
        )

        return image_guess, (resampled_trajs, resampled_log_probs)
    else:
        return image_guess, (trimmed_traj_particles, trimmed_log_prob)


def trim_particle_filter_em_iterate(
        em_particle_sampler: Stateless_EM_SIR_ParticleFilter,
        em_prox_problem: EMProxProblem,
        image_prior_problem: MaskedUnblindDenoiserPrior_HQS_ZProb,
        observed_spikes: torch.Tensor,
        frame_ix_and_mask: Tuple[int, torch.Tensor],
        particles_weights: Tuple[torch.Tensor, torch.Tensor],
        lambda_prior_weight: float,
        image_init_guess: torch.Tensor,
        get_em_iters: Get_iter_fn,
        particle_number: int = 25,
        em_inner_opt_verbose: bool = False,
        throwaway_log_prob: float = -6,
        do_resampling: bool = True) \
        -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''

    :param em_particle_sampler:
    :param em_prox_problem:
    :param image_prior_problem:
    :param observed_spikes:
    :param frame_ix_and_mask:
    :param traj_particles:
    :param lambda_prior_weight:
    :param image_init_guess:
    :param get_em_iters:
    :param em_inner_opt_verbose:
    :return:
    '''

    frame_ix, time_mask = frame_ix_and_mask

    em_prox_problem.set_init_guess(image_init_guess)
    em_prox_problem.assign_z(image_init_guess)

    particle_trajs, log_particle_weights = particles_weights

    with torch.no_grad():
        prob_table = torch.softmax(log_particle_weights, dim=0)

    em_hqs_rho_sched, em_hqs_x_solve_iter, em_hqs_z_solve_iter = get_em_iters()
    _ = iter_rho_fixed_prior_hqs_solve(
        em_prox_problem,
        iter(em_hqs_x_solve_iter),
        image_prior_problem,
        iter(em_hqs_z_solve_iter),
        iter(em_hqs_rho_sched),
        lambda_prior_weight,
        verbose=em_inner_opt_verbose,
        save_intermediates=False,
        observed_spikes=observed_spikes,
        eye_movements=particle_trajs,
        time_mask=time_mask,
        prob_table=prob_table
    )

    image_iter = image_prior_problem.get_reconstructed_image_torch().detach().clone()

    # update the variational distribution
    new_particle_trajs, new_particle_log_prob = em_particle_sampler.propose_compute_weights(
        particle_trajs,
        log_particle_weights,
        image_iter,
        image_init_guess,
        observed_spikes,
        frame_ix,
        time_mask
    )

    # combine particles with the same trajectory
    new_same_traj_combined_particles, new_same_traj_combined_log_prob = combine_particles_with_same_traj(
        new_particle_trajs, new_particle_log_prob)

    # eliminate particles with too low probability
    new_trimmed_traj_particles, new_trimmed_log_prob = trim_low_probability_particles(
        new_same_traj_combined_particles, new_same_traj_combined_log_prob, throwaway_log_prob)

    # if this is the (user-specified) final iteration of the algorithm, we don't
    # want to resample, since resampled trajectories are harder to analyze
    if do_resampling:
        # now resample the particles
        resampled_trajs, resampled_log_probs = resample_particles(
            new_trimmed_traj_particles,
            new_trimmed_log_prob,
            particle_number
        )

        return image_iter, (resampled_trajs, resampled_log_probs)
    else:
        return image_iter, (new_trimmed_traj_particles, new_trimmed_log_prob)


def batched_time_mask_history_and_frame_transition(
        history_frames: np.ndarray,
        all_frame_transition_times: np.ndarray,
        spike_bin_times: np.ndarray,
        spike_times: np.ndarray,
        integration_num_samples: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''

    :param history_frames: shape (batch, n_history_frames, height, width)
    :param all_frame_transition_times: shape (batch, n_frames_total + 1)
    :param spike_bin_times: shape (batch, n_bins + 1)
    :param integration_num_samples: int
    :return:
    '''

    batch, n_history_frames, height, width = history_frames.shape
    first_target_frame_ix = n_history_frames + 1

    # shape (batch, )
    first_target_frame_time = all_frame_transition_times[:, first_target_frame_ix]
    max_integration_sample = first_target_frame_time + integration_num_samples

    # we want to trim everything that occurs after max_integration_sample
    # that does not overlap with the associated bins
    # This is to save computing power (should be a factor of 2.5x for the integration
    # time analysis)

    # To be lazy, we simply take the maximum value across the batch
    # since the batches should be pretty close in timing
    # we might be off by about 1ms or so, which is small in the grand scheme of things
    max_integration_bin_ix = np.max(np.argmax(spike_bin_times > max_integration_sample[:, None], axis=1))
    max_integration_bin_end = spike_bin_times[:, max_integration_bin_ix + 1]

    max_frame_transition_ix = np.max(np.argmax(all_frame_transition_times > max_integration_bin_end[:, None], axis=1))
    subset_spike_bin_times = spike_bin_times[:, 0:max_integration_bin_ix + 1]
    subset_frame_transition_times = all_frame_transition_times[:, :max_frame_transition_ix + 1]

    return (
        history_frames,
        subset_frame_transition_times,
        spike_times[:, :, :max_integration_bin_ix],
        subset_spike_bin_times
    )


def time_mask_history_and_frame_transition(
        history_frames: np.ndarray,
        all_frame_transition_times: np.ndarray,
        spike_bin_times: np.ndarray,
        spike_times: np.ndarray,
        integration_num_samples: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''

    :param history_frames:
    :param all_frame_transition_times:
    :param spike_bin_times:
    :param lag_spike_bins:
    :param integration_time_samples: number of samples (units of recording array samples)
        that we want to perform the reconstruction over, starting at the time of the first
        appearance of the target frame
    :return:
    '''

    n_history_frames, height, width = history_frames.shape

    first_target_frame_ix = n_history_frames + 1
    first_target_frame_time = all_frame_transition_times[first_target_frame_ix]

    max_integration_sample = first_target_frame_time + integration_num_samples

    # we want to trim everything that occurs after max_integration_sample
    # that does not overlap with the associated bins
    # This is to save computing power (should be a factor of 2.5x for the integration
    # time analysis)
    max_integration_bin_ix = np.argmax(spike_bin_times > max_integration_sample)
    max_integration_bin_end = spike_bin_times[max_integration_bin_ix + 1]

    max_frame_transition_ix = np.argmax(all_frame_transition_times > max_integration_bin_end)

    subset_spike_bin_times = spike_bin_times[0:max_integration_bin_ix + 1]
    subset_frame_transition_times = all_frame_transition_times[:max_frame_transition_ix + 1]

    return (
        history_frames,
        subset_frame_transition_times,
        spike_times[:, :max_integration_bin_ix],
        subset_spike_bin_times
    )


def non_online_joint_em_estimation2(
        packed_glm_tensors: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors],
        unblind_denoiser_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        history_frames: np.ndarray,
        frame_transition_times: np.ndarray,
        observed_spikes: np.ndarray,
        spike_bin_times: np.ndarray,
        image_valid_mask: np.ndarray,
        spike_bin_width: int,
        lag_spike_bins: int,
        n_particles: int,
        proposal_shift_distribution: Tuple[np.ndarray, np.ndarray],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lambda_prior_weight: float,
        em_get_iter_fn: Get_iter_fn,
        em_update_get_iter_fn: Get_iter_fn,
        device: torch.device,
        image_init_guess: Optional[np.ndarray] = None,
        em_inner_opt_verbose: bool = False,
        throwaway_log_prob: float = -6,
        likelihood_scale: float = 2e-2,
        return_intermediate_em_results: bool = False,
        compute_image_every_n: int = 1,
        magic_rescale_constant=MAGIC_LOSS_RESCALE_CONST2):
    '''
    This algorithm assumes that ALL of the spike bins are visible at reconstruction time
        (i.e. this is NOT an online algorithm, we evaluate encoding log-likelihoods given
        all of the spikes)

    The notion of time lag no longer really applies, since we can see all of the spikes.
        This means that we only need one mask, corresponding to all of the spikes

    There are several advantages to doing EM in this way over the online method
        (1) It will probably be easier to produce a decent initial image that
            can be used to estimate eye movements, because we have more spikes
        (2) We don't need to fuss with tuning hyperparameters for doing the intermediate
            timestep reconstructions, since the magnitude of the encoding log-likelihood
            will be about the same for every intermediate step here

    :param packed_glm_tensors:
    :param unblind_denoiser_callable:
    :param history_frames:
    :param frame_transition_times:
    :param observed_spikes:
    :param spike_bin_times:
    :param image_valid_mask:
    :param spike_bin_width:
    :param lag_spike_bins:
    :param n_particles:
    :param proposal_shift_distribution:
    :param loss_fn:
    :param lambda_prior_weight:
    :param em_get_iter_fn:
    :param device:
    :param image_init_guess:
    :param em_inner_opt_verbose:
    :param resample_thresh:
    :param return_intermediate_em_results:
    :return:
    '''

    n_frames = frame_transition_times.shape[0] - 1
    n_history_frames, height, width = history_frames.shape
    n_eye_movements = n_frames - n_history_frames

    target_frame_transition_times = frame_transition_times[n_history_frames:]
    delta_lag = np.round(target_frame_transition_times + lag_spike_bins).astype(np.int64)
    max_lag_ix = np.argmax(np.argwhere(delta_lag < spike_bin_times[-1]).squeeze())
    time_estim_points = np.round((delta_lag[:max_lag_ix] - spike_bin_times[0]) / spike_bin_width).astype(np.int64)

    ########################################################################3
    # Build the modules that are needed to do the computation
    particle_filter_mod = Stateless_EM_SIR_ParticleFilter(
        packed_glm_tensors,
        loss_fn,
        n_history_frames,
        frame_transition_times,
        spike_bin_times,
        proposal_shift_distribution,
        likelihood_scale,
        dtype=torch.float32,
        magic_rescale_constant=magic_rescale_constant
    ).to(device)

    em_x_prob = EMProxProblem(
        packed_glm_tensors,
        loss_fn,
        n_history_frames,
        frame_transition_times,
        spike_bin_times,
        10.0,  # placeholder value
        dtype=torch.float32,
        magic_rescale_constant=magic_rescale_constant
    ).to(device)

    z_prob = MaskedUnblindDenoiserPrior_HQS_ZProb(
        unblind_denoiser_callable, (height, width), image_valid_mask,
        0.1,  # useless placeholder value
        prior_lambda=lambda_prior_weight,
        dtype=torch.float32
    ).to(device)

    n_bins_filter = particle_filter_mod.likelihood_model.n_bins_filter

    #########################################################################
    # Put data onto GPU
    # (not yet mixed precision, we may have to do that later)
    # shape (n_cells, n_bins)
    spikes_torch = torch.tensor(observed_spikes, dtype=torch.float32, device=device)

    # shape (n_history_frames, height, width)
    history_frames_torch = torch.tensor(history_frames, dtype=torch.float32, device=device)

    if image_init_guess is not None:
        image_torch = torch.tensor(image_init_guess, dtype=torch.float32, device=device)
    else:
        image_torch = torch.randn((height, width), dtype=torch.float32, device=device) * 0.1

    em_x_prob.assign_z(image_torch.detach().clone())

    mask_torch = torch.ones((spike_bin_times.shape[0] - n_bins_filter - 1,),
                            dtype=torch.float32, device=device)

    particle_trajs = torch.zeros((n_particles, n_eye_movements, 2), dtype=torch.long, device=device)
    particle_log_probs = torch.zeros((n_particles,), dtype=torch.float32, device=device) - np.log(n_particles)
    particle_and_log_prob = (particle_trajs, particle_log_probs)

    #########################################################################

    #########################################################################
    # precompute all of the quantities that require precomputing
    particle_filter_mod.precompute_gensig_components(spikes_torch)
    particle_filter_mod.precompute_history_frames(history_frames_torch)

    em_x_prob.precompute_gensig_components(spikes_torch)
    em_x_prob.precompute_history_frames(history_frames_torch)
    #########################################################################

    intermediates = []
    pbar = tqdm.tqdm(total=time_estim_points.shape[0] - 1)
    prev_iter_im_estimated = False
    is_first_estimate = True
    for count, timestep in enumerate(range(1, time_estim_points.shape[0])):

        is_last_iter = (timestep == (time_estim_points.shape[0] - 1))

        # specify whether we want to recompute the image
        # every N iterations
        if count % compute_image_every_n == 0:
            image_torch, particle_and_log_prob = trim_particle_filter_em_iterate(
                particle_filter_mod,
                em_x_prob,
                z_prob,
                spikes_torch,
                (timestep, mask_torch),
                particle_and_log_prob,
                lambda_prior_weight,
                image_torch,
                em_get_iter_fn if is_first_estimate else em_update_get_iter_fn,
                em_inner_opt_verbose=em_inner_opt_verbose,
                throwaway_log_prob=throwaway_log_prob,
                particle_number=n_particles,
                do_resampling=not is_last_iter
            )
            prev_iter_im_estimated = True
            is_first_estimate = False
        else:
            image_torch, particle_and_log_prob = trim_particle_sample_only(
                particle_filter_mod,
                spikes_torch,
                (timestep, mask_torch),
                particle_and_log_prob,
                image_torch,
                particle_number=n_particles,
                throwaway_log_prob=throwaway_log_prob,
                do_resampling=not is_last_iter
            )
            prev_iter_im_estimated = False

        if return_intermediate_em_results:
            image_np = image_torch.detach().cpu().numpy()
            traj_np = particle_and_log_prob[0].detach().cpu().numpy()
            part_weights_np = particle_and_log_prob[1].detach().cpu().numpy()
            intermediates.append((image_np, traj_np, part_weights_np))

        pbar.update(1)
    pbar.close()

    if not prev_iter_im_estimated:
        # we need to estimate the image again before the output
        particle_trajs, particle_log_probs = particle_and_log_prob
        with torch.no_grad():
            prob_table = torch.softmax(particle_log_probs, dim=0)
        em_hqs_rho_sched, em_hqs_x_solve_iter, em_hqs_z_solve_iter = em_get_iter_fn()
        _ = iter_rho_fixed_prior_hqs_solve(
            em_x_prob,
            iter(em_hqs_x_solve_iter),
            z_prob,
            iter(em_hqs_z_solve_iter),
            iter(em_hqs_rho_sched),
            lambda_prior_weight,
            verbose=em_inner_opt_verbose,
            save_intermediates=False,
            observed_spikes=spikes_torch,
            eye_movements=particle_trajs,
            time_mask=mask_torch,
            prob_table=prob_table
        )

        image_torch = z_prob.get_reconstructed_image_torch().detach().clone()

    image_np = image_torch.detach().cpu().numpy()
    traj_np = particle_and_log_prob[0].detach().cpu().numpy()
    part_weights_np = particle_and_log_prob[1].detach().cpu().numpy()

    if return_intermediate_em_results:
        return image_np, traj_np, part_weights_np, intermediates
    return image_np, traj_np, part_weights_np
