from typing import Callable, List, Iterator, Tuple, Union
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from movie_upsampling import batch_compute_interval_overlaps, TimeUpsampleTransposeFlat, JitterFrame

from convex_optim_base.optim_base import BatchParallelUnconstrainedProblem
from convex_optim_base.unconstrained_optim import batch_parallel_unconstrained_solve, FistaSolverParams
from dejitter_recons.dejitter_glm_components import precompute_feedback_exp_args, precompute_coupling_exp_args
from denoise_inverse_alg.glm_inverse_alg import PackedGLMTensors, FeedbackOnlyPackedGLMTensors
from denoise_inverse_alg.hqs_alg import BatchParallel_HQS_X_Problem, BatchParallel_MaskedUnblindDenoiserPrior_HQS_ZProb, \
    HQS_ParameterizedSolveFn, iter_rho_fixed_prior_hqs_solve
from denoise_inverse_alg.poisson_inverse_alg import PackedLNPTensors
from simple_priors.gaussian_prior import ConvPatch1FGaussianPrior


def construct_magic_rescale_const(time_length_to_use: Union[float, int]):
    return 400.0 / (750.0 - 500.0 + time_length_to_use)


MAGIC_LOSS_RESCALE_CONST2 = construct_magic_rescale_const(500.0)
MAGIC_LOSS_RESCALE_CONST_LNP = MAGIC_LOSS_RESCALE_CONST2 / 8.333


def compute_ground_truth_eye_movements(frames: np.ndarray,
                                       max_dist: int,
                                       device: torch.device,
                                       dtype: torch.dtype = torch.float32) \
        -> Tuple[np.ndarray, np.ndarray]:
    n_frames = frames.shape[0]

    per_side = (2 * max_dist + 1)
    kernel_needs_reshape = np.eye(per_side * per_side, dtype=np.float32)
    kernel = kernel_needs_reshape.reshape(per_side * per_side, per_side, per_side)

    shifts = -np.r_[-max_dist:max_dist + 1]
    mg_y, mg_x = np.meshgrid(shifts, shifts)
    mg_x_flat = mg_x.reshape(-1)
    mg_y_flat = mg_y.reshape(-1)

    computed_eye_position = []
    min_norms = []
    with torch.no_grad():
        first_frame_torch = torch.tensor(frames[0, ...], dtype=dtype, device=device)
        kernel_torch = torch.tensor(kernel, dtype=dtype, device=device)

        # shape (n_shifts, out_h, out_w)
        conv_shift = F.conv2d(first_frame_torch[None, None, :, :],
                              kernel_torch[:, None, :, :]).squeeze(0)

        # taking the outer difference is too big; instead just
        # loop over frames and solve one at a time
        for i in range(n_frames):
            # shape (out_h, out_w)
            relevant_frame_torch = torch.tensor(frames[i, max_dist:-max_dist, max_dist:-max_dist],
                                                dtype=dtype, device=device)
            # shape (n_shifts, )
            diff_norm = torch.linalg.norm(relevant_frame_torch[None, :, :] - conv_shift, dim=(1, 2))

            # should just be a number
            min_ix = torch.argmin(diff_norm).item()

            min_norms.append(diff_norm[min_ix].item())
            computed_eye_position.append([mg_x_flat[min_ix], mg_y_flat[min_ix]])

    return np.array(computed_eye_position, dtype=np.int64), np.array(min_norms, dtype=np.float32)


@torch.jit.script
def noreduce_batch_per_bin_bernoulli_neg_log_likelihood(
        generator_signal: torch.Tensor,
        observed_spikes: torch.Tensor,
        mask: torch.Tensor) -> torch.Tensor:
    '''

    :param generator_signal: shape (batch, n_cells, n_bins)
    :param observed_spikes: shape (batch, n_cells, n_bins)
    :return: shape (batch, n_cells, n_bins)
    '''
    prod = generator_signal * observed_spikes
    log_sum_exp_term = torch.log(torch.exp(generator_signal) + 1)

    per_cell_loss_per_bin = log_sum_exp_term - prod

    return mask * per_cell_loss_per_bin


@torch.jit.script
def noreduce_nomask_batch_bin_bernoulli_neg_LL(
        generator_signal: torch.Tensor,
        observed_spikes: torch.Tensor) -> torch.Tensor:
    prod = generator_signal * observed_spikes
    log_sum_exp_term = torch.log1p(torch.exp(generator_signal))
    return log_sum_exp_term - prod


@torch.jit.script
def noreduce_batch_per_bin_poisson_neg_log_likelihood(
        generator_signal: torch.Tensor,
        observed_spikes: torch.Tensor,
        mask: torch.Tensor) -> torch.Tensor:
    '''

    :param generator_signal: shape (batch, n_cells, n_bins)
    :param observed_spikes: shape (batch, n_cells, n_bins)
    :return: shape (batch, n_cells, n_bins)
    '''

    prod = generator_signal * observed_spikes
    unmasked = torch.exp(generator_signal) - prod
    return mask * unmasked


@torch.jit.script
def noreduce_nomask_batch_per_bin_poisson_neg_log_likelihood(
        generator_signal: torch.Tensor,
        observed_spikes: torch.Tensor) -> torch.Tensor:
    '''

    :param generator_signal: shape (batch, n_cells, n_bins)
    :param observed_spikes: shape (batch, n_cells, n_bins)
    :return: shape (batch, n_cells, n_bins)
    '''

    prod = generator_signal * observed_spikes
    unmasked = torch.exp(generator_signal) - prod
    return unmasked


class HasPrecomputedGenSig(metaclass=ABCMeta):

    @abstractmethod
    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        raise NotImplementedError


class HasPrecomputedHistoryFrame(metaclass=ABCMeta):

    @abstractmethod
    def precompute_history_frames(self, history_frames: torch.Tensor) -> None:
        raise NotImplementedError


class FBOnlyKnownEyeMovementImageLoss(nn.Module,
                                      HasPrecomputedGenSig,
                                      HasPrecomputedHistoryFrame):

    def __init__(self,
                 batch: int,
                 n_history_frames: int,
                 frame_transition_times: np.ndarray,
                 spike_bin_edges: np.ndarray,
                 model_params: FeedbackOnlyPackedGLMTensors,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32,
                 magic_rescale_constant: float = MAGIC_LOSS_RESCALE_CONST2):
        super().__init__()

        self.batch = batch
        self.n_bins_total = spike_bin_edges.shape[1] - 1
        self.n_frames = frame_transition_times.shape[1] - 1
        self.magic_rescale_constant = magic_rescale_constant

        stacked_spatial_filters = model_params.spatial_filters
        stacked_timecourse_filters = model_params.timecourse_filters
        stacked_feedback_filters = model_params.feedback_filters
        stacked_bias = model_params.bias

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_pixels = self.height * self.width
        self.n_history_frames = n_history_frames
        self.n_bins_filter = stacked_timecourse_filters.shape[1]

        self.spiking_loss_fn = spiking_loss_fn

        # precompute all of the quantities needed to upsample in time and do backprop
        forward_sel, forward_weights, backward_sel, backward_weights = batch_compute_interval_overlaps(
            frame_transition_times, spike_bin_edges)

        # store these quantities as buffers
        self.register_buffer('forward_sel', torch.tensor(forward_sel, dtype=torch.long))
        self.register_buffer('backward_sel', torch.tensor(backward_sel, dtype=torch.long))
        self.register_buffer('forward_weights', torch.tensor(forward_weights, dtype=dtype))
        self.register_buffer('backward_weights', torch.tensor(backward_weights, dtype=dtype))

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
                             torch.zeros((self.batch, self.n_cells, self.n_bins_total - self.n_bins_filter + 1),
                                         dtype=dtype))
        self.register_buffer('precomputed_history_frames',
                             torch.zeros((self.batch, n_history_frames, self.n_cells), dtype=dtype))

        self.upsampler = TimeUpsampleTransposeFlat.apply
        self.jitterer = JitterFrame.apply

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        with torch.no_grad():
            feedback_component = precompute_feedback_exp_args(
                self.stacked_feedback_filters,
                observed_spikes
            )

            precomputed = feedback_component + self.stacked_bias[None, :, :]
            self.precomputed_feedback_gensig.data[:] = precomputed.data[:]

    def precompute_history_frames(self, history_frames: torch.Tensor) -> None:
        '''

        :param history_frames: shape (batch, n_history_frames, height, width)
        :return:
        '''
        with torch.no_grad():
            # shape (batch, n_history_frames, n_pix)
            history_frames_flat = history_frames.reshape(self.batch, self.n_history_frames, self.n_pixels)

            # shape (1, 1, n_cells, n_pixels) @ (batch, n_history_frames, n_pix, 1)
            # -> (batch, n_history_frames, n_cells, 1) -> (batch, n_history_frames, n_cells)
            spat_filt_applied = (
                    self.stacked_flat_spat_filters[None, None, :, :] @ history_frames_flat[:, :, :, None]).squeeze(3)
            self.precomputed_history_frames.data[:] = spat_filt_applied.data[:]

    def forward(self,
                batched_image: torch.Tensor,
                batched_spikes: torch.Tensor,
                eye_movements: torch.Tensor,
                time_mask: torch.Tensor):
        '''

        :param batched_image: shape (batch, height, width)
        :param batched_spikes: shape (batch, n_cells, n_bins)
        :param eye_movements: shape (batch, n_jittered_frames, 2), each eye movement coordinate
            is associated with the corresponding frame transition time provided in the constructor
        :param time_mask: shape (batch, n_bins - n_bins_filter + 1)
        :param args:
        :param kwargs:
        :return:
        '''
        # steps to get the stimulus contribution to the generator signal
        # (1) apply eye movements to the estimated frame
        # (2) concatenate the history and the jittered estimated frame
        # (3) apply the spatial filters for each cell to reduce dimensionality
        # (4) time-upsample the filtered images

        # (1) apply eye movements
        # shape (batch, n_jittered_frames, height, width)
        jittered_frames = self.jitterer(batched_image, eye_movements)

        # (2) apply the spatial filters
        # shape (batch, n_frames, n_pix)
        jittered_frames_flat = jittered_frames.reshape(self.batch, -1, self.n_pixels)

        # shape (1, 1, n_cells, n_pixels) @ (batch, n_frames, n_pix, 1)
        # -> (batch, n_frames, n_cells, 1) -> (batch, n_frames, n_cells)
        jittered_frames_spat_filt_applied = (
                self.stacked_flat_spat_filters[None, None, :, :] @ jittered_frames_flat[:, :, :, None]).squeeze(3)

        # shape (batch, n_frames_total, n_cells)
        spat_filt_applied = torch.concat([self.precomputed_history_frames, jittered_frames_spat_filt_applied],
                                         dim=1)

        # (4) upsample the whole thing
        # shape (batch, n_cells, n_bins)
        upsampled_movie = self.upsampler(spat_filt_applied, self.forward_sel, self.forward_weights,
                                         self.backward_sel, self.backward_weights)

        time_filt_applied = F.conv1d(upsampled_movie,
                                     self.stacked_timecourse_filters[:, None, :],
                                     groups=self.n_cells)

        gen_sig = time_filt_applied + self.precomputed_feedback_gensig

        # shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        spiking_loss = self.spiking_loss_fn(gen_sig[:, :, :-1],
                                            batched_spikes[:, :, self.n_bins_filter:])

        spiking_loss_masked = spiking_loss * time_mask[:, None, :]

        # we need to fix the loss scaling here: the reasoning behind this is as follows:
        # since according to Bayes we should gain more certainty when we observe more
        # data, we should not divide out by the total number bins.
        # Instead, to get the scaling that we want, we will sum over the bins
        # and then divide by a constant that achieves the correct scaling when all of the
        # bins are observed
        spiking_loss_masked_rescaled = spiking_loss_masked * self.magic_rescale_constant

        # shape (batch, )
        return torch.sum(spiking_loss_masked_rescaled, dim=(1, 2))


class KnownEyeMovementImageLoss(nn.Module,
                                HasPrecomputedGenSig,
                                HasPrecomputedHistoryFrame):

    def __init__(self,
                 batch: int,
                 n_history_frames: int,
                 frame_transition_times: np.ndarray,
                 spike_bin_edges: np.ndarray,
                 glm_model_params: PackedGLMTensors,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32,
                 magic_rescale_constant: float = MAGIC_LOSS_RESCALE_CONST2):
        super().__init__()

        self.batch = batch
        self.n_bins_total = spike_bin_edges.shape[1] - 1
        self.n_frames = frame_transition_times.shape[1] - 1
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
            frame_transition_times, spike_bin_edges)

        # store these quantities as buffers
        self.register_buffer('forward_sel', torch.tensor(forward_sel, dtype=torch.long))
        self.register_buffer('backward_sel', torch.tensor(backward_sel, dtype=torch.long))
        self.register_buffer('forward_weights', torch.tensor(forward_weights, dtype=dtype))
        self.register_buffer('backward_weights', torch.tensor(backward_weights, dtype=dtype))

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
                             torch.zeros((self.batch, self.n_cells, self.n_bins_total - self.n_bins_filter + 1),
                                         dtype=dtype))
        self.register_buffer('precomputed_history_frames',
                             torch.zeros((self.batch, n_history_frames, self.n_cells), dtype=dtype))

        self.upsampler = TimeUpsampleTransposeFlat.apply
        self.jitterer = JitterFrame.apply

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        with torch.no_grad():
            coupling_component = precompute_coupling_exp_args(
                self.stacked_coupling_filters,
                self.coupled_sel,
                observed_spikes
            )

            feedback_component = precompute_feedback_exp_args(
                self.stacked_feedback_filters,
                observed_spikes
            )

            precomputed = coupling_component + feedback_component + self.stacked_bias[None, :, :]
            self.precomputed_feedback_coupling_gensig.data[:] = precomputed.data[:]

    def precompute_history_frames(self, history_frames: torch.Tensor) -> None:
        '''

        :param history_frames: shape (batch, n_history_frames, height, width)
        :return:
        '''
        with torch.no_grad():
            # shape (batch, n_history_frames, n_pix)
            history_frames_flat = history_frames.reshape(self.batch, self.n_history_frames, self.n_pixels)

            # shape (1, 1, n_cells, n_pixels) @ (batch, n_history_frames, n_pix, 1)
            # -> (batch, n_history_frames, n_cells, 1) -> (batch, n_history_frames, n_cells)
            spat_filt_applied = (
                    self.stacked_flat_spat_filters[None, None, :, :] @ history_frames_flat[:, :, :, None]).squeeze(3)
            self.precomputed_history_frames.data[:] = spat_filt_applied.data[:]

    def forward(self,
                batched_image: torch.Tensor,
                batched_spikes: torch.Tensor,
                eye_movements: torch.Tensor,
                time_mask: torch.Tensor):
        '''

        :param batched_image: shape (batch, height, width)
        :param batched_spikes: shape (batch, n_cells, n_bins)
        :param eye_movements: shape (batch, n_jittered_frames, 2), each eye movement coordinate
            is associated with the corresponding frame transition time provided in the constructor
        :param time_mask: shape (batch, n_bins - n_bins_filter + 1)
        :param args:
        :param kwargs:
        :return:
        '''
        # steps to get the stimulus contribution to the generator signal
        # (1) apply eye movements to the estimated frame
        # (2) concatenate the history and the jittered estimated frame
        # (3) apply the spatial filters for each cell to reduce dimensionality
        # (4) time-upsample the filtered images

        # (1) apply eye movements
        # shape (batch, n_jittered_frames, height, width)
        jittered_frames = self.jitterer(batched_image, eye_movements)

        # (2) apply the spatial filters
        # shape (batch, n_frames, n_pix)
        jittered_frames_flat = jittered_frames.reshape(self.batch, -1, self.n_pixels)

        # shape (1, 1, n_cells, n_pixels) @ (batch, n_frames, n_pix, 1)
        # -> (batch, n_frames, n_cells, 1) -> (batch, n_frames, n_cells)
        jittered_frames_spat_filt_applied = (
                self.stacked_flat_spat_filters[None, None, :, :] @ jittered_frames_flat[:, :, :, None]).squeeze(3)

        # shape (batch, n_frames_total, n_cells)
        spat_filt_applied = torch.concat([self.precomputed_history_frames, jittered_frames_spat_filt_applied],
                                         dim=1)

        # (4) upsample the whole thing
        # shape (batch, n_cells, n_bins)
        upsampled_movie = self.upsampler(spat_filt_applied, self.forward_sel, self.forward_weights,
                                         self.backward_sel, self.backward_weights)

        time_filt_applied = F.conv1d(upsampled_movie,
                                     self.stacked_timecourse_filters[:, None, :],
                                     groups=self.n_cells)

        gen_sig = time_filt_applied + self.precomputed_feedback_coupling_gensig

        # shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        spiking_loss = self.spiking_loss_fn(gen_sig[:, :, :-1],
                                            batched_spikes[:, :, self.n_bins_filter:])

        spiking_loss_masked = spiking_loss * time_mask[:, None, :]

        # we need to fix the loss scaling here: the reasoning behind this is as follows:
        # since according to Bayes we should gain more certainty when we observe more
        # data, we should not divide out by the total number bins.
        # Instead, to get the scaling that we want, we will sum over the bins
        # and then divide by a constant that achieves the correct scaling when all of the
        # bins are observed
        spiking_loss_masked_rescaled = spiking_loss_masked * self.magic_rescale_constant

        # shape (batch, )
        return torch.sum(spiking_loss_masked_rescaled, dim=(1, 2))


class KnownEyeMovementProxProblem2(BatchParallelUnconstrainedProblem,
                                   BatchParallel_HQS_X_Problem,
                                   HasPrecomputedGenSig,
                                   HasPrecomputedHistoryFrame):
    '''
    More efficient version of KnownEyeMovementProxProblem; pre-applies
        the spatial filters to the images prior to performing the
        time upsample operation, so the upsampling operation is much
        smaller (~700 cells rather than 160x256=40960 pixels)

    Reconstructs one image, assuming that eye movements are
        fixed and known.

    Needs the previous frames and spikes for history for the GLM.

    For simplicity, batch size is fixed to be 1, since we only
        reconstruct a single static image, and obviously no
        beam searching is necessary since the eye movements are
        fixed and known.

    Uses old-style optimization framework, since mixed precision
        is probably not going to be very useful and we don't yet
        have a new-style batch solver, which will be necessary
        for an eventual beam-search implementation
    '''

    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'
    EYE_MOVEMENTS_KWARGS = 'eye_movements'
    TIME_MASK_KWARGS = 'time_mask'

    def __init__(self,
                 batch: int,
                 n_history_frames: int,
                 frame_transition_times: np.ndarray,
                 spike_bin_edges: np.ndarray,
                 model_params: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors],
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 rho: float,
                 dtype: torch.dtype = torch.float32,
                 magic_rescale_constant: float = MAGIC_LOSS_RESCALE_CONST2):
        super().__init__()

        self.batch = batch
        self.n_bins_total = spike_bin_edges.shape[1] - 1
        self.n_frames = frame_transition_times.shape[1] - 1
        self.n_history_frames = n_history_frames

        self.rho = rho

        stacked_spatial_filters = model_params.spatial_filters
        stacked_timecourse_filters = model_params.timecourse_filters

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_pixels = self.height * self.width
        self.n_bins_filter = stacked_timecourse_filters.shape[1]

        # Create a buffer for the z const for HQS
        self.register_buffer('z_const_tensor', torch.empty((batch, self.height, self.width), dtype=dtype))

        if isinstance(model_params, PackedGLMTensors):
            self.loss_calc = KnownEyeMovementImageLoss(
                self.batch, self.n_history_frames, frame_transition_times, spike_bin_edges,
                model_params, spiking_loss_fn, dtype=dtype, magic_rescale_constant=magic_rescale_constant
            )
        else:
            self.loss_calc = FBOnlyKnownEyeMovementImageLoss(
                self.batch, self.n_history_frames, frame_transition_times, spike_bin_edges,
                model_params, spiking_loss_fn, dtype=dtype, magic_rescale_constant=magic_rescale_constant
            )

        self.image = nn.Parameter(torch.empty((self.batch, self.height, self.width), dtype=dtype),
                                  requires_grad=True)
        nn.init.uniform_(self.image, a=-1e-2, b=1e-2)

    @property
    def n_problems(self):
        return self.batch

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        self.loss_calc.precompute_gensig_components(observed_spikes)

    def precompute_history_frames(self, history_frames: torch.Tensor) -> None:
        self.loss_calc.precompute_history_frames(history_frames)

    def _eval_smooth_loss(self, *args, **kwargs):
        # shape (batch, height, width)
        batched_image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape (batch, n_cells, n_bins)
        batched_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (batch, n_bins - n_bins_filter + 1)
        time_mask = kwargs[self.TIME_MASK_KWARGS]

        # shape (batch, n_jittered_frames, 2)
        eye_movements = kwargs[self.EYE_MOVEMENTS_KWARGS]

        encoding_loss = self.loss_calc(batched_image_imshape,
                                       batched_spikes,
                                       eye_movements,
                                       time_mask)

        # shape (batch, )
        prox_diff = batched_image_imshape - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

        return encoding_loss + prox_loss

    def forward(self, **kwargs):
        # shape (batch, n_cells, n_bins)
        batched_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (batch, n_bins - n_bins_filter + 1)
        time_mask = kwargs[self.TIME_MASK_KWARGS]

        # shape (batch, n_jittered_frames, 2)
        eye_movements = kwargs[self.EYE_MOVEMENTS_KWARGS]

        encoding_loss = self.loss_calc(self.image, batched_spikes,
                                       eye_movements, time_mask)

        prox_diff = self.image - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

        return encoding_loss + prox_loss

    def assign_proxto(self, prox_to: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = prox_to.data[:]

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def get_output_image(self) -> torch.Tensor:
        return self.image.detach().clone()

    def get_reconstructed_image(self) -> np.ndarray:
        return self.image.detach().cpu().numpy()


def construct_likelihood_masks(n_history_frames: int,
                               frame_transition_times: np.ndarray,
                               spike_bin_times: np.ndarray,
                               delta_lag_samples: int,
                               spike_bin_width: int) -> List[np.ndarray]:
    '''

    :param n_history_frames: int, number of history frames given to the
        likelihood model. Must be the same for everything in the same batch
    :param frame_transition_times: shape (batch, n_frames_total + 1)
    :param spike_bin_times: shape (batch, n_bins + 1)
    :param delta_lag: int, number of samples (not bins, not frames) of lag
        after the frame transition that the mask should transition from valid
        to not valid
    :return:
    '''

    batch = frame_transition_times.shape[0]

    # shape (batch, n_frames_total - n_history_frames)
    target_frame_transition_times = frame_transition_times[:, n_history_frames:]
    n_eye_movements_to_estim = target_frame_transition_times.shape[1] - 1

    # shape (batch, n_frames_total)
    delta_lag = np.round(target_frame_transition_times + delta_lag_samples).astype(np.int64)

    # shape (batch, n_frames_total)
    is_true = (delta_lag - spike_bin_times[:, -2:-1]) > 0
    max_idx = np.argmax(is_true, axis=1)
    min_max_lag_ix = np.min(np.argmax(is_true, axis=1).squeeze())

    # shape (batch, min_max_lag_ix)
    time_estim_points = np.round((delta_lag[:, :min_max_lag_ix] - spike_bin_times[:, 0:1]) / spike_bin_width).astype(
        np.int64)

    mask_list = []
    for i in range(min_max_lag_ix):
        mask = np.zeros((batch, spike_bin_times.shape[1] - 1), dtype=np.float32)
        time_estim_point = time_estim_points[:, i]
        for b in range(batch):
            mask[b, :time_estim_point[b]] = 1.0

        mask_list.append(mask)

    return mask_list


def estimate_image_with_fixed_eye_movements(
        packed_model_tensors: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors],
        unblind_denoiser_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        history_frames: np.ndarray,
        frame_transition_times: np.ndarray,
        observed_spikes: np.ndarray,
        spike_bin_times: np.ndarray,
        image_valid_mask: np.ndarray,
        eye_movement_trajectory: np.ndarray,
        time_valid_mask: np.ndarray,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        rho_iterator: Iterator[float],
        lambda_prior_weight: float,
        optim_x_iter: Iterator[HQS_ParameterizedSolveFn],
        optim_z_iter: Iterator[HQS_ParameterizedSolveFn],
        device: torch.device,
        return_intermediate_images: bool = False,
        solver_verbose: bool = False,
        magic_rescale_const: float = MAGIC_LOSS_RESCALE_CONST2):
    '''
    Batched for improved performance, since in the no-eye-movement case we don't
        have to do any kind of searching

    :param packed_glm_tensors:
    :param unblind_denoiser_callable:
    :param history_frames: shape (batch, n_history_frames, hieght, width)
    :param frame_transition_times: shape (batch, n_frames_total + 1)
    :param observed_spikes: shape (batch, n_cells, n_bins)
    :param spike_bin_times: shape (batch, n_bins + 1)
    :param image_valid_mask: shape (height, width)
    :param eye_movement_trajectory: shape (batch, n_frames_target, 2)
    :param time_valid_mask:
    :param loss_fn:
    :param rho_iterator:
    :param lambda_prior_weight:
    :param optim_x_iter:
    :param optim_z_iter:
    :param device:
    :param image_init_guess:
    :param return_intermediate_images:
    :return:
    '''

    batch, n_history_frames, height, width = history_frames.shape

    hqs_x_prob = KnownEyeMovementProxProblem2(
        batch,
        n_history_frames,
        frame_transition_times,
        spike_bin_times,
        packed_model_tensors,
        loss_fn,
        0.1,  # temporary placeholder value, will never be used
        dtype=torch.float32,
        magic_rescale_constant=magic_rescale_const
    ).to(device)

    hqs_z_prob = BatchParallel_MaskedUnblindDenoiserPrior_HQS_ZProb(
        batch,
        unblind_denoiser_callable,
        (height, width),
        image_valid_mask,
        0.1,  # temporary placeholder value, will never be used
        prior_lambda=lambda_prior_weight
    ).to(device)

    spikes_torch = torch.tensor(observed_spikes, dtype=torch.float32, device=device)
    history_frames_torch = torch.tensor(history_frames, dtype=torch.float32, device=device)
    jitter_coords_torch = torch.tensor(eye_movement_trajectory, dtype=torch.long, device=device)

    n_bins_filter = hqs_x_prob.n_bins_filter
    relevant_time_mask_torch = torch.tensor(time_valid_mask[:, n_bins_filter:],
                                            dtype=torch.float32, device=device)

    hqs_x_prob.precompute_gensig_components(spikes_torch)

    hqs_x_prob.precompute_history_frames(history_frames_torch)

    hqs_x_prob.assign_z(0.1 * torch.randn((history_frames_torch.shape[0], height, width), device=device,
                                          dtype=torch.float32))
    hqs_z_prob.reinitialize_variables()

    intermediates = iter_rho_fixed_prior_hqs_solve(
        hqs_x_prob,
        iter(optim_x_iter),
        hqs_z_prob,
        iter(optim_z_iter),
        iter(rho_iterator),
        lambda_prior_weight,
        verbose=solver_verbose,
        save_intermediates=True,
        observed_spikes=spikes_torch,
        eye_movements=jitter_coords_torch,
        time_mask=relevant_time_mask_torch
    )

    reconstructed_images_proper = hqs_z_prob.get_reconstructed_image()

    del hqs_x_prob, hqs_z_prob, spikes_torch, history_frames_torch, jitter_coords_torch
    del relevant_time_mask_torch

    if return_intermediate_images:
        return reconstructed_images_proper, intermediates
    else:
        return reconstructed_images_proper


class KnownEyeMovement1FReconstruction(BatchParallelUnconstrainedProblem,
                                       HasPrecomputedGenSig,
                                       HasPrecomputedHistoryFrame):
    '''
    More efficient version of KnownEyeMovementProxProblem; pre-applies
        the spatial filters to the images prior to performing the
        time upsample operation, so the upsampling operation is much
        smaller (~700 cells rather than 160x256=40960 pixels)

    Reconstructs one image, assuming that eye movements are
        fixed and known.

    Needs the previous frames and spikes for history for the GLM.

    For simplicity, batch size is fixed to be 1, since we only
        reconstruct a single static image, and obviously no
        beam searching is necessary since the eye movements are
        fixed and known.

    Uses old-style optimization framework, since mixed precision
        is probably not going to be very useful and we don't yet
        have a new-style batch solver, which will be necessary
        for an eventual beam-search implementation
    '''

    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'
    EYE_MOVEMENTS_KWARGS = 'eye_movements'
    TIME_MASK_KWARGS = 'time_mask'

    def __init__(self,
                 batch: int,
                 patch_zca_matrix: np.ndarray,
                 gaussian_prior_lambda: float,
                 n_history_frames: int,
                 frame_transition_times: np.ndarray,
                 spike_bin_edges: np.ndarray,
                 model_params: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors],
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32,
                 magic_rescale_constant: float = MAGIC_LOSS_RESCALE_CONST2,
                 patch_stride: int = 1):
        super().__init__()

        self.batch = batch

        self.gaussian_prior_lambda = gaussian_prior_lambda

        self.n_bins_total = spike_bin_edges.shape[1] - 1
        self.n_frames = frame_transition_times.shape[1] - 1
        self.n_history_frames = n_history_frames

        stacked_spatial_filters = model_params.spatial_filters
        stacked_timecourse_filters = model_params.timecourse_filters

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_pixels = self.height * self.width
        self.n_bins_filter = stacked_timecourse_filters.shape[1]

        if isinstance(model_params, PackedGLMTensors):
            self.loss_calc = KnownEyeMovementImageLoss(
                self.batch, self.n_history_frames, frame_transition_times, spike_bin_edges,
                model_params, spiking_loss_fn, dtype=dtype, magic_rescale_constant=magic_rescale_constant
            )
        else:
            self.loss_calc = FBOnlyKnownEyeMovementImageLoss(
                self.batch, self.n_history_frames, frame_transition_times, spike_bin_edges,
                model_params, spiking_loss_fn, dtype=dtype, magic_rescale_constant=magic_rescale_constant
            )

        self.batch_prior_callable = ConvPatch1FGaussianPrior(patch_zca_matrix,
                                                             patch_stride=patch_stride,
                                                             dtype=dtype)

        self.image = nn.Parameter(torch.empty((self.batch, self.height, self.width), dtype=dtype),
                                  requires_grad=True)
        nn.init.uniform_(self.image, a=-1e-2, b=1e-2)

    @property
    def n_problems(self):
        return self.batch

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        self.loss_calc.precompute_gensig_components(observed_spikes)

    def precompute_history_frames(self, history_frames: torch.Tensor) -> None:
        self.loss_calc.precompute_history_frames(history_frames)

    def _eval_smooth_loss(self, *args, **kwargs):
        # shape (batch, height, width)
        batched_image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape (batch, n_cells, n_bins)
        batched_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (batch, n_bins - n_bins_filter + 1)
        time_mask = kwargs[self.TIME_MASK_KWARGS]

        # shape (batch, n_jittered_frames, 2)
        eye_movements = kwargs[self.EYE_MOVEMENTS_KWARGS]

        encoding_loss = self.loss_calc(batched_image_imshape,
                                       batched_spikes,
                                       eye_movements,
                                       time_mask)

        # shape (batch, )
        gaussian_prior_penalty = 0.5 * self.gaussian_prior_lambda * self.batch_prior_callable(batched_image_imshape)

        return encoding_loss + gaussian_prior_penalty

    def forward(self, **kwargs):
        # shape (batch, n_cells, n_bins)
        batched_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (batch, n_bins - n_bins_filter + 1)
        time_mask = kwargs[self.TIME_MASK_KWARGS]

        # shape (batch, n_jittered_frames, 2)
        eye_movements = kwargs[self.EYE_MOVEMENTS_KWARGS]

        encoding_loss = self.loss_calc(self.image, batched_spikes,
                                       eye_movements, time_mask)

        # shape (batch, )
        gaussian_prior_penalty = 0.5 * self.gaussian_prior_lambda * self.batch_prior_callable(self.image)

        return encoding_loss + gaussian_prior_penalty

    def get_output_image(self) -> torch.Tensor:
        return self.image.detach().clone()

    def get_reconstructed_image(self) -> np.ndarray:
        return self.image.detach().cpu().numpy()


def estimate_1F_image_with_fixed_eye_movements(
        packed_model_tensors: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors],
        patch_zca_matrix: np.ndarray,
        history_frames: np.ndarray,
        frame_transition_times: np.ndarray,
        observed_spikes: np.ndarray,
        spike_bin_times: np.ndarray,
        eye_movement_trajectory: np.ndarray,
        time_valid_mask: np.ndarray,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        gaussian_prior_weight: float,
        device: torch.device,
        fista_solver_params: FistaSolverParams = FistaSolverParams(
            initial_learning_rate=1.0,
            max_iter=250,
            converge_epsilon=1e-6,
            backtracking_beta=0.5),
        solver_verbose: bool = False,
        magic_rescale_const: float = MAGIC_LOSS_RESCALE_CONST2,
        patch_stride: int = 1):
    '''

    :param packed_glm_tensors:
    :param history_frames: shape (batch, n_history_frames, hieght, width)
    :param frame_transition_times: shape (batch, n_frames_total + 1)
    :param observed_spikes: shape (batch, n_cells, n_bins)
    :param spike_bin_times: shape (batch, n_bins + 1)
    :param eye_movement_trajectory: shape (batch, n_frames_target, 2)
    :param time_valid_mask:
    :param loss_fn:
    :param gaussian_prior_weight:
    :param device:
    :param solver_verbose:
    :param magic_rescale_const:
    :return:
    '''

    batch, n_history_frames, height, width = history_frames.shape

    batch_gaussian_problem = KnownEyeMovement1FReconstruction(
        batch,
        patch_zca_matrix,
        gaussian_prior_weight,
        n_history_frames,
        frame_transition_times,
        spike_bin_times,
        packed_model_tensors,
        loss_fn,
        dtype=torch.float32,
        magic_rescale_constant=magic_rescale_const,
        patch_stride=patch_stride,
    ).to(device)

    spikes_torch = torch.tensor(observed_spikes, dtype=torch.float32, device=device)
    history_frames_torch = torch.tensor(history_frames, dtype=torch.float32, device=device)
    jitter_coords_torch = torch.tensor(eye_movement_trajectory, dtype=torch.long, device=device)

    n_bins_filter = batch_gaussian_problem.n_bins_filter
    relevant_time_mask_torch = torch.tensor(time_valid_mask[:, n_bins_filter:],
                                            dtype=torch.float32, device=device)

    batch_gaussian_problem.precompute_gensig_components(spikes_torch)
    batch_gaussian_problem.precompute_history_frames(history_frames_torch)

    _ = batch_parallel_unconstrained_solve(
        batch_gaussian_problem,
        fista_solver_params,
        verbose=solver_verbose,
        observed_spikes=spikes_torch,
        eye_movements=jitter_coords_torch,
        time_mask=relevant_time_mask_torch
    )

    reconstructed_images_proper = batch_gaussian_problem.get_reconstructed_image()

    del batch_gaussian_problem, spikes_torch, history_frames_torch, jitter_coords_torch
    del relevant_time_mask_torch

    return reconstructed_images_proper


class FrameRateLNPKnownEyeMovementLoss(nn.Module):

    def __init__(self,
                 batch: int,
                 n_history_frames: int,
                 n_frames_total: int,
                 lnp_model_params: PackedLNPTensors,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32,
                 magic_rescale_constant: float = MAGIC_LOSS_RESCALE_CONST2):
        super().__init__()

        self.batch = batch
        self.n_frames = n_frames_total
        self.magic_rescale_constant = magic_rescale_constant

        stacked_spatial_filters = lnp_model_params.spatial_filters
        stacked_timecourse_filters = lnp_model_params.timecourse_filters
        stacked_bias = lnp_model_params.bias

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_pixels = self.height * self.width
        self.n_history_frames = n_history_frames
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.spiking_loss_fn = spiking_loss_fn

        # store GLM quantities as buffers as well
        # shape (n_cells, n_pixels)
        stacked_flat_spat_filters = stacked_spatial_filters.reshape(self.n_cells, -1)
        self.register_buffer('stacked_flat_spat_filters', torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_timecourse_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_timecourse_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_timecourse_filters', torch.tensor(stacked_timecourse_filters, dtype=dtype))

        # shape (n_cells, 1)
        assert stacked_bias.shape == (self.n_cells, 1), f'stacked_bias must have shape {(self.n_cells, 1)}'
        self.register_buffer('stacked_bias', torch.tensor(stacked_bias, dtype=dtype))

        self.register_buffer('precomputed_history_frames',
                             torch.zeros((self.batch, n_history_frames, self.n_cells), dtype=dtype))

        self.jitterer = JitterFrame.apply

    def precompute_history_frames(self, history_frames: torch.Tensor) -> None:
        '''

        :param history_frames: shape (batch, n_history_frames, height, width)
        :return:
        '''
        with torch.no_grad():
            # shape (batch, n_history_frames, n_pix)
            history_frames_flat = history_frames.reshape(self.batch, self.n_history_frames, self.n_pixels)

            # shape (1, 1, n_cells, n_pixels) @ (batch, n_history_frames, n_pix, 1)
            # -> (batch, n_history_frames, n_cells, 1) -> (batch, n_history_frames, n_cells)
            spat_filt_applied = (
                    self.stacked_flat_spat_filters[None, None, :, :] @ history_frames_flat[:, :, :, None]).squeeze(3)
            self.precomputed_history_frames.data[:] = spat_filt_applied.data[:]

    def forward(self,
                batched_image: torch.Tensor,
                batched_spikes: torch.Tensor,
                eye_movements: torch.Tensor,
                time_mask: torch.Tensor):
        '''

        :param batched_image: shape (batch, height, width)
        :param batched_spikes: shape (batch, n_cells, n_bins)
        :param eye_movements: shape (batch, n_jittered_frames, 2), each eye movement coordinate
            is associated with the corresponding frame transition time provided in the constructor
        :param time_mask: shape (batch, n_bins - n_bins_filter + 1)
        :param args:
        :param kwargs:
        :return:
        '''
        # steps to get the stimulus contribution to the generator signal
        # (1) apply eye movements to the estimated frame
        # (2) concatenate the history and the jittered estimated frame
        # (3) apply the spatial filters for each cell to reduce dimensionality
        # (4) time-upsample the filtered images

        # (1) apply eye movements
        # shape (batch, n_jittered_frames, height, width)
        jittered_frames = self.jitterer(batched_image, eye_movements)

        # (2) apply the spatial filters
        # shape (batch, n_frames, n_pix)
        jittered_frames_flat = jittered_frames.reshape(self.batch, -1, self.n_pixels)

        # shape (1, 1, n_cells, n_pixels) @ (batch, n_frames, n_pix, 1)
        # -> (batch, n_frames, n_cells, 1) -> (batch, n_frames, n_cells)
        jittered_frames_spat_filt_applied = (
                self.stacked_flat_spat_filters[None, None, :, :] @ jittered_frames_flat[:, :, :, None]).squeeze(3)

        # shape (batch, n_frames_total, n_cells) -> (batch, n_cells, n_frames_total)
        spat_filt_applied = torch.concat([self.precomputed_history_frames, jittered_frames_spat_filt_applied],
                                         dim=1).permute((0, 2, 1))

        time_filt_applied = F.conv1d(spat_filt_applied,
                                     self.stacked_timecourse_filters[:, None, :],
                                     groups=self.n_cells)

        gen_sig = time_filt_applied + self.stacked_bias[None, :, :]

        # shape (batch, n_cells, n_frames - n_bins_filter + 1)
        spiking_loss = self.spiking_loss_fn(gen_sig,
                                            batched_spikes[:, :, self.n_bins_filter - 1:])

        spiking_loss_masked = spiking_loss * time_mask[:, None, :]

        # we need to fix the loss scaling here: the reasoning behind this is as follows:
        # since according to Bayes we should gain more certainty when we observe more
        # data, we should not divide out by the total number bins.
        # Instead, to get the scaling that we want, we will sum over the bins
        # and then divide by a constant that achieves the correct scaling when all of the
        # bins are observed
        spiking_loss_masked_rescaled = spiking_loss_masked * self.magic_rescale_constant

        # shape (batch, )
        return torch.sum(spiking_loss_masked_rescaled, dim=(1, 2))


class FrameRateLNPProxProblem(BatchParallelUnconstrainedProblem,
                              BatchParallel_HQS_X_Problem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'
    EYE_MOVEMENTS_KWARGS = 'eye_movements'
    TIME_MASK_KWARGS = 'time_mask'

    def __init__(self,
                 batch: int,
                 n_history_frames: int,
                 n_frames_total: int,
                 lnp_model_params: PackedLNPTensors,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 rho: float,
                 dtype: torch.dtype = torch.float32,
                 magic_rescale_constant: float = MAGIC_LOSS_RESCALE_CONST2):
        super().__init__()

        self.batch = batch
        self.n_frames = n_frames_total
        self.n_history_frames = n_history_frames

        self.rho = rho

        stacked_spatial_filters = lnp_model_params.spatial_filters
        stacked_timecourse_filters = lnp_model_params.timecourse_filters

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_pixels = self.height * self.width
        self.n_bins_filter = stacked_timecourse_filters.shape[1]

        # Create a buffer for the z const for HQS
        self.register_buffer('z_const_tensor', torch.empty((batch, self.height, self.width), dtype=dtype))

        self.loss_calc = FrameRateLNPKnownEyeMovementLoss(
            self.batch, self.n_history_frames, self.n_frames,
            lnp_model_params, spiking_loss_fn, dtype=dtype, magic_rescale_constant=magic_rescale_constant
        )

        self.image = nn.Parameter(torch.empty((self.batch, self.height, self.width), dtype=dtype),
                                  requires_grad=True)
        nn.init.uniform_(self.image, a=-1e-2, b=1e-2)

    @property
    def n_problems(self):
        return self.batch

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def precompute_history_frames(self, history_frames: torch.Tensor) -> None:
        self.loss_calc.precompute_history_frames(history_frames)

    def _eval_smooth_loss(self, *args, **kwargs):
        # shape (batch, height, width)
        batched_image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape (batch, n_cells, n_bins)
        batched_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (batch, n_bins - n_bins_filter + 1)
        time_mask = kwargs[self.TIME_MASK_KWARGS]

        # shape (batch, n_jittered_frames, 2)
        eye_movements = kwargs[self.EYE_MOVEMENTS_KWARGS]

        encoding_loss = self.loss_calc(batched_image_imshape,
                                       batched_spikes,
                                       eye_movements,
                                       time_mask)

        # shape (batch, )
        prox_diff = batched_image_imshape - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

        return encoding_loss + prox_loss

    def forward(self, **kwargs):
        # shape (batch, n_cells, n_bins)
        batched_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (batch, n_bins - n_bins_filter + 1)
        time_mask = kwargs[self.TIME_MASK_KWARGS]

        # shape (batch, n_jittered_frames, 2)
        eye_movements = kwargs[self.EYE_MOVEMENTS_KWARGS]

        encoding_loss = self.loss_calc(self.image, batched_spikes,
                                       eye_movements, time_mask)

        prox_diff = self.image - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

        return encoding_loss + prox_loss

    def assign_proxto(self, prox_to: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = prox_to.data[:]

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def get_output_image(self) -> torch.Tensor:
        return self.image.detach().clone()

    def get_reconstructed_image(self) -> np.ndarray:
        return self.image.detach().cpu().numpy()


def estimate_frame_rate_lnp_with_fixed_eye_movements(
        packed_lnp_tensors: PackedLNPTensors,
        unblind_denoiser_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        history_frames: np.ndarray,
        n_frames_total: int,
        observed_spikes: np.ndarray,
        image_valid_mask: np.ndarray,
        eye_movement_trajectory: np.ndarray,
        time_valid_mask: np.ndarray,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        rho_iterator: Iterator[float],
        lambda_prior_weight: float,
        optim_x_iter: Iterator[HQS_ParameterizedSolveFn],
        optim_z_iter: Iterator[HQS_ParameterizedSolveFn],
        device: torch.device,
        return_intermediate_images: bool = False,
        solver_verbose: bool = False,
        magic_rescale_const: float = MAGIC_LOSS_RESCALE_CONST_LNP):
    '''

    :param packed_lnp_tensors:
    :param unblind_denoiser_callable:
    :param history_frames:
    :param n_frames_total: int, total number of frames used in the reconstruction procedure,
        including all of the history frames
    :param observed_spikes:
    :param image_valid_mask:
    :param eye_movement_trajectory:
    :param time_valid_mask:
    :param loss_fn:
    :param rho_iterator:
    :param lambda_prior_weight:
    :param optim_x_iter:
    :param optim_z_iter:
    :param device:
    :param return_intermediate_images:
    :param solver_verbose:
    :param magic_rescale_const:
    :return:
    '''

    batch, n_history_frames, height, width = history_frames.shape

    hqs_x_prob = FrameRateLNPProxProblem(
        batch,
        n_history_frames,
        n_frames_total,
        packed_lnp_tensors,
        loss_fn,
        0.1,  # temporary placeholder value, will never be used
        dtype=torch.float32,
        magic_rescale_constant=magic_rescale_const
    ).to(device)

    hqs_z_prob = BatchParallel_MaskedUnblindDenoiserPrior_HQS_ZProb(
        batch,
        unblind_denoiser_callable,
        (height, width),
        image_valid_mask,
        0.1,  # temporary placeholder value, will never be used
        prior_lambda=lambda_prior_weight
    ).to(device)

    spikes_torch = torch.tensor(observed_spikes, dtype=torch.float32, device=device)
    history_frames_torch = torch.tensor(history_frames, dtype=torch.float32, device=device)
    jitter_coords_torch = torch.tensor(eye_movement_trajectory, dtype=torch.long, device=device)

    n_bins_filter = hqs_x_prob.n_bins_filter
    relevant_time_mask_torch = torch.tensor(time_valid_mask[:, n_bins_filter - 1:],
                                            dtype=torch.float32, device=device)

    hqs_x_prob.precompute_history_frames(history_frames_torch)

    hqs_x_prob.assign_z(0.1 * torch.randn((history_frames_torch.shape[0], height, width), device=device,
                                          dtype=torch.float32))
    hqs_z_prob.reinitialize_variables()

    intermediates = iter_rho_fixed_prior_hqs_solve(
        hqs_x_prob,
        iter(optim_x_iter),
        hqs_z_prob,
        iter(optim_z_iter),
        iter(rho_iterator),
        lambda_prior_weight,
        verbose=solver_verbose,
        save_intermediates=True,
        observed_spikes=spikes_torch,
        eye_movements=jitter_coords_torch,
        time_mask=relevant_time_mask_torch
    )

    reconstructed_images_proper = hqs_z_prob.get_reconstructed_image()

    del hqs_x_prob, hqs_z_prob, spikes_torch, history_frames_torch, jitter_coords_torch
    del relevant_time_mask_torch

    if return_intermediate_images:
        return reconstructed_images_proper, intermediates
    else:
        return reconstructed_images_proper
