from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from denoise_inverse_alg.flashed_recons_glm_components import single_bin_flashed_recons_compute_coupling_exp_arg, \
    mixreal_single_bin_flashed_recons_compute_coupling_exp_arg, \
    mixreal_single_bin_flashed_recons_compute_feedback_exp_arg
from denoise_inverse_alg.glm_inverse_alg import PackedGLMTensors


class NS_CT_TotalSimRetina(nn.Module):
    '''
    Class for jointly simulating spikes for every cell from GLM

    All of the spikes here are simulated, and then fed back into
        the model

    Requires that the GLM be strictly cuasal (output in the current timestep
        only depends on spikes that have occured in previous time bins), otherwise
        this doesn't work conceptually

    Uses the same timing design as the dejittering reconstruction, namely, that we
        only simulate spikes for one stimulus image (to prevent error compounding)
        and we initialize with real recorded data spikes (again, to minimize error
        compounding)
    '''

    def __init__(self,
                 glm_model_params: PackedGLMTensors,
                 input_frames: np.ndarray,
                 frame_transition_times: np.ndarray,
                 spike_bin_edges: np.ndarray,
                 spike_generation_callable: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.spike_generation_callable = spike_generation_callable

        stacked_spatial_filters = glm_model_params.spatial_filters
        stacked_timecourse_filters = glm_model_params.timecourse_filters
        stacked_feedback_filters = glm_model_params.feedback_filters
        stacked_coupling_filters = glm_model_params.coupling_filters
        stacked_bias = glm_model_params.bias
        coupling_idx_sel = glm_model_params.coupling_indices

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_pixels = self.height * self.width
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.max_coupled_cells = stacked_coupling_filters.shape[1]

        # precompute all of the quantities needed to upsample in time and do backprop
        forward_sel, forward_weights, backward_sel, backward_weights = batch_compute_interval_overlaps(
            frame_transition_times[None, :], spike_bin_edges[None, :])

        # store these quantities as buffers
        self.register_buffer('forward_sel', torch.tensor(forward_sel.squeeze(0), dtype=torch.long))
        self.register_buffer('backward_sel', torch.tensor(backward_sel.squeeze(0), dtype=torch.long))
        self.register_buffer('forward_weights', torch.tensor(forward_weights.squeeze(0), dtype=dtype))
        self.register_buffer('backward_weights', torch.tensor(backward_weights.squeeze(0), dtype=dtype))

        # shape (n_frames, n_pixels)
        self.register_buffer('input_frames',
                             torch.tensor(input_frames.reshape(input_frames.shape[0], -1), dtype=dtype))

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

        self.upsampler = TimeUpsampleTransposeFlat.apply

    def simulate_spikes(self,
                        initial_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param initial_spikes: shape (n_cells, n_bins_initial)
        :return:
        '''

        n_bins_initial = initial_spikes.shape[1]

        with torch.no_grad():
            # shape (n_input_frames, n_pixels)  @ (n_pixels, n_cells)
            # -> (n_input_frames, n_cells)
            spat_filt_applied = self.input_frames @ self.stacked_flat_spat_filters.T

            # (4) upsample the whole thing
            # shape (n_cells, n_bins)
            upsampled_movie = self.upsampler(spat_filt_applied[None, ...],
                                             self.forward_sel[None, ...],
                                             self.forward_weights[None, ...],
                                             self.backward_sel[None, ...],
                                             self.backward_weights[None, ...]).squeeze(0)

            # shape (n_cells, n_bins)
            stimulus_contrib_gensig = upsampled_movie + self.stacked_bias

            output_spikes = torch.zeros((self.n_cells, upsampled_movie.shape[1]), dtype=torch.float32,
                                        device=initial_spikes.device)
            output_spikes[:, :n_bins_initial] = initial_spikes
            for i in range(n_bins_initial, upsampled_movie.shape[1]):
                # shape (n_cells, n_bins_filter)
                relevant_observed_spikes = output_spikes[:, i - self.n_bins_filter:i]

                # first compute the feedback contribution to the generator signal
                # shape (n_cells, 1, n_bins_filter) @ (n_cells, n_bins_filter, 1)
                # -> (n_cells, 1, 1) -> (n_cells, )
                feedback_val = (relevant_observed_spikes[:, None, :] @
                                self.stacked_feedback_filters[:, :, None]).squeeze(2).squeeze(1)

                # then compute the coupling contribution to the generator signal

                # we want an output set of spike trains with shape
                # (n_cells, max_coupled_cells, n_bins_filter)

                # we need to pick our data out of output_bins_acc, which has shape
                # (n_cells, n_bins)
                # using indices contained in self.coupled_sel, which has shape
                # (n_cells, max_coupled_cells), which contains indices from 0 to (n_cells - 1)

                # in order to use gather, the number of dimensions of each need to match
                # (we need 3 total dimensions)

                # shape (n_cells, max_coupled_cells, n_bins_filter), index dimension is dim1 max_coupled_cells
                indices_repeated = self.coupled_sel[:, :, None].expand(-1, -1, self.n_bins_filter)

                # shape (n_cells, n_cells, n_bins_filter)
                observed_spikes_repeated = relevant_observed_spikes[None, :, :].expand(self.n_cells, -1, -1)

                # shape (n_cells, max_coupled_cells, n_bins_filter)
                selected_coupled_spike_trains = torch.gather(observed_spikes_repeated, 1, indices_repeated)

                # shape (n_cells, max_coupled_cells, 1, n_bins_filter) @ (n_cells, max_coupled_cells, n_bins_filter, 1)
                # -> (n_cells, max_coupled_cells, 1, 1)
                # -> (n_cells, max_coupled_cells)
                couple_val = (selected_coupled_spike_trains[:, :, None, :]
                              @ self.stacked_coupling_filters[:, :, :, None]).squeeze(3).squeeze(2)

                # shape (n_cells, )
                summed_couple_val = torch.sum(couple_val, dim=1)

                # shape (n_cells, )
                total_gensig = stimulus_contrib_gensig[:, i - 1] + summed_couple_val + feedback_val

                generated_spikes = self.spike_generation_callable(total_gensig)
                output_spikes[:, i] = generated_spikes

        return output_spikes


class NS_Flashed_TotalSimRetina(nn.Module):
    '''
    Class for jointly simulating spikes for every cell from the GLM
        for the flashed static stimuli.

    Makes use of the assumption that the stimulus is space-time separable

    All of the spikes are simulated and then fed back into the model
    '''

    def __init__(self,
                 glm_model_params: PackedGLMTensors,
                 stimulus_time_component: np.ndarray,
                 spike_generation_callable: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.spike_generation_callable = spike_generation_callable
        self.dtype = dtype

        stacked_spatial_filters = glm_model_params.spatial_filters
        stacked_timecourse_filters = glm_model_params.timecourse_filters
        stacked_feedback_filters = glm_model_params.feedback_filters
        stacked_coupling_filters = glm_model_params.coupling_filters
        stacked_bias = glm_model_params.bias
        coupling_idx_sel = glm_model_params.coupling_indices

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_pixels = self.height * self.width
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.max_coupled_cells = stacked_coupling_filters.shape[1]
        self.n_bins = stimulus_time_component.shape[0]

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

        #########################################################
        # shape (n_bins, )
        self.register_buffer('stimulus_time_component', torch.tensor(stimulus_time_component, dtype=dtype))

        # shape (n_cells, n_bins)
        self.register_buffer('precomputed_stim_time_conv',
                             torch.zeros((self.n_cells, stimulus_time_component.shape[0] - self.n_bins_filter + 1),
                                         dtype=dtype))

    def precompute_stim_time_conv(self) -> None:
        with torch.no_grad():
            # -> (n_cells, n_bins - n_bins_filter + 1)
            conv_output = F.conv1d(self.stimulus_time_component[None, None, :],
                                   self.stacked_timecourse_filters[:, None, :]).squeeze(0)
            self.precomputed_stim_time_conv[:] = conv_output[:]

    def simulate_spikes_with_kaput_cells(self,
                                         stimulus_frame: torch.Tensor,
                                         initial_spikes: torch.Tensor,
                                         real_data_spikes: torch.Tensor,
                                         is_kaput: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        :param stimulus_frame: shape (batch, height, width)
        :param initial_spikes: shape (batch, n_cells, n_bins_initial)
        :param real_data_spikes: shape (batch, n_cells, n_bins)
        :param is_kaput: shape (n_cells, ), boolean-valued, True if the cell GLM fit is deemed to be
            bad and shouldn't be used to simulate, False if good enough
        :return:
        '''

        n_bins_initial = initial_spikes.shape[2]
        batch = stimulus_frame.shape[0]

        # shape (batch, n_pixels)
        stimulus_flat = stimulus_frame.reshape(batch, -1)

        # shape (1, n_cells, n_pixels) @ (batch, n_pixels, 1)
        # -> (batch, n_cells, 1) -> (batch, n_cells)
        stim_filt_applied = (self.stacked_flat_spat_filters[None, :, :] @ stimulus_flat[:, :, None]).squeeze(2)

        # shape (batch, n_cells, 1) * *(1, n_cells, n_bins)
        # -> (batch, n_cells, n_bins)
        stimulus_gensig_contrib = stim_filt_applied[:, :, None] * self.precomputed_stim_time_conv[None, :, :]
        stimulus_gensig_contrib = stimulus_gensig_contrib + self.stacked_bias[None, :, :]

        generator_signal = torch.zeros((batch, self.n_cells, self.n_bins - n_bins_initial), dtype=self.dtype,
                                       device=initial_spikes.device)

        output_spikes = torch.zeros((batch, self.n_cells, self.n_bins), dtype=self.dtype,
                                    device=initial_spikes.device)
        output_spikes[:, :, :n_bins_initial] = initial_spikes[:]

        for gensig_write_ix, spike_ix in enumerate(range(n_bins_initial, self.n_bins)):
            # shape (batch, n_cells, n_bins_filter)
            relevant_simulated_spikes = output_spikes[:, :, spike_ix - self.n_bins_filter:spike_ix]
            relevant_data_spikes = real_data_spikes[:, :, spike_ix - self.n_bins_filter:spike_ix]

            # compute the feedback contribution to the generator signal
            # shape (batch, n_cells, 1, n_bins_filter) @ (1, n_cells, n_bins_filter, 1)
            # -> (batch, n_cells, 1, 1) -> (batch, n_cells)
            feedback_val = mixreal_single_bin_flashed_recons_compute_feedback_exp_arg(
                self.stacked_feedback_filters,
                relevant_simulated_spikes,
                relevant_data_spikes,
                is_kaput
            )

            coupling_val = mixreal_single_bin_flashed_recons_compute_coupling_exp_arg(
                self.stacked_coupling_filters,
                self.coupled_sel,
                relevant_simulated_spikes,
                relevant_data_spikes,
                is_kaput
            ).squeeze(2)

            timestep_gensig_val = coupling_val + feedback_val + stimulus_gensig_contrib[:, :, gensig_write_ix]

            generator_signal[:, :, gensig_write_ix] += timestep_gensig_val

            generated_spikes = self.spike_generation_callable(timestep_gensig_val)
            output_spikes[:, ~is_kaput, spike_ix] = generated_spikes[:, ~is_kaput]
            output_spikes[:, is_kaput, spike_ix] = real_data_spikes[:, is_kaput, spike_ix]

        return output_spikes, generator_signal

    def simulate_spikes(self,
                        stimulus_frame: torch.Tensor,
                        initial_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        :param stimulus_frame: shape (batch, height, width)
        :param initial_spikes: shape (batch, n_cells, n_bins_initial)
        :return:
        '''

        n_bins_initial = initial_spikes.shape[2]
        batch = stimulus_frame.shape[0]

        # shape (batch, n_pixels)
        stimulus_flat = stimulus_frame.reshape(batch, -1)

        # shape (1, n_cells, n_pixels) @ (batch, n_pixels, 1)
        # -> (batch, n_cells, 1) -> (batch, n_cells)
        stim_filt_applied = (self.stacked_flat_spat_filters[None, :, :] @ stimulus_flat[:, :, None]).squeeze(2)

        # shape (batch, n_cells, 1) * *(1, n_cells, n_bins)
        # -> (batch, n_cells, n_bins)
        stimulus_gensig_contrib = stim_filt_applied[:, :, None] * self.precomputed_stim_time_conv[None, :, :]
        stimulus_gensig_contrib = stimulus_gensig_contrib + self.stacked_bias[None, :, :]

        generator_signal = torch.zeros((batch, self.n_cells, self.n_bins - n_bins_initial), dtype=self.dtype,
                                       device=initial_spikes.device)

        output_spikes = torch.zeros((batch, self.n_cells, self.n_bins), dtype=self.dtype,
                                    device=initial_spikes.device)
        output_spikes[:, :, :n_bins_initial] = initial_spikes[:]

        for gensig_write_ix, spike_ix in enumerate(range(n_bins_initial, self.n_bins)):
            # shape (batch, n_cells, n_bins_filter)
            relevant_observed_spikes = output_spikes[:, :, spike_ix - self.n_bins_filter:spike_ix]

            # compute the feedback contribution to the generator signal
            # shape (batch, n_cells, 1, n_bins_filter) @ (1, n_cells, n_bins_filter, 1)
            # -> (batch, n_cells, 1, 1) -> (batch, n_cells)
            feedback_val = relevant_observed_spikes[:, :, None, :] @ self.stacked_feedback_filters[None, :, :, None]
            feedback_val = feedback_val.squeeze(3).squeeze(2)

            # then compute the coupling contribution to the generator signal
            # shape (batch, n_cells)
            coupling_val = single_bin_flashed_recons_compute_coupling_exp_arg(self.stacked_coupling_filters,
                                                                              self.coupled_sel,
                                                                              relevant_observed_spikes).squeeze(2)

            # shape (batch, n_cells)
            timestep_gensig_val = coupling_val + feedback_val + stimulus_gensig_contrib[:, :, gensig_write_ix]

            generator_signal[:, :, gensig_write_ix] += timestep_gensig_val

            generated_spikes = self.spike_generation_callable(timestep_gensig_val)
            output_spikes[:, :, spike_ix] = generated_spikes

        return output_spikes, generator_signal
