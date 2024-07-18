from abc import abstractmethod
from typing import Union, Callable, Tuple

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Binomial, Poisson
from torch.nn import functional as F


class SeparableBernoulliSigmoidSimGLM(nn.Module):

    def __init__(self,
                 stim_timecourse_filter: Union[np.ndarray, torch.Tensor],
                 feedback_filter: Union[np.ndarray, torch.Tensor],
                 coupling_filter: Union[np.ndarray, torch.Tensor],
                 spike_generation_callable: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.dtype = dtype
        self.n_bins_filter = stim_timecourse_filter.shape[0]

        # shape (n_bins_filter, )
        if isinstance(stim_timecourse_filter, np.ndarray):
            self.register_buffer('stim_timecourse_filter', torch.tensor(stim_timecourse_filter, dtype=dtype))
        else:
            self.register_buffer('stim_timecourse_filter', stim_timecourse_filter.detach().clone())

        # shape (n_bins_filter, )
        if isinstance(feedback_filter, np.ndarray):
            self.register_buffer('feedback_filter', torch.tensor(feedback_filter, dtype=dtype))
        else:
            self.register_buffer('feedback_filter', feedback_filter.detach().clone())

        # shape (n_coupled_cells, n_bins_filter)
        if isinstance(coupling_filter, np.ndarray):
            self.register_buffer('coupling_filter', torch.tensor(coupling_filter, dtype=dtype))
        else:
            self.register_buffer('coupling_filter', coupling_filter.detach().clone())

        self.spike_generation_callable = spike_generation_callable

    @abstractmethod
    def compute_spatial_exp_arg(self,
                                batched_spatial_stim: torch.Tensor) -> torch.Tensor:
        '''

        :param batched_spatial_stim: shape (batch, n_pixels, n_bins - n_bins_filter + 1)
        :return: shape (batch, n_bins - n_bins_filter + 1)
        '''
        raise NotImplementedError

    def compute_stimulus_time_component(self,
                                        stim_time: torch.Tensor) -> torch.Tensor:
        '''
        Applies the timecourse filter to the separable time component
            of the visual stimulus

        :param stim_time: shape (n_bins, )
        :return: shape (n_bins - n_bins_filter + 1, )
        '''

        # shape (1, 1, n_bins - n_bins_filter + 1)
        # -> (1, n_bins - n_bins_filter + 1)
        # -> (n_bins - n_bins_filter + 1, )
        filtered_stim_time = F.conv1d(stim_time[None, None, :],
                                      self.stim_timecourse_filter[None, None, :]).squeeze(1).squeeze(0)
        return filtered_stim_time

    def compute_coupling_exp_arg(self,
                                 batched_coupling_cells: torch.Tensor) -> torch.Tensor:
        '''

        :param batched_coupling_cells: shape (batch, n_coupled_cells, n_bins)
        :return: shape (batch, n_bins - n_bins_filter + 1)
        '''

        # shape (batch, 1, n_bins - n_bins_filter + 1)
        # -> (batch, n_bins - n_bins_filter + 1)
        filtered_coupling = F.conv1d(batched_coupling_cells,
                                     self.coupling_filter[None, :, :]).squeeze(1)

        return filtered_coupling

    def generate_spikes(self, spike_rate: torch.Tensor) -> torch.Tensor:
        '''

        :param spike_rate: any shape
        :return: same shape as spike_rate
        '''
        return self.spike_generation_callable(spike_rate)

    def simulate_cell(self,
                      batched_stim_spat: torch.Tensor,
                      stim_time: torch.Tensor,
                      batched_initial_spike_section: torch.Tensor,
                      batched_coupling_cells: torch.Tensor,
                      n_repeats: int = 1,
                      debug: bool = False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''
        Simulates GLM repeats for single-trial data (i.e. each stimulus
            image in batched_stim_spat was shown exactly once), with an
            optional number of repeats for each data point so that expectations
            and variances can be estimated empirically.

        :param batched_stim_spat: shape (batch, n_pixels)
        :param stim_time: shape (n_bins, )
        :param batched_initial_spike_section: shape (batch, n_bins_filt)
        :param batched_coupling_cells: (batch, n_coupled_cells, n_bins)
        :param n_repeats: number of repeats to simulate (so we can generate
            empirical statistics)
        :return: shape (batch, n_repeats, n_bins)
        '''

        with torch.inference_mode():
            batch, n_pixels = batched_stim_spat.shape
            n_bins = stim_time.shape[0]

            # first compute all of the components of the generator signal
            # that do not depend on time

            # shape (1, 1, n_bins - n_bins_filter + 1)
            # -> (n_bins - n_bins_filter + 1, )
            filtered_stim_time = self.compute_stimulus_time_component(stim_time)

            # shape (batch, n_pixels, 1) * (1, 1, n_bins - n_bins_filter + 1)
            # -> (batch, n_pixels, n_bins - n_bins_filter + 1)
            batched_stimulus_applied = batched_stim_spat[:, :, None] * filtered_stim_time[None, None, :]

            # shape (batch, n_bins - n_bins_filter + 1)
            stimulus_gensig_contrib = self.compute_spatial_exp_arg(batched_stimulus_applied)

            # shape (batch, n_bins - n_bins_filter + 1)
            coupling_exp_arg = self.compute_coupling_exp_arg(batched_coupling_cells)

            # shape (batch, n_bins - n_bins_filter + 1)
            generator_signal_piece = stimulus_gensig_contrib + coupling_exp_arg

            len_bins_gensig = generator_signal_piece.shape[1]
            len_bins_sim = len_bins_gensig - 1

            output_bins_acc = torch.zeros((batch, n_repeats, n_bins), dtype=self.dtype,
                                          device=generator_signal_piece.device)
            output_bins_acc[:, :, :self.n_bins_filter] = batched_initial_spike_section[:, None, :]

            output_gensig_acc = torch.zeros((batch, n_repeats, len_bins_sim), dtype=self.dtype,
                                            device=generator_signal_piece.device)

            for i in range(len_bins_sim):
                # shape (batch, n_repeats, n_bins_filter) @ (1, n_bins_filter, 1)
                # -> (batch, n_repeats, 1) -> (batch, n_repeats)
                batched_feedback_value = (output_bins_acc[:, :, i:i + self.n_bins_filter] @
                                          self.feedback_filter[None, :, None]).squeeze(2)

                # shape (batch, n_repeats)
                relev_gen_sig = generator_signal_piece[:, i][:, None] + batched_feedback_value

                output_bins_acc[:, :, i + self.n_bins_filter] = self.generate_spikes(relev_gen_sig)
                output_gensig_acc[:, :, i] = relev_gen_sig

            if debug:
                return output_bins_acc, stimulus_gensig_contrib, coupling_exp_arg, output_gensig_acc
            return output_bins_acc


class SimFlashedGLM(SeparableBernoulliSigmoidSimGLM):

    def __init__(self,
                 stim_spatial_filter: Union[np.ndarray, torch.Tensor],
                 bias: Union[np.ndarray, torch.Tensor],
                 stim_timecourse_filter: Union[np.ndarray, torch.Tensor],
                 feedback_filter: Union[np.ndarray, torch.Tensor],
                 coupling_filter: Union[np.ndarray, torch.Tensor],
                 spike_generation_callable: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):

        super().__init__(stim_timecourse_filter,
                         feedback_filter,
                         coupling_filter,
                         spike_generation_callable,
                         dtype=dtype)

        # shape (n_pixels, )
        if isinstance(stim_spatial_filter, np.ndarray):
            self.register_buffer('stim_spatial_filter', torch.tensor(stim_spatial_filter, dtype=dtype))
        else:
            self.register_buffer('stim_spatial_filter', stim_spatial_filter.detach().clone())

        # shape (1, )
        if isinstance(bias, np.ndarray):
            self.register_buffer('bias', torch.tensor(bias, dtype=dtype))
        else:
            self.register_buffer('bias', bias.detach().clone())

    def compute_spatial_exp_arg(self, batched_spatial_stim: torch.Tensor) -> torch.Tensor:
        '''

        :param batched_spatial_stim: shape shape (batch, n_pixels, n_bins - n_bins_filter + 1)
        :return: shape (batch, n_bins - n_bins_filter + 1)
        '''

        # shape (1, 1, n_pixels) @ (batch, n_pixels, n_bins - n_bins_filter + 1)
        # -> (batch, 1, n_bins - n_bins_filter + 1)
        # -> (batch, n_bins - n_bins_filter + 1)
        # -> (batch, )
        return (self.stim_spatial_filter[None, None, :] @ batched_spatial_stim).squeeze(1) + self.bias[None, :]


def batch_bernoulli_spike_generation(gensig: torch.Tensor) -> torch.Tensor:
    '''
    Simulates Bernoulli spiking process for a single bin with sigmoid
        nonlinearity

    (can only do a single bin because GLM has feedback filter, so future
     probabilities depend on what happens in the present)

    :param spike_rate: shape (batch, n_repeats)
    :return: shape (batch, n_repeats), 0 or 1-valued tensor
    '''

    return torch.bernoulli(torch.sigmoid(gensig))


def make_batch_binomial_spike_generation(binom_max: int) \
        -> Callable[[torch.Tensor], torch.Tensor]:
    def batch_binomial_spike_generation(gensig: torch.Tensor) -> torch.Tensor:
        binom_var = Binomial(total_count=binom_max,
                             probs=torch.sigmoid(gensig))
        return binom_var.sample()

    return batch_binomial_spike_generation


def batch_poisson_spike_generation(gensig: torch.Tensor) -> torch.Tensor:
    poisson_var = Poisson(torch.exp(gensig))
    return poisson_var.sample()


class ForwardSim_FeedbackOnlyTrialGLM(nn.Module):

    def __init__(self,
                 stim_spatial_filter: Union[np.ndarray, torch.Tensor],
                 stim_timecourse_filter: Union[np.ndarray, torch.Tensor],
                 feedback_filter: Union[np.ndarray, torch.Tensor],
                 bias: Union[np.ndarray, torch.Tensor],
                 spike_generation_callable: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        # length of the filter in bins
        self.n_bins_filter = stim_timecourse_filter.shape[0]
        self.dtype = dtype

        # shape (n_pixels, )
        if isinstance(stim_spatial_filter, np.ndarray):
            self.register_buffer('stim_spatial_filter', torch.tensor(stim_spatial_filter, dtype=dtype))
        else:
            self.register_buffer('stim_spatial_filter', stim_spatial_filter.detach().clone())

        # shape (n_bins_filter, )
        if isinstance(stim_timecourse_filter, np.ndarray):
            self.register_buffer('stim_timecourse_filter', torch.tensor(stim_timecourse_filter, dtype=dtype))
        else:
            self.register_buffer('stim_timecourse_filter', stim_timecourse_filter.detach().clone())

        # shape (n_bins_filter, )
        if isinstance(feedback_filter, np.ndarray):
            self.register_buffer('feedback_filter', torch.tensor(feedback_filter, dtype=dtype))
        else:
            self.register_buffer('feedback_filter', feedback_filter.detach().clone())

        # shape (1, )
        if isinstance(bias, np.ndarray):
            self.register_buffer('bias', torch.tensor(bias, dtype=dtype))
        else:
            self.register_buffer('bias', bias.detach().clone())

        self.spike_generation_fn = spike_generation_callable

    def simulate_cell(self,
                      batched_stim_spat: torch.Tensor,
                      stim_time: torch.Tensor,
                      batched_initial_spike_section: torch.Tensor,
                      n_repeats: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Simulates GLM repeats for single-trial data (i.e. each stimulus
            image in batched_stim_spat was shown exactly once)

        :param batched_stim_spat: shape (batch, n_pixels)
        :param stim_time: shape (n_bins, )
        :param batched_initial_spike_section: shape (batch, n_bins_filt)
        :param n_repeats: number of repeats to simulate (so we can generate
            empirical statistics)
        :return: shape (batch, n_repeats, n_bins - n_bins_filter)
        '''
        batch, n_pixels = batched_stim_spat.shape
        n_bins = stim_time.shape[0]

        # first compute all of the components of the generator signal
        # that do not depend on time

        # (batch, n_pixels) @ (n_pixels, 1) -> (batch, 1)
        filtered_stimulus_spat = (batched_stim_spat @ self.stim_spatial_filter[:, None])

        # shape (1, 1, n_bins - n_bins_filter + 1)
        # -> (1, n_bins - n_bins_filter + 1)
        filtered_stim_time = F.conv1d(stim_time[None, None, :],
                                      self.stim_timecourse_filter[None, None, :]).squeeze(1)

        # shape (batch, n_bins - n_bins_filter + 1)
        batched_stimulus_applied = filtered_stimulus_spat @ filtered_stim_time

        # shape (batch, n_bins - n_bins_filter + 1)
        generator_signal_piece = batched_stimulus_applied + self.bias[:, None]

        len_bins_gensig = generator_signal_piece.shape[1]
        len_bins_sim = len_bins_gensig - 1

        output_bins_acc = torch.zeros((batch, n_repeats, n_bins), dtype=self.dtype,
                                      device=generator_signal_piece.device)
        output_bins_acc[:, :, :self.n_bins_filter] = batched_initial_spike_section[:, None, :]

        output_gensig_acc = torch.zeros((batch, n_repeats, len_bins_sim), dtype=self.dtype,
                                        device=generator_signal_piece.device)

        for i in range(len_bins_sim):
            # shape (batch, n_repeats, n_bins_filter) @ (1, n_bins_filter, 1)
            # -> (batch, n_repeats, 1) -> (batch, n_repeats)
            batched_feedback_value = (output_bins_acc[:, :, i:i + self.n_bins_filter] @
                                      self.feedback_filter[None, :, None]).squeeze(2)

            # shape (batch, n_repeats)
            relev_gen_sig = generator_signal_piece[:, i][:, None] + batched_feedback_value

            output_bins_acc[:, :, i + self.n_bins_filter] = self.spike_generation_fn(relev_gen_sig)
            output_gensig_acc[:, :, i] = relev_gen_sig

        return output_bins_acc, output_gensig_acc


class CTSimGLM(nn.Module):
    '''
    Module for forward-simulation of GLM cell responses,
        for non-separable movie stimuli

    Since this module is mainly for human inspection
        (i.e. I'm just going to look at the simulatd spike trains
        and determine how good the model is by eye),
        it doesn't need to support large data and therefore
        we do not need to do anything in terms of low-rank bases
    '''

    def __init__(self,
                 spatial_filter: np.ndarray,
                 timecourse_filter: np.ndarray,
                 feedback_filter: np.ndarray,
                 coupling_filters: np.ndarray,
                 bias: np.ndarray,
                 spike_generation_callable: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        '''

        :param spatial_filter:
        :param timecourse_filter:
        :param feedback_filter:
        :param coupling_filters:
        :param bias:
        :param spike_generation_callable:
        :param dtype:
        '''

        super().__init__()

        self.dtype = dtype
        self.n_bins_filter = timecourse_filter.shape[0]

        # shape (n_pixels, )
        self.register_buffer('spatial_filter', torch.tensor(spatial_filter, dtype=dtype))

        # shape (n_bins_filter, )
        self.register_buffer('timecourse_filter', torch.tensor(timecourse_filter, dtype=dtype))

        # shape (n_bins_filter, )
        self.register_buffer('feedback_filter', torch.tensor(feedback_filter, dtype=dtype))

        # shape (n_coupled_cells, n_bins_filter)
        self.register_buffer('coupling_filters', torch.tensor(coupling_filters, dtype=dtype))

        # shape (1, )
        self.register_buffer('bias', torch.tensor(bias, dtype=dtype))

        self.spike_generation_callable = spike_generation_callable

    def compute_stimulus_exp_arg(self,
                                 stimulus_movie: torch.Tensor) -> torch.Tensor:
        '''

        :param stimulus_movie:  Flattened (in space) stimulus movie for the cell,
            shape (batch, n_pixels, n_bins)
        :return: argument to the generator signal nonlinearity, shape (batch, n_bins - n_bins_filter + 1)
        '''

        # shape (1, 1, n_pixels) @ (batch, n_pixels, n_bins)
        # -> (batch, 1, n_bins)
        spatial_filter_applied = self.spatial_filter[None, None, :] @ stimulus_movie

        # shape (batch, n_bins - n_bins_filter + 1)
        stimulus_exp_arg = F.conv1d(spatial_filter_applied,
                                    self.timecourse_filter[None, None, :]).squeeze(1)

        return stimulus_exp_arg

    def compute_coupling_exp_arg(self,
                                 coupling_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param coupling_spikes: Spike trains for the coupled cells,
            shape (batch, n_coupled_cells, n_bins)
        :return: argument to the generator signal nonlinearity, shape (batch, n_bins - n_bins_filter + 1)
        '''

        # shape (n_bins - n_bins_filter + 1, )
        filtered_coupling = F.conv1d(coupling_spikes,
                                     self.coupling_filters[None, :, :]).squeeze(1)

        return filtered_coupling

    def simulate_cell(self,
                      stim_movie: torch.Tensor,
                      initial_spike_section: torch.Tensor,
                      coupled_cell_spikes: torch.Tensor,
                      n_repeats: int = 1) \
            -> torch.Tensor:
        '''
        Simulates GLM for single movie (i.e. the stimulus was shown once and we have recorded
            responses from coupled cells for one trial), and we simulate N repeats
            of the center cell so that we can compute empirical expectations, variances, and
            other summary statisitics, as well as plot rasters.

        Note that we have to loop over the time bins, since the feedback contribution to the
            generator signal depends on spikes of the center cell, which have to be simulated
            one bin at a time.

        :param stim_movie: flattened stimulus movie, shape (batch, n_pixels, n_bins)
        :param initial_spike_section: shape (batch, n_bins_filt)
        :param coupled_cell_spikes: shape (batch, n_coupled_cells, n_bins)
        :param n_repeats: number of simulated repeats for the center cell
        :return: simulated center cell spikes, shape (batch, n_repeats, n_bins)
        '''

        with torch.no_grad():
            batch, n_pixels, n_bins = stim_movie.shape

            # first compute all components of the generator signal that do not depend on the
            # center cell spikes, since these are invariant

            # shape (batch, n_bins - n_bins_filter + 1)
            stimulus_contrib_gensig = self.compute_stimulus_exp_arg(stim_movie)

            # shape (batch, n_bins - n_bins_filter + 1, )
            coupling_contrib_gensig = self.compute_coupling_exp_arg(coupled_cell_spikes)

            # shape (batch, n_bins - n_bins_filter + 1)
            generator_signal_piece = stimulus_contrib_gensig + coupling_contrib_gensig + self.bias[None, :]

            len_bins_gensig = generator_signal_piece.shape[-1]
            len_bins_sim = len_bins_gensig - 1

            # shape (batch, n_repeats, n_bins)
            output_bins_acc = torch.zeros((batch, n_repeats, n_bins), dtype=self.dtype,
                                          device=generator_signal_piece.device)
            output_bins_acc[:, :, :self.n_bins_filter] = initial_spike_section[:, None, :]

            # now do the center cell spike simulation, one bin at a time
            # (note that we can parallelize over the repeats)
            for i in range(len_bins_sim):
                # shape (batch, n_repeats, n_bins_filter) @ (1, n_bins_filter, 1)
                # -> (batch, n_repeats, 1)
                feedback_value = output_bins_acc[:, :, i:i + self.n_bins_filter] @ self.feedback_filter[None, :, None]

                # shape (batch, )
                relev_gen_sig_piece = generator_signal_piece[:, i]

                # -> (batch, n_repeats, 1)
                relev_gen_sig = relev_gen_sig_piece[:, None, None] + feedback_value

                # shape (batch, n_repeats)
                spikes_to_put = self.spike_generation_callable(relev_gen_sig).squeeze(2)

                output_bins_acc[:, :, i + self.n_bins_filter] = spikes_to_put

            return output_bins_acc


class CTSimFBOnlyGLM(nn.Module):
    '''
    Module for forward-simulation of GLM cell responses,
        for non-separable movie stimuli

    Since this module is mainly for human inspection
        (i.e. I'm just going to look at the simulated spike trains
        and determine how good the model is by eye),
        it doesn't need to support large data and therefore
        we do not need to do anything in terms of low-rank bases
    '''

    def __init__(self,
                 spatial_filter: np.ndarray,
                 timecourse_filter: np.ndarray,
                 feedback_filter: np.ndarray,
                 bias: np.ndarray,
                 spike_generation_callable: Callable[[torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        '''

        :param spatial_filter:
        :param timecourse_filter:
        :param feedback_filter:
        :param coupling_filters:
        :param bias:
        :param spike_generation_callable:
        :param dtype:
        '''

        super().__init__()

        self.dtype = dtype
        self.n_bins_filter = timecourse_filter.shape[0]

        # shape (n_pixels, )
        self.register_buffer('spatial_filter', torch.tensor(spatial_filter, dtype=dtype))

        # shape (n_bins_filter, )
        self.register_buffer('timecourse_filter', torch.tensor(timecourse_filter, dtype=dtype))

        # shape (n_bins_filter, )
        self.register_buffer('feedback_filter', torch.tensor(feedback_filter, dtype=dtype))

        # shape (1, )
        self.register_buffer('bias', torch.tensor(bias, dtype=dtype))

        self.spike_generation_callable = spike_generation_callable

    def compute_stimulus_exp_arg(self,
                                 stimulus_movie: torch.Tensor) -> torch.Tensor:
        '''

        :param stimulus_movie:  Flattened (in space) stimulus movie for the cell,
            shape (batch, n_pixels, n_bins)
        :return: argument to the generator signal nonlinearity, shape (batch, n_bins - n_bins_filter + 1)
        '''

        # shape (1, 1, n_pixels) @ (batch, n_pixels, n_bins)
        # -> (batch, 1, n_bins)
        spatial_filter_applied = self.spatial_filter[None, None, :] @ stimulus_movie

        # shape (batch, n_bins - n_bins_filter + 1)
        stimulus_exp_arg = F.conv1d(spatial_filter_applied,
                                    self.timecourse_filter[None, None, :]).squeeze(1)

        return stimulus_exp_arg

    def simulate_cell(self,
                      stim_movie: torch.Tensor,
                      initial_spike_section: torch.Tensor,
                      n_repeats: int = 1) \
            -> torch.Tensor:
        '''
        Simulates GLM for single movie (i.e. the stimulus was shown once and we have recorded
            responses from coupled cells for one trial), and we simulate N repeats
            of the center cell so that we can compute empirical expectations, variances, and
            other summary statisitics, as well as plot rasters.

        Note that we have to loop over the time bins, since the feedback contribution to the
            generator signal depends on spikes of the center cell, which have to be simulated
            one bin at a time.

        :param stim_movie: flattened stimulus movie, shape (batch, n_pixels, n_bins)
        :param initial_spike_section: shape (batch, n_bins_filt)
        :param n_repeats: number of simulated repeats for the center cell
        :return: simulated center cell spikes, shape (batch, n_repeats, n_bins)
        '''

        with torch.no_grad():
            batch, n_pixels, n_bins = stim_movie.shape

            # first compute all components of the generator signal that do not depend on the
            # center cell spikes, since these are invariant

            # shape (batch, n_bins - n_bins_filter + 1)
            stimulus_contrib_gensig = self.compute_stimulus_exp_arg(stim_movie)

            # shape (batch, n_bins - n_bins_filter + 1)
            generator_signal_piece = stimulus_contrib_gensig + self.bias[None, :]

            len_bins_gensig = generator_signal_piece.shape[-1]
            len_bins_sim = len_bins_gensig - 1

            # shape (n_repeats, n_bins)
            output_bins_acc = torch.zeros((batch, n_repeats, n_bins), dtype=self.dtype,
                                          device=generator_signal_piece.device)
            output_bins_acc[:, :, :self.n_bins_filter] = initial_spike_section[:, None, :]

            # now do the center cell spike simulation, one bin at a time
            # (note that we can parallelize over the repeats)
            for i in range(len_bins_sim):
                # shape (batch, n_repeats, n_bins_filter) @ (1, n_bins_filter, 1)
                # -> (batch, n_repeats, 1)
                feedback_value = output_bins_acc[:, :, i:i + self.n_bins_filter] @ self.feedback_filter[None, :, None]

                # shape (batch, )
                relev_gen_sig_piece = generator_signal_piece[:, i]

                # -> (batch, n_repeats, 1)
                relev_gen_sig = relev_gen_sig_piece[:, None, None] + feedback_value

                # -> (batch, n_repeats)
                spikes_to_put = self.spike_generation_callable(relev_gen_sig).squeeze(2)

                output_bins_acc[:, :, i + self.n_bins_filter] = spikes_to_put

            return output_bins_acc
