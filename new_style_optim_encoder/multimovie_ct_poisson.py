'''
Module for fitting LNP encoding models, where spikes are binned according to
    the stimulus monitor frame rate, approximately 120 Hz or so. In practice,
    this means that the spike bin cutoff times are inferred from the stimulus
    synchronization triggers rather than from the recording array sample rate.

These models don't require any sort of temporal upsampling or downsampling of
    the stimulus movie, since we work explicitly at the movie frame rate only.
'''

import numpy as np

import torch
import torch.nn as nn

from typing import Callable, Optional, Union, List, Tuple


class NS_MM_Timecourse_FrameRateLNP(nn.Module):
    '''
    Used to fit the separable timecourse stimulus filter component of the LNP model, jointly
        with the bias. Holds the separable spatial stimulus filter component fixed;
        in fact, this is done by precomputing the application of the stimulus filter
        outside of this module for better computational efficiency

    Implicitly assumes exponential nonlinearity, since that is the only
        nonlinearity that I know of that guarantees that the negative log-likelihood
        is convex; this property is not so important when fitting the LNP model,
        but is critical when doing the reconstruction optimization.

    Since there are no coupling filters or feedback filters, and the bin size for LNP
        is quite large, we discard the requirement for strict causality; this way
        the stimulus frame at time t CAN AFFECT the firing rate at time t.

    '''
    def __init__(self,
                 n_basis_stim_time: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 stim_time_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 multimovie_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.n_basis_stim_time = n_basis_stim_time

        self.loss_callable = loss_callable

        self.use_multimovie_weights = False
        if multimovie_weights is not None:
            if isinstance(multimovie_weights, np.ndarray):
                self.register_buffer('multimovie_weights', torch.tensor(multimovie_weights, dtype=dtype))
            else:
                self.register_buffer('multimovie_weights', multimovie_weights.detach().clone())

        # OPTIMIZATION VARIABLE 0, shape (1, n_basis_stim_time)
        if stim_time_init_guess is not None:
            if stim_time_init_guess.shape != (1, n_basis_stim_time):
                raise ValueError("stim_time_init_guess must be (1, {0})".format(n_basis_stim_time))
            if isinstance(stim_time_init_guess, np.ndarray):
                self.stim_time_w = nn.Parameter(torch.tensor(stim_time_init_guess, dtype=dtype),
                                                requires_grad=True)
            else:
                self.stim_time_w = nn.Parameter(stim_time_init_guess.detach().clone(),
                                                requires_grad=True)
        else:
            self.stim_time_w = nn.Parameter(torch.empty((1, n_basis_stim_time), dtype=dtype),
                                            requires_grad=True)
            nn.init.uniform_(self.stim_time_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1
        self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
        nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

    def _stimulus_contrib_gensig(self,
                                 time_filt_w: torch.Tensor,
                                 multimovie_filtered_list: List[torch.Tensor]) -> List[torch.Tensor]:
        '''
        Computes stimulus-driven contribution to the generator signal

        :param time_filt_w: shape (1, n_basis_stim_time)
        :param multimovie_filter_list: List of (n_basis_stim_time, n_bins - n_bins_filter + 1),
            where n_bins could be different for each item in the list
        :return:
        '''

        ret_list = []  # type: List[torch.Tensor]
        for movie_flat_binrate in multimovie_filtered_list:
            # shape (1, n_basis_stim_time) @ (n_basis_stim_time, n_bins - n_bins_filter + 1)
            # -> (1, n_bins - n_bins_filter + 1)
            # -> (n_bins - n_bins_filter + 1, )
            time_filt_applied = (time_filt_w @ movie_flat_binrate).squeeze(0)
            ret_list.append(time_filt_applied)

        return ret_list

    def _eval_loss(self,
                   timecourse_w: torch.Tensor,
                   bias: torch.Tensor,
                   filt_multimovie: List[torch.Tensor],
                   cell_spike_multibin: List[torch.Tensor]) -> torch.Tensor:
        '''

        :param timecourse_w:
        :param bias:
        :param filt_multimovie:
        :param cell_spike_multibin:
        :return:
        '''

        stimulus_contrib = self._stimulus_contrib_gensig(timecourse_w, filt_multimovie)
        gen_sig_list = [bias + x for x in stimulus_contrib]

        # we ditch the strict causality assumption since the frame rate is rather low
        # and this assumption is not necessary without the GLM in any case
        spiking_loss_per_movie = [self.loss_callable(gen_sig, binned_spikes)
                                  for gen_sig, binned_spikes in zip(gen_sig_list, cell_spike_multibin)]

        if self.use_multimovie_weights:
            stacked_losses = torch.stack(spiking_loss_per_movie, dim=0)
            spiking_loss = torch.sum(self.multimovie_weights * stacked_losses)
        else:
            spiking_loss = torch.mean(torch.stack(spiking_loss_per_movie, dim=0))

        return spiking_loss

    def forward(self,
                filt_multimovie: List[torch.Tensor],
                cell_spike_multibin: List[torch.Tensor]) -> torch.Tensor:
        return self._eval_loss(self.stim_time_w,
                               self.bias,
                               filt_multimovie,
                               cell_spike_multibin)

    def make_loss_eval_callable(self,
                                filt_multimovie: List[torch.Tensor],
                                cell_spike_multibin: List[torch.Tensor]) \
            -> Callable[[torch.Tensor, torch.Tensor], float]:

        def loss_callable(timecourse_w: torch.Tensor,
                          bias: torch.Tensor) -> float:
            with torch.no_grad():
                return self._eval_loss(
                    timecourse_w, bias,
                    filt_multimovie, cell_spike_multibin).item()

        return loss_callable

    def prox_project_variables(self,
                               timecourse_w: torch.Tensor,
                               bias: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        return timecourse_w, bias

    def return_timecourse_params_np(self) -> np.ndarray:
        return self.stim_time_w.detach().cpu().numpy()

    def return_timecourse_params(self) -> torch.Tensor:
        return self.stim_time_w.detach().clone()

    def set_time_filter(self, set_to: torch.Tensor) -> None:
        '''
        Sets the timecourse filter value (i.e. for setting an initialization point)
            for more efficient coordinate descent
        :param set_to: value ot set self.stim_time_w to
        :return:
        '''
        assert set_to.shape == self.stim_time_w.shape, f'cannot set self.stim_time_w to shape {set_to.shape}'
        self.stim_time_w.data[:] = set_to.data[:]

    def clone_parameters_model(self, coord_desc_other) -> None:
        self.bias.data[:] = coord_desc_other.bias.data[:]

        self.zero_grad()

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray]:

        return (
            self.stim_time_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy()
        )


class NS_MM_Spatial_FrameRateLNP(nn.Module):
    '''
    Used to fit the separable spatial stimulus filter component of the LNP model, jointly
        with the bias. Holds the separable timecourse stimulus filter component fixed;
        in fact, this is done by precomputing the convolution of the stimulus with the timecourse
        outside of this module for better computational efficiency

    Implicitly assumes exponential nonlinearity, since that is the only
        nonlinearity that I know of that guarantees that the negative log-likelihood
        is convex; this property is not so important when fitting the LNP model,
        but is critical when doing the reconstruction optimization.

    Since there are no coupling filters or feedback filters, and the bin size for LNP
        is quite large, we discard the requirement for strict causality; this way
        the stimulus frame at time t CAN AFFECT the firing rate at time t.
    '''

    def __init__(self,
                 n_basis_stim_spat: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 spatial_sparsity_l1_lambda: float = 0.0,
                 stim_spat_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 multimovie_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 dtype: torch.dtype = torch.float32):
        '''

        :param n_basis_stim_spat:
        :param loss_callable:
        :param spatial_sparsity_l1_lambda:
        :param stim_spat_init_guess:
        :param multimovie_weights:
        :param dtype:
        '''

        super().__init__()

        self.n_basis_stim_spat = n_basis_stim_spat

        self.loss_callable = loss_callable
        self.spatial_sparsity_l1_lambda = spatial_sparsity_l1_lambda

        self.use_multimovie_weights = False
        if multimovie_weights is not None:
            if isinstance(multimovie_weights, np.ndarray):
                self.register_buffer('multimovie_weights', torch.tensor(multimovie_weights, dtype=dtype))
            else:
                self.register_buffer('multimovie_weights', multimovie_weights.detach().clone())

        # OPTIMIZATION VARIABLE 0, shape (1, n_basis_stim_spat)
        if stim_spat_init_guess is not None:
            if stim_spat_init_guess.shape != (1, n_basis_stim_spat):
                raise ValueError("stim_spat_init_guess must be (1, {0})".format(n_basis_stim_spat))
            if isinstance(stim_spat_init_guess, np.ndarray):
                self.stim_spat_w = nn.Parameter(torch.tensor(stim_spat_init_guess, dtype=dtype),
                                                requires_grad=True)
            else:
                self.stim_spat_w = nn.Parameter(stim_spat_init_guess.detach().clone(),
                                                requires_grad=True)
        else:
            self.stim_spat_w = nn.Parameter(torch.empty((1, n_basis_stim_spat), dtype=dtype),
                                            requires_grad=True)
            nn.init.uniform_(self.stim_spat_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1
        self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
        nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

    def _stimulus_contrib_gensig(self,
                                 spat_filt_w: torch.Tensor,
                                 multimovie_filtered_list: List[torch.Tensor]):
        '''
        Computes the stimulus-driven contribution to the generator signal,
            for each snippet of movie

        :param spat_filt_w: shape (1, n_basis_stim_spat)
        :param multimovie_filtered_list: List of (n_basis_stim_spat, n_bins - n_bins_filter + 1)
            where n_bins could be different for each item in the list
        :return:
        '''
        ret_list = []  # type: List[torch.Tensor]
        for time_filt_applied in multimovie_filtered_list:
            # shape (1, n_basis_stim_spat) @ (n_basis_stim_spat, n_bins - n_bins_filter + 1)
            # -> (1, n_bins - n_bins_filter + 1) -> (n_bins - n_bins_filter + 1)
            spat_filt_applied = (spat_filt_w @ time_filt_applied).squeeze(0)
            ret_list.append(spat_filt_applied)

        return ret_list

    def _eval_loss(self,
                   spat_filt_w: torch.Tensor,
                   bias: torch.Tensor,
                   filt_multimovie: List[torch.Tensor],
                   cell_spike_multibin: List[torch.Tensor]) -> torch.Tensor:

        stimulus_contrib = self._stimulus_contrib_gensig(spat_filt_w,
                                                         filt_multimovie)

        gen_sig_list = [bias + x for x in stimulus_contrib]

        # Note: the lack of strict causality results in this line of code
        spiking_loss_per_movie = [self.loss_callable(gen_sig, binned_spikes)
                                  for gen_sig, binned_spikes in zip(gen_sig_list, cell_spike_multibin)]

        if self.use_multimovie_weights:
            stacked_losses = torch.stack(spiking_loss_per_movie, dim=0)
            spiking_loss = torch.sum(self.multimovie_weights * stacked_losses)
        else:
            spiking_loss = torch.mean(torch.stack(spiking_loss_per_movie, dim=0))

        return spiking_loss

    def forward(self,
                filt_multimovie: List[torch.Tensor],
                cell_spike_multibin: List[torch.Tensor]) -> torch.Tensor:

        return self._eval_loss(self.stim_spat_w, self.bias,
                               filt_multimovie, cell_spike_multibin)

    def make_loss_eval_callable(self,
                                filt_multimovie: List[torch.Tensor],
                                cell_spike_multibin: List[torch.Tensor]) \
            -> Callable[[torch.Tensor, torch.Tensor], float]:

        def loss_callable(spat_filt_w: torch.Tensor,
                          bias: torch.Tensor) -> float:
            return self._eval_loss(spat_filt_w, bias,
                                   filt_multimovie, cell_spike_multibin).item()

        return loss_callable

    def prox_project_variables(self,
                               spat_filter_w: torch.Tensor,
                               bias_w: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        '''
        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        Everything else we can just pass through

        :param args:
        :param kwargs:
        :return:
        '''

        with torch.no_grad():

            if self.spatial_sparsity_l1_lambda != 0.0:
                spat_filter_w = torch.clamp_min_(spat_filter_w - self.spatial_sparsity_l1_lambda, 0.0) \
                                - torch.clamp_min_(-spat_filter_w - self.spatial_sparsity_l1_lambda, 0.0)

        return spat_filter_w, bias_w

    def return_spat_filt_parameters_np(self) -> np.ndarray:
        return self.stim_spat_w.detach().cpu().numpy()

    def return_spat_filt_parameters(self) -> torch.Tensor:
        return self.stim_spat_w.detach().clone()

    def set_spat_filter(self, set_to: torch.Tensor) -> None:
        '''
        Sets the spatial filter value (i.e. for setting an initialization point)
            for more efficient coordinate descent
        :param set_to: value to set self.stim_spat_w to
        :return:
        '''
        assert set_to.shape == self.stim_spat_w.shape, f'cannot set self.stim_spat_w with tensor of shape {set_to.shape}'
        self.stim_spat_w.data[:] = set_to.data[:]

    def clone_parameters_model(self, coord_desc_other) \
            -> None:
        self.bias.data[:] = coord_desc_other.bias.data[:]
        self.zero_grad()

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.stim_spat_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy()
        )

