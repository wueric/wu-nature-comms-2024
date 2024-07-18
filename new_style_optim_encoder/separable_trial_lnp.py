from typing import Callable, Optional, Union, Tuple, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from convex_optim_base.prox_optim import ProxSolverParams
from new_style_optim.prox_optim import _optim_FISTA
from new_style_optim.accelerated_unconstrained_optim import _optim_unconstrained_FISTA
from new_style_optim_encoder.glm_components import _flashed_spatial_stimulus_contrib_gensig, \
    _movie_spatial_stimulus_contrib_gensig, _flashed_timecourse_stimulus_contrib_gensig, \
    _movie_timecourse_stimulus_contrib_gensig
from new_style_optim_encoder.glm_prox import _lnp_prox_project_variables


class Flashed_PreappliedSpatialSingleCellLNPLoss(nn.Module):

    def __init__(self,
                 spat_filt_basis_weights: Union[torch.Tensor, np.ndarray],
                 timecourse_basis_weights: Union[torch.Tensor, np.ndarray],
                 bias: Union[torch.Tensor, np.ndarray],
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.loss_callable = loss_callable

        # shape (1, n_basis_spat_filt)
        if isinstance(spat_filt_basis_weights, np.ndarray):
            self.register_buffer('spat_filt_basis_weights', torch.tensor(spat_filt_basis_weights, dtype=dtype))
        else:
            self.register_buffer('spat_filt_basis_weights', spat_filt_basis_weights)

        # shape (1, n_basis_timecourse)
        if isinstance(timecourse_basis_weights, np.ndarray):
            self.register_buffer('timecourse_basis_weights', torch.tensor(timecourse_basis_weights, dtype=dtype))
        else:
            self.register_buffer('timecourse_basis_weights', timecourse_basis_weights)

        # shape (1, )
        if isinstance(bias, np.ndarray):
            self.register_buffer('bias', torch.tensor(bias, dtype=dtype))
        else:
            self.register_buffer('bias', bias)

    def forward(self,
                stim_spat: torch.Tensor,
                timecourse_basis_applied: torch.Tensor,
                observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param spat_preapplied: shape (batch, n_dim_spat)
        :param timecourse_basis_applied: shape (n_basis_time, n_bins)
        :param observed_spikes: shape (batch, n_bins)
        :return:
        '''

        # shape (batch, n_dim_spat) @ (n_dim_spat, 1)
        # -> (batch, 1)
        spat_mult = (stim_spat @ self.spat_filt_basis_weights.T)

        # shpae (1, n_basis_time) @ (n_basis_time, n_bins)
        # -> (1, n_bins)
        timecourse_mult = self.timecourse_basis_weights @ timecourse_basis_applied

        # shape (batch, n_bins)
        gensig = spat_mult * timecourse_mult + self.bias[None, :]

        return self.loss_callable(gensig, observed_spikes)


class Flashed_WNReg_SpatFitLNP(nn.Module):
    '''
    New-style optimization module for learning LNP spatial filters
        for flashed natural scenes data, regularized by fitting jointly
        with a small amount of white noise data as well

    Optimized using the new-style optimizer so that we can support mixed
        precision training
    '''

    def __init__(self,
                 dim_stimulus: int,
                 wn_weight: float,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 spat_filter_sparsity_lambda: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 init_bias: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 stim_spat_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None):
        super().__init__()

        self.dim_stimulus = dim_stimulus

        self.wn_weight = wn_weight
        self.loss_callable = loss_callable
        self.spatial_sparsity_l1_lambda = spat_filter_sparsity_lambda

        # OPTIMIZATION VARIABLE 0, shape (1, n_basis_stim_spat)
        if stim_spat_init_guess is not None:
            if stim_spat_init_guess.shape != (1, self.dim_stimulus):
                raise ValueError(f"stim_spat_init_guess must be (1, {self.dim_stimulus})")
            if isinstance(stim_spat_init_guess, np.ndarray):
                self.stim_spat_w = nn.Parameter(torch.tensor(stim_spat_init_guess, dtype=dtype),
                                                requires_grad=True)
            else:
                self.stim_spat_w = nn.Parameter(stim_spat_init_guess.detach().clone(),
                                                requires_grad=True)
        else:
            self.stim_spat_w = nn.Parameter(torch.empty((1, self.dim_stimulus), dtype=dtype),
                                            requires_grad=True)
            nn.init.uniform_(self.stim_spat_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1, bias, shape (1, )
        if init_bias is not None:
            if init_bias.shape != (1,):
                raise ValueError(f"init_bias must have shape {(1,)}")
            if isinstance(init_bias, np.ndarray):
                self.bias = nn.Parameter(torch.tensor(init_bias, dtype=dtype), requires_grad=True)
            else:
                self.bias = nn.Parameter(init_bias.detach().clone(), requires_grad=True)
        else:
            self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
            nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   spat_filt_w: torch.Tensor,
                   bias: torch.Tensor,

                   nscenes_spat_filt_frame: torch.Tensor,
                   nscenes_time_filt: torch.Tensor,
                   nscenes_cell_spikes: torch.Tensor,

                   wn_filt_movie: torch.Tensor,
                   wn_cell_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param spat_filt_w: shape (1, n_basis_stim_spat)
        :param bias: shape (1, )
        :param nscenes_spat_filt_frame: shape (batch, dim_stim)
        :param nscenes_time_filt: shape (n_bins - n_bins_filter + 1, )
        :param nscenes_cell_spikes: shape (batch, n_bins - n_bins_filter + 1)
        :param wn_filt_movie: shape (dim_stim, n_bins_wn - n_bins_filter + 1)
        :param wn_cell_spikes: shape (n_bins_wn - n_bins_filter + 1, )
        :return:
        '''

        ns_stimulus_contrib = _flashed_spatial_stimulus_contrib_gensig(spat_filt_w, nscenes_spat_filt_frame,
                                                                       nscenes_time_filt)
        ns_gen_sig = bias + ns_stimulus_contrib
        nscenes_spiking_loss = self.loss_callable(ns_gen_sig, nscenes_cell_spikes)

        wn_stimulus_contrib = _movie_spatial_stimulus_contrib_gensig(spat_filt_w, wn_filt_movie)
        wn_gen_sig = bias + wn_stimulus_contrib
        wn_spiking_loss = self.loss_callable(wn_gen_sig, wn_cell_spikes)

        spiking_loss = nscenes_spiking_loss + self.wn_weight * wn_spiking_loss
        return spiking_loss


    def forward(self, *data_args) -> torch.Tensor:
        return self._eval_loss(*self.parameters(recurse=False),
                               *data_args)

    def make_loss_eval_callable(self, *data_args) \
            -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]:
        '''
        This function is solely for evaluating a forward pass without needing
            gradients, for doing the backtracking line search
        :return:
        '''

        def loss_callable(*optim_var_args) -> float:
            with torch.no_grad():
                return self._eval_loss(*optim_var_args, *data_args).item()

        return loss_callable

    def prox_project_variables(self,
                               spat_filter_w: torch.Tensor,
                               bias_w: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        IMPORTANT: the order of the parameters for this method MUST match
            the order of the nn.Parameters declared in the constructor
            otherwise the FISTA implementation will fail.

        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        :return:
        '''

        return _lnp_prox_project_variables(spat_filter_w, bias_w,
                                           self.spatial_sparsity_l1_lambda)

    def return_spat_filt_parameters_np(self) -> np.ndarray:
        return self.stim_spat_w.detach().cpu().numpy()

    def return_spat_filt_parameters(self) -> torch.Tensor:
        return self.stim_spat_w.detach().clone()

    def set_optimization_parameters(self,
                                    stim_spat_w: Optional[torch.Tensor] = None,
                                    bias: Optional[torch.Tensor] = None):
        if stim_spat_w is not None:
            self.stim_spat_w.data[:] = stim_spat_w.data[:]
        if bias is not None:
            self.bias.data[:] = bias.data[:]

    def return_parameters_torch(self) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        return (
            self.stim_spat_w.detach().clone(),
            self.bias.detach().clone(),
        )

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray]:

        return (
            self.stim_spat_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
        )


class Flashed_WNReg_TimeFiltLNP(nn.Module):

    def __init__(self,
                 n_basis_stim_time: int,
                 wn_regularizer_weight: float,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32,
                 stim_time_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_bias: Optional[Union[np.ndarray, torch.Tensor]] = None):

        super().__init__()

        self.n_basis_stim_time = n_basis_stim_time

        self.wn_regularizer_weight = wn_regularizer_weight

        self.loss_callable = loss_callable

        # OPTIMIZATION VARIABLE 0, shape (1, n_basis_stim_time)
        if stim_time_init_guess is not None:
            if stim_time_init_guess.shape != (1, n_basis_stim_time):
                raise ValueError(f"stim_time_init_guess must be (1, {n_basis_stim_time}), got {stim_time_init_guess.shape}")
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

        # OPTIMIZATION VARIABLE 1, bias, shape (1, )
        if init_bias is not None:
            if init_bias.shape != (1,):
                raise ValueError(f"init_bias must have shape {(1,)}")
            if isinstance(init_bias, np.ndarray):
                self.bias = nn.Parameter(torch.tensor(init_bias, dtype=dtype), requires_grad=True)
            else:
                self.bias = nn.Parameter(init_bias.detach().clone(), requires_grad=True)
        else:
            self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
            nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   time_filt_w: torch.Tensor,
                   bias: torch.Tensor,

                   ns_spat_filt_movie: torch.Tensor,
                   ns_time_basis_filt_movie: torch.Tensor,
                   ns_binned_spikes_cell: torch.Tensor,

                   wn_time_basis_filt_movie: torch.Tensor,
                   wn_binned_spikes_cell: torch.Tensor) -> torch.Tensor:
        '''

        :param coupling_filt_w: shape (n_coupled_cells, n_basis_coupling)
        :param feedback_filt_w: shape (1, n_basis_feedback)
        :param time_filt_w: shape (1, n_basis_stim_time)
        :param bias: shape (1, )
        :param coupling_aux: shape (n_coupled_cells, )
        :param ns_spat_filt_movie: shape (batch, )
        :param ns_time_basis_filt_movie: shape (n_basis_stim_time, n_bins_ - n_bins_filter + 1)
        :param ns_binned_spikes_cell: shape (batch, n_bins - n_bins_filter + 1)
        :param ns_filtered_feedback: shape (batch, n_basis_feedback, n_bins - n_bins_filter + 1)
        :param ns_filtered_coupling: shape (batch, n_coupled_cells, n_basis_feedback, n_bins - n_bins_filter + 1)
        :param wn_time_basis_filt_movie: shape (n_basis_stim_time, n_bins_wn - n_bins_filter + 1)
        :param wn_binned_spikes_cell: shape (n_bins_wn - n_bins_filter + 1, )
        :param wn_filtered_feedback: shape (1, n_basis_feedback, n_bins - n_bins_filter + 1)
        :param wn_filtered_coupling: shape (n_coupled_cells, n_basis_coupling)
        :return:
        '''

        ns_stimulus_contrib = _flashed_timecourse_stimulus_contrib_gensig(time_filt_w,
                                                                          ns_spat_filt_movie,
                                                                          ns_time_basis_filt_movie)

        ns_gen_sig = bias + ns_stimulus_contrib
        ns_spiking_loss = self.loss_callable(ns_gen_sig, ns_binned_spikes_cell)
        wn_stimulus_contrib = _movie_timecourse_stimulus_contrib_gensig(time_filt_w,
                                                                        wn_time_basis_filt_movie)

        wn_gen_sig = bias + wn_stimulus_contrib
        wn_spiking_loss = self.loss_callable(wn_gen_sig, wn_binned_spikes_cell)

        spiking_loss = ns_spiking_loss + self.wn_regularizer_weight * wn_spiking_loss

        return spiking_loss

    def forward(self, *data_args) -> torch.Tensor:
        return self._eval_loss(*self.parameters(recurse=False),
                               *data_args)

    def make_loss_eval_callable(self, *data_args) \
            -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]:
        '''
        This function is solely for evaluating a forward pass without needing
            gradients, for doing the backtracking line search
        :return:
        '''

        def loss_callable(*optim_var_args) -> float:
            with torch.no_grad():
                return self._eval_loss(*optim_var_args, *data_args).item()

        return loss_callable

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

    def set_optimization_parameters(self,
                                    timecourse_w: Optional[torch.Tensor] = None,
                                    bias: Optional[torch.Tensor] = None):
        if timecourse_w is not None:
            self.stim_time_w.data[:] = timecourse_w.data[:]
        if bias is not None:
            self.bias.data[:] = bias.data[:]

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray]:

        return (
            self.stim_time_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
        )

    def return_parameters_torch(self) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        return (
            self.stim_time_w.detach().clone(),
            self.bias.detach().clone(),
        )


def new_style_LNP_joint_wn_flashed_ns_alternating_optim(
        ns_stimulus_frames: torch.Tensor,
        ns_spikes_cell: torch.Tensor,
        ns_stim_time_basis_conv: torch.Tensor,
        wn_stimulus_frames: torch.Tensor,
        wn_spikes_cell: torch.Tensor,
        stim_time_basis: torch.Tensor,
        loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        solver_params_iter: Tuple[Iterator[ProxSolverParams], Iterator[ProxSolverParams]],
        n_iters_outer_opt: int,
        device: torch.device,
        weight_wn: float = 0.1,
        l1_spat_sparse_lambda: float = 0.0,
        initial_guess_timecourse: Optional[torch.Tensor] = None,
        initial_guess_bias: Optional[torch.Tensor] = None,
        outer_opt_verbose: bool = False):

    '''

    :param ns_stimulus_frames: raw stimulus frames (either with or without pre-application of basis filter),
        shape (batch, dim_spat)
    :param ns_spikes_cell: binned spikes for the cells being fit, shape (batch, n_bins - n_bins_filter + 1)
        if fitting strictly causal model, (batch, n_bins - n_bins_filter) if fitting non-strictly causal model
    :param ns_stim_time_basis_conv: shape (n_basis_time, n_bins - n_bins_filter + 1)
    :param wn_stimulus_frames: shape (dim_spat, n_bins_wn, )
    :param wn_spikes_cell: shape (n_bins_wn - n_bins_filter + 1, ) if fitting strictly causal model,
    :param stim_time_basis: shape (n_basis_time, n_bins_filter)
    :param loss_callable:
    :param solver_params_iter:
    :param n_iters_outer_opt:
    :param device:
    :param weight_wn:
    :param l1_spat_sparse_lambda:
    :param initial_guess_timecourse:
    :param initial_guess_bias:
    :param outer_opt_verbose:
    :return:
    '''

    n_flashed_trials, dim_spat = ns_stimulus_frames.shape
    assert wn_stimulus_frames.shape[0] == dim_spat, f'wn dimension and nscenes dimension must match, got {wn_stimulus_frames.shape[0]} and {dim_spat}'

    n_basis_stim_time, _ = ns_stim_time_basis_conv.shape

    # shape (n_timecourse_basis, )
    timecourse_w = initial_guess_timecourse
    bias= initial_guess_bias

    prev_iter_spatial_filter = None
    with torch.no_grad(), autocast('cuda'):

        # shape (1, n_timecourse_basis) @ (n_timecourse_basis, n_bins_filter)
        # -> (1, n_bins_filter) -> (n_bins_filter, )
        timecourse_filter = (timecourse_w[None, :] @ stim_time_basis).squeeze(0)
        max_timecourse = torch.linalg.norm(timecourse_filter)

        timecourse_w = timecourse_w.div_(max_timecourse)
        timecourse_filter = timecourse_filter.div_(max_timecourse)

    spat_opt_module = Flashed_WNReg_SpatFitLNP(
        dim_spat,
        weight_wn,
        loss_callable,
        spat_filter_sparsity_lambda=l1_spat_sparse_lambda,
        init_bias=initial_guess_bias,
        dtype=torch.float32
    ).to(device)

    timecourse_opt_module = Flashed_WNReg_TimeFiltLNP(
        n_basis_stim_time,
        weight_wn,
        loss_callable,
        stim_time_init_guess=timecourse_w,
        init_bias=bias,
        dtype=torch.float32
    ).to(device)

    for (iter_num, spat_solver_params, time_solver_params) in \
            zip(range(n_iters_outer_opt), solver_params_iter[0], solver_params_iter[1]):

        with torch.no_grad(), autocast('cuda'):
            # shape (1, n_basis_time) @ (n_basis_time, n_bins - n_bins_filter + 1)
            # -> (1, n_bins - n_bins_filter + 1) -> (n_bins - n_bins_filter + 1, )
            ns_filtered_stim_time = (timecourse_w @ ns_stim_time_basis_conv).squeeze(0)

            # shape (dim_spat, n_bins_wn - n_bins_filter + 1)
            wn_filtered_stim_time = F.conv1d(wn_stimulus_frames[:, None, :],
                                             timecourse_filter[None, :, :]).squeeze(1)

        spat_opt_module.set_optimization_parameters(
            stim_spat_w=prev_iter_spatial_filter,
            bias=bias,
        )

        loss_spatial = _optim_FISTA(
            spat_opt_module,
            spat_solver_params,
            ns_stimulus_frames,
            ns_filtered_stim_time,
            ns_spikes_cell,
            wn_filtered_stim_time,
            wn_spikes_cell,
        )

        del ns_filtered_stim_time, wn_filtered_stim_time

        prev_iter_spatial_filter, bias = spat_opt_module.return_parameters_torch()

        if outer_opt_verbose:
            print(f"Iter {iter_num} spatial opt. loss {loss_spatial}")

        with torch.no_grad(), autocast('cuda'):
            # shape (batch, n_pixels) @ (n_pixels, 1) -> (batch, 1) -> (batch, )
            ns_spatial_filter_applied = (ns_stimulus_frames @ prev_iter_spatial_filter.T).squeeze(1)

            # shape (1, dim_spat)(dim_spat, n_bins_wn)
            # -> (1, n_bins_wn)
            wn_spatial_filter_applied = (prev_iter_spatial_filter @ wn_stimulus_frames)

            # shape (n_basis_stim_time, 1, n_bins_wn - n_bins_filter + 1)
            # -> (n_basis_stim_time, n_bins_wn - n_bins_filter + 1)
            wn_spatial_filter_time_basis_applied = F.conv1d(wn_spatial_filter_applied[:, None, :],
                                                            stim_time_basis[:, None, :]).squeeze(1).squeeze(0)

        timecourse_opt_module.set_optimization_parameters(
            timecourse_w=timecourse_w,
            bias=bias,
        )

        loss_timecourse = _optim_unconstrained_FISTA(
            timecourse_opt_module,
            time_solver_params,
            ns_spatial_filter_applied,
            ns_stim_time_basis_conv,
            ns_spikes_cell,
            wn_spatial_filter_time_basis_applied,
            wn_spikes_cell,
        )

        del ns_spatial_filter_applied, wn_spatial_filter_applied, wn_spatial_filter_time_basis_applied

        timecourse_w, bias = timecourse_opt_module.return_parameters_torch()

        if outer_opt_verbose:
            print(f"Iter {iter_num} timecourse opt. loss {loss_timecourse}")

        with torch.no_grad(), autocast('cuda'):

            # shape (1, n_timecourse_basis) @ (n_timecourse_basis, n_bins_filter)
            # -> (1, n_bins_filter) -> (n_bins_filter, )
            timecourse_filter = (timecourse_w[None, :] @ stim_time_basis).squeeze(0)
            max_timecourse = torch.linalg.norm(timecourse_filter)

            timecourse_w = timecourse_w / max_timecourse
            timecourse_filter = timecourse_filter / max_timecourse

            prev_iter_spatial_filter = prev_iter_spatial_filter * max_timecourse

    del spat_opt_module, timecourse_opt_module
    return loss_timecourse, (prev_iter_spatial_filter, timecourse_w, bias)
