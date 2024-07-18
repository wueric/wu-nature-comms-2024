from typing import Callable, Optional, Union, Tuple, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from convex_optim_base.prox_optim import ProxSolverParams
from new_style_optim.prox_optim import _optim_FISTA
from new_style_optim_encoder.glm_components import _flashed_timecourse_stimulus_contrib_gensig, \
    _movie_coupling_feedback_contrib_gensig, \
    _movie_timecourse_stimulus_contrib_gensig, _flashed_spatial_stimulus_contrib_gensig, \
    _movie_spatial_stimulus_contrib_gensig, _flashed_feedback_contrib_gensig, _flashed_coupling_contrib_gensig
from new_style_optim_encoder.glm_prox import _time_model_prox_project_variables, _spatial_model_prox_project_variables


class EvalFlashedGLM(nn.Module):

    def __init__(self,
                 stim_spat_w: Union[torch.Tensor, np.ndarray],
                 stim_time_w: Union[torch.Tensor, np.ndarray],
                 feedback_w: Union[torch.Tensor, np.ndarray],
                 coupling_w: Union[torch.Tensor, np.ndarray],
                 bias: Union[torch.Tensor, np.ndarray],
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__()

        self.loss_callable = loss_callable

        # shape (1, n_dim_stim_spat)
        if isinstance(stim_spat_w, np.ndarray):
            stim_spat_w = torch.tensor(stim_spat_w, dtype=torch.float32)
        self.register_buffer('stim_spat_w', stim_spat_w)

        # shape (1, n_basis_stim_time)
        if isinstance(stim_time_w, np.ndarray):
            stim_time_w = torch.tensor(stim_time_w, dtype=torch.float32)
        self.register_buffer('stim_time_w', stim_time_w)

        # shape (1, n_basis_feedback)
        if isinstance(feedback_w, np.ndarray):
            feedback_w = torch.tensor(feedback_w, dtype=torch.float32)
        self.register_buffer('feedback_w', feedback_w)

        # shape (n_coupled_cells, n_basis_coupling)_
        if isinstance(coupling_w, np.ndarray):
            coupling_w = torch.tensor(coupling_w, dtype=torch.float32)
        self.register_buffer('coupling_w', coupling_w)

        # shape (1, )
        if isinstance(bias, np.ndarray):
            bias = torch.tensor(bias, dtype=torch.float32)
        self.register_buffer('bias', bias)

    def forward(self,
                nscenes_spat_basis_filt: torch.Tensor,
                nscenes_time_basis_filt: torch.Tensor,
                nscenes_cell_spikes: torch.Tensor,
                nscenes_filt_feedback: torch.Tensor,
                nscenes_filt_coupling: torch.Tensor) -> torch.Tensor:

        '''

        :param nscenes_spat_basis_filt: shape (batch, n_basis_stim_spat), flashed stimulus
            image for each trial
        :param nscenes_time_basis_filt: shape (n_basis_stim_time, n_bins - n_bins_filter + 1)
        :param nscenes_cell_spikes:
        :param nscenes_filt_feedback:
        :param nscenes_filt_coupling:
        :return:
        '''

        # shape (1, n_basis_stim_time) @ (n_basis_stim_time, n_bins - n_bins_filter + 1)
        # -> (1, n_bins - n_bins_filter + 1)  -> (n_bins - n_bins_filter + 1, )
        time_filter_applied = (self.stim_time_w @ nscenes_time_basis_filt).squeeze(0)

        ns_stimulus_contrib = _flashed_spatial_stimulus_contrib_gensig(self.stim_spat_w,
                                                                       nscenes_spat_basis_filt,
                                                                       time_filter_applied)

        ns_feedback_contrib = _flashed_feedback_contrib_gensig(self.feedback_w,
                                                               nscenes_filt_feedback)
        ns_coupling_contrib = _flashed_coupling_contrib_gensig(self.coupling_w,
                                                               nscenes_filt_coupling)

        ns_gen_sig = self.bias[None, :] + ns_stimulus_contrib + ns_feedback_contrib + ns_coupling_contrib

        nscenes_spiking_loss = self.loss_callable(ns_gen_sig[:, :-1],
                                                  nscenes_cell_spikes[:, 1:])

        return nscenes_spiking_loss


class EvalFlashedFBOnlyGLM(nn.Module):

    def __init__(self,
                 stim_spat_w: Union[torch.Tensor, np.ndarray],
                 stim_time_w: Union[torch.Tensor, np.ndarray],
                 feedback_w: Union[torch.Tensor, np.ndarray],
                 bias: Union[torch.Tensor, np.ndarray],
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__()

        self.loss_callable = loss_callable

        # shape (1, n_dim_stim_spat)
        if isinstance(stim_spat_w, np.ndarray):
            stim_spat_w = torch.tensor(stim_spat_w, dtype=torch.float32)
        self.register_buffer('stim_spat_w', stim_spat_w)

        # shape (1, n_basis_stim_time)
        if isinstance(stim_time_w, np.ndarray):
            stim_time_w = torch.tensor(stim_time_w, dtype=torch.float32)
        self.register_buffer('stim_time_w', stim_time_w)

        # shape (1, n_basis_feedback)
        if isinstance(feedback_w, np.ndarray):
            feedback_w = torch.tensor(feedback_w, dtype=torch.float32)
        self.register_buffer('feedback_w', feedback_w)

        # shape (1, )
        if isinstance(bias, np.ndarray):
            bias = torch.tensor(bias, dtype=torch.float32)
        self.register_buffer('bias', bias)

    def forward(self,
                nscenes_spat_basis_filt: torch.Tensor,
                nscenes_time_basis_filt: torch.Tensor,
                nscenes_cell_spikes: torch.Tensor,
                nscenes_filt_feedback: torch.Tensor) -> torch.Tensor:

        '''

        :param nscenes_spat_basis_filt: shape (batch, n_basis_stim_spat), flashed stimulus
            image for each trial
        :param nscenes_time_basis_filt: shape (n_basis_stim_time, n_bins - n_bins_filter + 1)
        :param nscenes_cell_spikes:
        :param nscenes_filt_feedback:
        :param nscenes_filt_coupling:
        :return:
        '''

        # shape (1, n_basis_stim_time) @ (n_basis_stim_time, n_bins - n_bins_filter + 1)
        # -> (1, n_bins - n_bins_filter + 1)  -> (n_bins - n_bins_filter + 1, )
        time_filter_applied = (self.stim_time_w @ nscenes_time_basis_filt).squeeze(0)

        ns_stimulus_contrib = _flashed_spatial_stimulus_contrib_gensig(self.stim_spat_w,
                                                                       nscenes_spat_basis_filt,
                                                                       time_filter_applied)
        ns_feedback_contrib = _flashed_feedback_contrib_gensig(self.feedback_w,
                                                               nscenes_filt_feedback)
        ns_gen_sig = self.bias[None, :] + ns_stimulus_contrib + ns_feedback_contrib

        nscenes_spiking_loss = self.loss_callable(ns_gen_sig[:, :-1],
                                                  nscenes_cell_spikes[:, 1:])

        return nscenes_spiking_loss


class NS_SpatFitGLM(nn.Module):
    '''
    New-style module for learning GLM + spatial filters

    No white noise regularization

    Optimized using the new-style optimizer that subclasses
        torch.optim.Optimizer, so this is compatible with amp
        mixed precision optimization for smaller memory footprint
        and faster training
    '''

    def __init__(self,
                 dim_stimulus: int,
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 spat_filter_sparsity_lambda: float = 0.0,
                 group_sparse_reg_lambda: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 stim_spat_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_feedback_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_bias: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_norm: Optional[Union[np.ndarray, torch.Tensor]] = None):

        super().__init__()

        self.dim_stimulus = dim_stimulus

        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.loss_callable = loss_callable
        self.spatial_sparsity_l1_lambda = spat_filter_sparsity_lambda
        self.group_sparse_reg_lambda = group_sparse_reg_lambda

        # OPT VARIABLE 0: coupling_w, shape (n_coupled_cells, n_basis_coupling)
        if init_coupling_w is not None:
            if init_coupling_w.shape != (n_coupled_cells, n_basis_coupling):
                raise ValueError(f"init_coupling_w must have shape {(n_coupled_cells, n_basis_coupling)}")
            if isinstance(init_coupling_w, np.ndarray):
                self.coupling_w = nn.Parameter(torch.tensor(init_coupling_w, dtype=dtype), requires_grad=True)
            else:
                self.coupling_w = nn.Parameter(init_coupling_w.detach().clone(), requires_grad=True)
        else:
            self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1, feedback_w, shape (1, n_basis_feedback)
        if init_feedback_w is not None:
            if init_feedback_w.shape != (1, n_basis_feedback):
                raise ValueError(f"init_feedback_w must have shape {(1, n_basis_feedback)}")
            if isinstance(init_feedback_w, np.ndarray):
                self.feedback_w = nn.Parameter(torch.tensor(init_feedback_w, dtype=dtype), requires_grad=True)
            else:
                self.feedback_w = nn.Parameter(init_feedback_w.detach().clone(), requires_grad=True)
        else:
            self.feedback_w = nn.Parameter(torch.empty((1, n_basis_feedback), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.feedback_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 2, shape (1, n_basis_stim_spat)
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

        # OPTIMIZATION VARIABLE 3, bias, shape (1, )
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

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )
        if init_coupling_norm is not None:
            if init_coupling_norm.shape != (n_coupled_cells,):
                raise ValueError(f"init_coupling_norm must have shape {(n_coupled_cells,)}")
            if isinstance(init_coupling_norm, np.ndarray):
                self.coupling_filter_norm = nn.Parameter(torch.tensor(init_coupling_norm, dtype=dtype),
                                                         requires_grad=True)
            else:
                self.coupling_filter_norm = nn.Parameter(init_coupling_norm.detach().clone(),
                                                         requires_grad=True)
        else:
            self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                     requires_grad=True)
            nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   coupling_filt_w: torch.Tensor,
                   feedback_filt_w: torch.Tensor,
                   spat_filt_w: torch.Tensor,
                   bias: torch.Tensor,
                   coupling_aux: torch.Tensor,
                   nscenes_spat_filt_frame: torch.Tensor,
                   nscenes_time_filt: torch.Tensor,
                   nscenes_cell_spikes: torch.Tensor,
                   nscenes_filt_feedback: torch.Tensor,
                   nscenes_filt_coupling: torch.Tensor) -> torch.Tensor:
        '''

        :param coupling_filt_w: shape (n_coupled_cells, n_basis_coupling)
        :param feedback_filt_w: shape (1, n_basis_feedback)
        :param spat_filt_w: shape (1, n_basis_stim_spat)
        :param bias: shape (1, )
        :param coupling_aux: shape (n_coupled_cells, )
        :param nscenes_spat_filt_frame: shape (batch, dim_stim)
        :param nscenes_time_filt: shape (n_bins - n_bins_filter + 1, )
        :param nscenes_cell_spikes: shape (batch, n_bins - n_bins_filter + 1)
        :param nscenes_filt_feedback: shape (batch, n_basis_feedback, n_bins - n_bins_filter + 1)
        :param nscenes_filt_coupling: shape (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
        :param wn_filt_movie: shape (dim_stim, n_bins_wn - n_bins_filter + 1)
        :param wn_cell_spikes: shape (n_bins_wn - n_bins_filter + 1, )
        :param wn_filt_feedback: shape (1, n_basis_feedback, n_bins_wn - n_bins_filter + 1)
        :param wn_filt_coupling: shape (n_coupled_cells, n_basis_coupling, n_bins_wn - n_bins_filter + 1)
        :return:
        '''

        ns_stimulus_contrib = _flashed_spatial_stimulus_contrib_gensig(spat_filt_w, nscenes_spat_filt_frame,
                                                                       nscenes_time_filt)

        ns_feedback_contrib = _flashed_feedback_contrib_gensig(feedback_filt_w,
                                                               nscenes_filt_feedback)
        ns_coupling_contrib = _flashed_coupling_contrib_gensig(coupling_filt_w,
                                                               nscenes_filt_coupling)

        ns_gen_sig = bias + ns_stimulus_contrib + ns_feedback_contrib + ns_coupling_contrib

        spiking_loss = self.loss_callable(ns_gen_sig[:, :-1],
                                          nscenes_cell_spikes[:, 1:])

        if self.group_sparse_reg_lambda != 0.0:
            regularization_penalty = self.group_sparse_reg_lambda * torch.sum(coupling_aux)
            return spiking_loss + regularization_penalty
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
                               coupling_filter_w: torch.Tensor,
                               feedback_filter_w: torch.Tensor,
                               spat_filter_w: torch.Tensor,
                               bias_w: torch.Tensor,
                               coupling_auxvar: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        IMPORTANT: the order of the parameters for this method MUST match
            the order of the nn.Parameters declared in the constructor
            otherwise the FISTA implementation will fail.

        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        :return:
        '''

        return _spatial_model_prox_project_variables(coupling_filter_w, feedback_filter_w,
                                                     spat_filter_w, bias_w, coupling_auxvar,
                                                     self.spatial_sparsity_l1_lambda)

    def return_spat_filt_parameters_np(self) -> np.ndarray:
        return self.stim_spat_w.detach().cpu().numpy()

    def return_spat_filt_parameters(self) -> torch.Tensor:
        return self.stim_spat_w.detach().clone()

    def set_optimization_parameters(self,
                                    stim_spat_w: Optional[torch.Tensor] = None,
                                    coupling_w: Optional[torch.Tensor] = None,
                                    feedback_w: Optional[torch.Tensor] = None,
                                    bias: Optional[torch.Tensor] = None,
                                    coupling_filt_norm: Optional[torch.Tensor] = None):
        if stim_spat_w is not None:
            self.stim_spat_w.data[:] = stim_spat_w.data[:]
        if coupling_w is not None:
            self.coupling_w.data[:] = coupling_w.data[:]
        if feedback_w is not None:
            self.feedback_w.data[:] = feedback_w.data[:]
        if bias is not None:
            self.bias.data[:] = bias.data[:]
        if coupling_filt_norm is not None:
            self.coupling_filter_norm.data[:] = coupling_filt_norm.data[:]

    def return_parameters_torch(self) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            self.coupling_w.detach().clone(),
            self.feedback_w.detach().clone(),
            self.stim_spat_w.detach().clone(),
            self.bias.detach().clone(),
            self.coupling_filter_norm.detach().clone()
        )

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            self.stim_spat_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
            self.coupling_filter_norm.detach().cpu().numpy()
        )


class NS_TimecourseFitGLM(nn.Module):

    def __init__(self,
                 n_basis_stim_time: int,
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 group_sparse_reg_lambda: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 stim_time_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_feedback_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_bias: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_norm: Optional[Union[np.ndarray, torch.Tensor]] = None):

        super().__init__()

        self.n_basis_stim_time = n_basis_stim_time
        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.loss_callable = loss_callable

        self.group_sparse_reg_lambda = group_sparse_reg_lambda

        # OPT VARIABLE 0: coupling_w, shape (n_coupled_cells, n_basis_coupling)
        if init_coupling_w is not None:
            if init_coupling_w.shape != (n_coupled_cells, n_basis_coupling):
                raise ValueError(f"init_coupling_w must have shape {(n_coupled_cells, n_basis_coupling)}")
            if isinstance(init_coupling_w, np.ndarray):
                self.coupling_w = nn.Parameter(torch.tensor(init_coupling_w, dtype=dtype), requires_grad=True)
            else:
                self.coupling_w = nn.Parameter(init_coupling_w.detach().clone(), requires_grad=True)
        else:
            self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1, feedback_w, shape (1, n_basis_feedback)
        if init_feedback_w is not None:
            if init_feedback_w.shape != (1, n_basis_feedback):
                raise ValueError(f"init_feedback_w must have shape {(1, n_basis_feedback)}")
            if isinstance(init_feedback_w, np.ndarray):
                self.feedback_w = nn.Parameter(torch.tensor(init_feedback_w, dtype=dtype), requires_grad=True)
            else:
                self.feedback_w = nn.Parameter(init_feedback_w.detach().clone(), requires_grad=True)
        else:
            self.feedback_w = nn.Parameter(torch.empty((1, n_basis_feedback), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.feedback_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 2, shape (1, n_basis_stim_time)
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

        # OPTIMIZATION VARIABLE 3, bias, shape (1, )
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

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )
        if init_coupling_norm is not None:
            if init_coupling_norm.shape != (n_coupled_cells,):
                raise ValueError(f"init_coupling_norm must have shape {(n_coupled_cells,)}")
            if isinstance(init_coupling_norm, np.ndarray):
                self.coupling_filter_norm = nn.Parameter(torch.tensor(init_coupling_norm, dtype=dtype),
                                                         requires_grad=True)
            else:
                self.coupling_filter_norm = nn.Parameter(init_coupling_norm.detach().clone(),
                                                         requires_grad=True)
        else:
            self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                     requires_grad=True)
            nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   coupling_filt_w: torch.Tensor,
                   feedback_filt_w: torch.Tensor,
                   time_filt_w: torch.Tensor,
                   bias: torch.Tensor,
                   coupling_aux: torch.Tensor,

                   ns_spat_filt_movie: torch.Tensor,
                   ns_time_basis_filt_movie: torch.Tensor,
                   ns_binned_spikes_cell: torch.Tensor,
                   ns_filtered_feedback: torch.Tensor,
                   ns_filtered_coupling: torch.Tensor) -> torch.Tensor:
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

        ns_feedback_contrib = _flashed_feedback_contrib_gensig(feedback_filt_w,
                                                               ns_filtered_feedback)
        ns_coupling_contrib = _flashed_coupling_contrib_gensig(coupling_filt_w,
                                                               ns_filtered_coupling)

        ns_gen_sig = bias + ns_stimulus_contrib + ns_feedback_contrib + ns_coupling_contrib
        spiking_loss = self.loss_callable(ns_gen_sig[:, :-1],
                                          ns_binned_spikes_cell[:, 1:])

        if self.group_sparse_reg_lambda != 0.0:
            regularization_penalty = self.group_sparse_reg_lambda * torch.sum(coupling_aux)
            return spiking_loss + regularization_penalty

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
                               coupling_filter_w: torch.Tensor,
                               feedback_filter_w: torch.Tensor,
                               timecourse_w: torch.Tensor,
                               bias_w: torch.Tensor,
                               coupling_auxvar: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        '''
        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        IMPORTANT: the parameters here need to be in the same order as the order
            of the nn.Parameter declared in the constructor, otherwise the FISTA
            algorithm isn't going to be able to successfully apply the projection
        :param args:
        :param kwargs:
        :return:
        '''

        return _time_model_prox_project_variables(coupling_filter_w, feedback_filter_w,
                                                  timecourse_w, bias_w, coupling_auxvar)

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
                                    coupling_w: Optional[torch.Tensor] = None,
                                    feedback_w: Optional[torch.Tensor] = None,
                                    bias: Optional[torch.Tensor] = None,
                                    coupling_filt_norm: Optional[torch.Tensor] = None):
        if timecourse_w is not None:
            self.stim_time_w.data[:] = timecourse_w.data[:]
        if coupling_w is not None:
            self.coupling_w.data[:] = coupling_w.data[:]
        if feedback_w is not None:
            self.feedback_w.data[:] = feedback_w.data[:]
        if bias is not None:
            self.bias.data[:] = bias.data[:]
        if coupling_filt_norm is not None:
            self.coupling_filter_norm.data[:] = coupling_filt_norm.data[:]

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            self.stim_time_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
            self.coupling_filter_norm.detach().cpu().numpy()
        )

    def return_parameters_torch(self) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            self.coupling_w.detach().clone(),
            self.feedback_w.detach().clone(),
            self.stim_time_w.detach().clone(),
            self.bias.detach().clone(),
            self.coupling_filter_norm.detach().clone()
        )


def new_style_flashed_ns_alternating_optim(
        ns_stimulus_frames: torch.Tensor,
        ns_spikes_cell: torch.Tensor,
        ns_stim_time_basis_conv: torch.Tensor,
        ns_feedback_basis_conv: torch.Tensor,
        ns_coupling_basis_conv: torch.Tensor,
        stim_time_basis: torch.Tensor,
        loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        solver_params_iter: Tuple[Iterator[ProxSolverParams], Iterator[ProxSolverParams]],
        n_iters_outer_opt: int,
        device: torch.device,
        l21_group_sparse_lambda: float = 0.0,
        l1_spat_sparse_lambda: float = 0.0,
        initial_guess_timecourse: Optional[torch.Tensor] = None,
        initial_guess_coupling: Optional[torch.Tensor] = None,
        initial_guess_coupling_norm: Optional[torch.Tensor] = None,
        initial_guess_feedback: Optional[torch.Tensor] = None,
        initial_guess_bias: Optional[torch.Tensor] = None,
        outer_opt_verbose: bool = False):
    '''
    Function for performing alternating optimization between

    :param ns_stimulus_frames: raw stimulus frames (either with or without pre-application of basis filter),
        shape (batch, dim_spat)
    :param ns_spikes_cell: binned spikes for the cells being fit, shape (batch, n_bins - n_bins_filter + 1)
        if fitting strictly causal model, (batch, n_bins - n_bins_filter) if fitting non-strictly causal model
    :param ns_stim_time_basis_conv: shape (n_basis_time, n_bins - n_bins_filter + 1)
    :param ns_feedback_basis_conv: shape (batch, n_basis_feedback, n_bins - n_bins_filter + 1)
    :param ns_coupling_basis_conv: shape (batch, n_coupled_cells, n_bins_total - n_bins_filter + 1)
    :param stim_time_basis: shape (n_basis_time, n_bins_filter)
    :param loss_callable:
    :param solver_params_iter:
    :param n_iters_outer_opt:
    :param device:
    :param l21_group_sparse_lambda:
    :param l1_spat_sparse_lambda:
    :param initial_guess_timecourse:
    :param initial_guess_coupling:
    :param initial_guess_coupling_norm:
    :param initial_guess_feedback:
    :param initial_guess_bias:
    :param outer_opt_verbose:
    :return:
    '''

    n_flashed_trials, dim_spat = ns_stimulus_frames.shape

    n_basis_stim_time, _ = ns_stim_time_basis_conv.shape
    n_basis_feedback = ns_feedback_basis_conv.shape[1]
    n_coupled_cells, n_basis_coupling = ns_coupling_basis_conv.shape[1], ns_coupling_basis_conv.shape[2]

    # shape (n_timecourse_basis, )
    timecourse_w = initial_guess_timecourse

    coupling_w, feedback_w = initial_guess_coupling, initial_guess_feedback
    bias, coupling_filt_norm = initial_guess_bias, initial_guess_coupling_norm

    prev_iter_spatial_filter = None

    with torch.no_grad(), autocast('cuda'):

        # shape (1, n_timecourse_basis) @ (n_timecourse_basis, n_bins_filter)
        # -> (1, n_bins_filter) -> (n_bins_filter, )
        timecourse_filter = (timecourse_w[None, :] @ stim_time_basis).squeeze(0)
        max_timecourse = torch.linalg.norm(timecourse_filter)

        timecourse_w = timecourse_w.div_(max_timecourse)

    spat_opt_module = NS_SpatFitGLM(
        dim_spat,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        loss_callable,
        spat_filter_sparsity_lambda=l1_spat_sparse_lambda,
        group_sparse_reg_lambda=l21_group_sparse_lambda,
        init_feedback_w=initial_guess_feedback,
        init_coupling_w=initial_guess_coupling,
        init_coupling_norm=initial_guess_coupling_norm,
        init_bias=initial_guess_bias,
    ).to(device)

    timecourse_opt_module = NS_TimecourseFitGLM(
        n_basis_stim_time,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        loss_callable,
        group_sparse_reg_lambda=l21_group_sparse_lambda,
        stim_time_init_guess=timecourse_w,
        init_feedback_w=feedback_w,
        init_coupling_w=coupling_w,
        init_bias=bias,
        init_coupling_norm=coupling_filt_norm,
    ).to(device)

    for (iter_num, spat_solver_params, time_solver_params) in \
            zip(range(n_iters_outer_opt), solver_params_iter[0], solver_params_iter[1]):

        with torch.no_grad(), autocast('cuda'):
            # shape (1, n_basis_time) @ (n_basis_time, n_bins - n_bins_filter + 1)
            # -> (1, n_bins - n_bins_filter + 1) -> (n_bins - n_bins_filter + 1, )
            ns_filtered_stim_time = (timecourse_w @ ns_stim_time_basis_conv).squeeze(0)

        spat_opt_module.set_optimization_parameters(
            stim_spat_w=prev_iter_spatial_filter,
            coupling_w=coupling_w,
            feedback_w=feedback_w,
            bias=bias,
            coupling_filt_norm=coupling_filt_norm
        )

        loss_spatial = _optim_FISTA(
            spat_opt_module,
            spat_solver_params,
            ns_stimulus_frames,
            ns_filtered_stim_time,
            ns_spikes_cell,
            ns_feedback_basis_conv,
            ns_coupling_basis_conv,
        )

        del ns_filtered_stim_time

        coupling_w, feedback_w, prev_iter_spatial_filter, bias, coupling_filt_norm = spat_opt_module.return_parameters_torch()

        if outer_opt_verbose:
            print(f"Iter {iter_num} spatial opt. loss {loss_spatial}")

        with torch.no_grad(), autocast('cuda'):
            # shape (batch, n_pixels) @ (n_pixels, 1) -> (batch, 1) -> (batch, )
            ns_spatial_filter_applied = (ns_stimulus_frames @ prev_iter_spatial_filter.T).squeeze(1)

        timecourse_opt_module.set_optimization_parameters(
            timecourse_w=timecourse_w,
            coupling_w=coupling_w,
            feedback_w=feedback_w,
            bias=bias,
            coupling_filt_norm=coupling_filt_norm
        )

        loss_timecourse = _optim_FISTA(
            timecourse_opt_module,
            time_solver_params,
            ns_spatial_filter_applied,
            ns_stim_time_basis_conv,
            ns_spikes_cell,
            ns_feedback_basis_conv,
            ns_coupling_basis_conv,
        )

        del ns_spatial_filter_applied

        coupling_w, feedback_w, timecourse_w, bias, coupling_filt_norm = timecourse_opt_module.return_parameters_torch()

        if outer_opt_verbose:
            print(f"Iter {iter_num} timecourse opt. loss {loss_timecourse}")

        with torch.no_grad(), autocast('cuda'):

            # shape (1, n_timecourse_basis) @ (n_timecourse_basis, n_bins_filter)
            # -> (1, n_bins_filter) -> (n_bins_filter, )
            timecourse_filter = (timecourse_w[None, :] @ stim_time_basis).squeeze(0)
            max_timecourse = torch.linalg.norm(timecourse_filter)

            timecourse_w = timecourse_w / max_timecourse

            prev_iter_spatial_filter = prev_iter_spatial_filter * max_timecourse

    del spat_opt_module, timecourse_opt_module
    return loss_timecourse, (prev_iter_spatial_filter, timecourse_w, coupling_w, feedback_w, bias)


class NS_WNReg_SpatFitGLM(nn.Module):
    '''
    New-style module for learning GLM + spatial filters for
        flashed natural scenes data, regularized with a little
        bit of white noise

    Optimized using the new-style optimizer that subclasses
        torch.optim.Optimizer, so this is compatible with amp
        mixed precision optimization for smaller memory footprint
        and faster training
    '''

    def __init__(self,
                 dim_stimulus: int,
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 wn_regularizer_weight: float,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 spat_filter_sparsity_lambda: float = 0.0,
                 group_sparse_reg_lambda: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 stim_spat_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_feedback_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_bias: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_norm: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 debug_wn_only: bool = False):

        super().__init__()

        self.dim_stimulus = dim_stimulus

        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.wn_regularizer_weight = wn_regularizer_weight
        self.loss_callable = loss_callable
        self.spatial_sparsity_l1_lambda = spat_filter_sparsity_lambda
        self.group_sparse_reg_lambda = group_sparse_reg_lambda

        self.debug_wn_only = debug_wn_only

        # OPT VARIABLE 0: coupling_w, shape (n_coupled_cells, n_basis_coupling)
        if init_coupling_w is not None:
            if init_coupling_w.shape != (n_coupled_cells, n_basis_coupling):
                raise ValueError(f"init_coupling_w must have shape {(n_coupled_cells, n_basis_coupling)}")
            if isinstance(init_coupling_w, np.ndarray):
                self.coupling_w = nn.Parameter(torch.tensor(init_coupling_w, dtype=dtype), requires_grad=True)
            else:
                self.coupling_w = nn.Parameter(init_coupling_w.detach().clone(), requires_grad=True)
        else:
            self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1, feedback_w, shape (1, n_basis_feedback)
        if init_feedback_w is not None:
            if init_feedback_w.shape != (1, n_basis_feedback):
                raise ValueError(f"init_feedback_w must have shape {(1, n_basis_feedback)}")
            if isinstance(init_feedback_w, np.ndarray):
                self.feedback_w = nn.Parameter(torch.tensor(init_feedback_w, dtype=dtype), requires_grad=True)
            else:
                self.feedback_w = nn.Parameter(init_feedback_w.detach().clone(), requires_grad=True)
        else:
            self.feedback_w = nn.Parameter(torch.empty((1, n_basis_feedback), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.feedback_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 2, shape (1, n_basis_stim_spat)
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

        # OPTIMIZATION VARIABLE 3, bias, shape (1, )
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

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )
        if init_coupling_norm is not None:
            if init_coupling_norm.shape != (n_coupled_cells,):
                raise ValueError(f"init_coupling_norm must have shape {(n_coupled_cells,)}")
            if isinstance(init_coupling_norm, np.ndarray):
                self.coupling_filter_norm = nn.Parameter(torch.tensor(init_coupling_norm, dtype=dtype),
                                                         requires_grad=True)
            else:
                self.coupling_filter_norm = nn.Parameter(init_coupling_norm.detach().clone(),
                                                         requires_grad=True)
        else:
            self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                     requires_grad=True)
            nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   coupling_filt_w: torch.Tensor,
                   feedback_filt_w: torch.Tensor,
                   spat_filt_w: torch.Tensor,
                   bias: torch.Tensor,
                   coupling_aux: torch.Tensor,
                   nscenes_spat_filt_frame: torch.Tensor,
                   nscenes_time_filt: torch.Tensor,
                   nscenes_cell_spikes: torch.Tensor,
                   nscenes_filt_feedback: torch.Tensor,
                   nscenes_filt_coupling: torch.Tensor,
                   wn_filt_movie: torch.Tensor,
                   wn_cell_spikes: torch.Tensor,
                   wn_filt_feedback: torch.Tensor,
                   wn_filt_coupling: torch.Tensor) -> torch.Tensor:
        '''

        :param coupling_filt_w: shape (n_coupled_cells, n_basis_coupling)
        :param feedback_filt_w: shape (1, n_basis_feedback)
        :param spat_filt_w: shape (1, n_basis_stim_spat)
        :param bias: shape (1, )
        :param coupling_aux: shape (n_coupled_cells, )
        :param nscenes_spat_filt_frame: shape (batch, dim_stim)
        :param nscenes_time_filt: shape (n_bins - n_bins_filter + 1, )
        :param nscenes_cell_spikes: shape (batch, n_bins - n_bins_filter + 1)
        :param nscenes_filt_feedback: shape (batch, n_basis_feedback, n_bins - n_bins_filter + 1)
        :param nscenes_filt_coupling: shape (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
        :param wn_filt_movie: shape (dim_stim, n_bins_wn - n_bins_filter + 1)
        :param wn_cell_spikes: shape (n_bins_wn - n_bins_filter + 1, )
        :param wn_filt_feedback: shape (1, n_basis_feedback, n_bins_wn - n_bins_filter + 1)
        :param wn_filt_coupling: shape (n_coupled_cells, n_basis_coupling, n_bins_wn - n_bins_filter + 1)
        :return:
        '''

        ns_stimulus_contrib = _flashed_spatial_stimulus_contrib_gensig(spat_filt_w, nscenes_spat_filt_frame,
                                                                       nscenes_time_filt)

        ns_feedback_contrib = _flashed_feedback_contrib_gensig(feedback_filt_w,
                                                               nscenes_filt_feedback)
        ns_coupling_contrib = _flashed_coupling_contrib_gensig(coupling_filt_w,
                                                               nscenes_filt_coupling)

        ns_gen_sig = bias + ns_stimulus_contrib + ns_feedback_contrib + ns_coupling_contrib

        nscenes_spiking_loss = self.loss_callable(ns_gen_sig[:, :-1],
                                                  nscenes_cell_spikes[:, 1:])

        wn_stimulus_contrib = _movie_spatial_stimulus_contrib_gensig(spat_filt_w, wn_filt_movie)
        wn_feedback_coupling_contrib = _movie_coupling_feedback_contrib_gensig(coupling_filt_w,
                                                                               feedback_filt_w,
                                                                               wn_filt_coupling,
                                                                               wn_filt_feedback)

        wn_gen_sig = bias + wn_stimulus_contrib + wn_feedback_coupling_contrib
        wn_spiking_loss = self.loss_callable(wn_gen_sig[:-1],
                                             wn_cell_spikes[1:])

        spiking_loss = nscenes_spiking_loss + self.wn_regularizer_weight * wn_spiking_loss

        if self.group_sparse_reg_lambda != 0.0:
            regularization_penalty = self.group_sparse_reg_lambda * torch.sum(coupling_aux)
            return spiking_loss + regularization_penalty
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
                               coupling_filter_w: torch.Tensor,
                               feedback_filter_w: torch.Tensor,
                               spat_filter_w: torch.Tensor,
                               bias_w: torch.Tensor,
                               coupling_auxvar: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        IMPORTANT: the order of the parameters for this method MUST match
            the order of the nn.Parameters declared in the constructor
            otherwise the FISTA implementation will fail.

        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        :return:
        '''

        return _spatial_model_prox_project_variables(coupling_filter_w, feedback_filter_w,
                                                     spat_filter_w, bias_w, coupling_auxvar,
                                                     self.spatial_sparsity_l1_lambda)

    def return_spat_filt_parameters_np(self) -> np.ndarray:
        return self.stim_spat_w.detach().cpu().numpy()

    def return_spat_filt_parameters(self) -> torch.Tensor:
        return self.stim_spat_w.detach().clone()

    def set_optimization_parameters(self,
                                    stim_spat_w: Optional[torch.Tensor] = None,
                                    coupling_w: Optional[torch.Tensor] = None,
                                    feedback_w: Optional[torch.Tensor] = None,
                                    bias: Optional[torch.Tensor] = None,
                                    coupling_filt_norm: Optional[torch.Tensor] = None):
        if stim_spat_w is not None:
            self.stim_spat_w.data[:] = stim_spat_w.data[:]
        if coupling_w is not None:
            self.coupling_w.data[:] = coupling_w.data[:]
        if feedback_w is not None:
            self.feedback_w.data[:] = feedback_w.data[:]
        if bias is not None:
            self.bias.data[:] = bias.data[:]
        if coupling_filt_norm is not None:
            self.coupling_filter_norm.data[:] = coupling_filt_norm.data[:]

    def return_parameters_torch(self) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            self.coupling_w.detach().clone(),
            self.feedback_w.detach().clone(),
            self.stim_spat_w.detach().clone(),
            self.bias.detach().clone(),
            self.coupling_filter_norm.detach().clone()
        )

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            self.stim_spat_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
            self.coupling_filter_norm.detach().cpu().numpy()
        )


class NS_WNReg_TimecourseFitGLM(nn.Module):

    def __init__(self,
                 n_basis_stim_time: int,
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 wn_regularizer_weight: float,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 group_sparse_reg_lambda: float = 0.0,
                 dtype: torch.dtype = torch.float32,
                 stim_time_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_feedback_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_bias: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 init_coupling_norm: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 debug_wn_only: bool = False):

        super().__init__()

        self.n_basis_stim_time = n_basis_stim_time
        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.wn_regularizer_weight = wn_regularizer_weight

        self.loss_callable = loss_callable

        self.group_sparse_reg_lambda = group_sparse_reg_lambda

        self.debug_wn_only = debug_wn_only

        # OPT VARIABLE 0: coupling_w, shape (n_coupled_cells, n_basis_coupling)
        if init_coupling_w is not None:
            if init_coupling_w.shape != (n_coupled_cells, n_basis_coupling):
                raise ValueError(f"init_coupling_w must have shape {(n_coupled_cells, n_basis_coupling)}")
            if isinstance(init_coupling_w, np.ndarray):
                self.coupling_w = nn.Parameter(torch.tensor(init_coupling_w, dtype=dtype), requires_grad=True)
            else:
                self.coupling_w = nn.Parameter(init_coupling_w.detach().clone(), requires_grad=True)
        else:
            self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1, feedback_w, shape (1, n_basis_feedback)
        if init_feedback_w is not None:
            if init_feedback_w.shape != (1, n_basis_feedback):
                raise ValueError(f"init_feedback_w must have shape {(1, n_basis_feedback)}")
            if isinstance(init_feedback_w, np.ndarray):
                self.feedback_w = nn.Parameter(torch.tensor(init_feedback_w, dtype=dtype), requires_grad=True)
            else:
                self.feedback_w = nn.Parameter(init_feedback_w.detach().clone(), requires_grad=True)
        else:
            self.feedback_w = nn.Parameter(torch.empty((1, n_basis_feedback), dtype=dtype),
                                           requires_grad=True)
            nn.init.uniform_(self.feedback_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 2, shape (1, n_basis_stim_time)
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

        # OPTIMIZATION VARIABLE 3, bias, shape (1, )
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

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )
        if init_coupling_norm is not None:
            if init_coupling_norm.shape != (n_coupled_cells,):
                raise ValueError(f"init_coupling_norm must have shape {(n_coupled_cells,)}")
            if isinstance(init_coupling_norm, np.ndarray):
                self.coupling_filter_norm = nn.Parameter(torch.tensor(init_coupling_norm, dtype=dtype),
                                                         requires_grad=True)
            else:
                self.coupling_filter_norm = nn.Parameter(init_coupling_norm.detach().clone(),
                                                         requires_grad=True)
        else:
            self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                     requires_grad=True)
            nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   coupling_filt_w: torch.Tensor,
                   feedback_filt_w: torch.Tensor,
                   time_filt_w: torch.Tensor,
                   bias: torch.Tensor,
                   coupling_aux: torch.Tensor,

                   ns_spat_filt_movie: torch.Tensor,
                   ns_time_basis_filt_movie: torch.Tensor,
                   ns_binned_spikes_cell: torch.Tensor,
                   ns_filtered_feedback: torch.Tensor,
                   ns_filtered_coupling: torch.Tensor,

                   wn_time_basis_filt_movie: torch.Tensor,
                   wn_binned_spikes_cell: torch.Tensor,
                   wn_filtered_feedback: torch.Tensor,
                   wn_filtered_coupling: torch.Tensor) -> torch.Tensor:
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

        ns_feedback_contrib = _flashed_feedback_contrib_gensig(feedback_filt_w,
                                                               ns_filtered_feedback)
        ns_coupling_contrib = _flashed_coupling_contrib_gensig(coupling_filt_w,
                                                               ns_filtered_coupling)

        ns_gen_sig = bias + ns_stimulus_contrib + ns_feedback_contrib + ns_coupling_contrib
        ns_spiking_loss = self.loss_callable(ns_gen_sig[:, :-1],
                                             ns_binned_spikes_cell[:, 1:])

        wn_stimulus_contrib = _movie_timecourse_stimulus_contrib_gensig(time_filt_w,
                                                                        wn_time_basis_filt_movie)
        wn_feedback_coupling_contrib = _movie_coupling_feedback_contrib_gensig(coupling_filt_w,
                                                                               feedback_filt_w,
                                                                               wn_filtered_coupling,
                                                                               wn_filtered_feedback)

        wn_gen_sig = bias + wn_stimulus_contrib + wn_feedback_coupling_contrib
        wn_spiking_loss = self.loss_callable(wn_gen_sig[:-1], wn_binned_spikes_cell[1:])

        spiking_loss = ns_spiking_loss + self.wn_regularizer_weight * wn_spiking_loss

        if self.group_sparse_reg_lambda != 0.0:
            regularization_penalty = self.group_sparse_reg_lambda * torch.sum(coupling_aux)
            return spiking_loss + regularization_penalty

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
                               coupling_filter_w: torch.Tensor,
                               feedback_filter_w: torch.Tensor,
                               timecourse_w: torch.Tensor,
                               bias_w: torch.Tensor,
                               coupling_auxvar: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        '''
        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        IMPORTANT: the parameters here need to be in the same order as the order
            of the nn.Parameter declared in the constructor, otherwise the FISTA
            algorithm isn't going to be able to successfully apply the projection
        :param args:
        :param kwargs:
        :return:
        '''

        return _time_model_prox_project_variables(coupling_filter_w, feedback_filter_w,
                                                  timecourse_w, bias_w, coupling_auxvar)

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
                                    coupling_w: Optional[torch.Tensor] = None,
                                    feedback_w: Optional[torch.Tensor] = None,
                                    bias: Optional[torch.Tensor] = None,
                                    coupling_filt_norm: Optional[torch.Tensor] = None):
        if timecourse_w is not None:
            self.stim_time_w.data[:] = timecourse_w.data[:]
        if coupling_w is not None:
            self.coupling_w.data[:] = coupling_w.data[:]
        if feedback_w is not None:
            self.feedback_w.data[:] = feedback_w.data[:]
        if bias is not None:
            self.bias.data[:] = bias.data[:]
        if coupling_filt_norm is not None:
            self.coupling_filter_norm.data[:] = coupling_filt_norm.data[:]

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            self.stim_time_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy(),
            self.coupling_filter_norm.detach().cpu().numpy()
        )

    def return_parameters_torch(self) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            self.coupling_w.detach().clone(),
            self.feedback_w.detach().clone(),
            self.stim_time_w.detach().clone(),
            self.bias.detach().clone(),
            self.coupling_filter_norm.detach().clone()
        )


def new_style_joint_wn_flashed_ns_alternating_optim(
        ns_stimulus_frames: torch.Tensor,
        ns_spikes_cell: torch.Tensor,
        ns_stim_time_basis_conv: torch.Tensor,
        ns_feedback_basis_conv: torch.Tensor,
        ns_coupling_basis_conv: torch.Tensor,
        wn_stimulus_frames: torch.Tensor,
        wn_spikes_cell: torch.Tensor,
        wn_feedback_basis_conv: torch.Tensor,
        wn_coupling_basis_conv: torch.Tensor,
        stim_time_basis: torch.Tensor,
        loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        solver_params_iter: Tuple[Iterator[ProxSolverParams], Iterator[ProxSolverParams]],
        n_iters_outer_opt: int,
        device: torch.device,
        weight_wn: float = 0.1,
        l21_group_sparse_lambda: float = 0.0,
        l1_spat_sparse_lambda: float = 0.0,
        initial_guess_timecourse: Optional[torch.Tensor] = None,
        initial_guess_coupling: Optional[torch.Tensor] = None,
        initial_guess_coupling_norm: Optional[torch.Tensor] = None,
        initial_guess_feedback: Optional[torch.Tensor] = None,
        initial_guess_bias: Optional[torch.Tensor] = None,
        outer_opt_verbose: bool = False,
        debug_wn_only: bool = False):
    '''
    Function for performing alternating optimization between

    :param ns_stimulus_frames: raw stimulus frames (either with or without pre-application of basis filter),
        shape (batch, dim_spat)
    :param ns_spikes_cell: binned spikes for the cells being fit, shape (batch, n_bins - n_bins_filter + 1)
        if fitting strictly causal model, (batch, n_bins - n_bins_filter) if fitting non-strictly causal model
    :param ns_stim_time_basis_conv: shape (n_basis_time, n_bins - n_bins_filter + 1)
    :param ns_feedback_basis_conv: shape (batch, n_basis_feedback, n_bins - n_bins_filter + 1)
    :param ns_coupling_basis_conv: shape (batch, n_coupled_cells, n_bins_total - n_bins_filter + 1)
    :param wn_stimulus_frames: shape (dim_spat, n_bins_wn, )
    :param wn_spikes_cell: shape (n_bins_wn - n_bins_filter + 1, ) if fitting strictly causal model,
    :param wn_feedback_basis_conv: shape (1, n_basis_feedback, n_bins_wn - n_bins_filter + 1)
    :param wn_coupling_basis_conv: shape (n_coupled_cells, n_basis_coupling, n_bins_wn - n_bins_filter + 1)
    :param stim_time_basis: shape (n_basis_time, n_bins_filter)
    :param loss_callable:
    :param solver_params_iter:
    :param n_iters_outer_opt:
    :param device:
    :param l21_group_sparse_lambda:
    :param l1_spat_sparse_lambda:
    :param initial_guess_timecourse:
    :param initial_guess_coupling:
    :param initial_guess_coupling_norm:
    :param initial_guess_feedback:
    :param initial_guess_bias:
    :param inner_opt_verbose:
    :param outer_opt_verbose:
    :return:
    '''

    n_flashed_trials, dim_spat = ns_stimulus_frames.shape
    assert wn_stimulus_frames.shape[0] == dim_spat, 'wn dimension and nscenes dimension must match'

    n_basis_stim_time, _ = ns_stim_time_basis_conv.shape
    n_basis_feedback = ns_feedback_basis_conv.shape[1]
    n_coupled_cells, n_basis_coupling = ns_coupling_basis_conv.shape[1], ns_coupling_basis_conv.shape[2]

    # shape (n_timecourse_basis, )
    timecourse_w = initial_guess_timecourse

    coupling_w, feedback_w = initial_guess_coupling, initial_guess_feedback
    bias, coupling_filt_norm = initial_guess_bias, initial_guess_coupling_norm

    prev_iter_spatial_filter = None

    with torch.no_grad(), autocast('cuda'):

        # shape (1, n_timecourse_basis) @ (n_timecourse_basis, n_bins_filter)
        # -> (1, n_bins_filter) -> (n_bins_filter, )
        timecourse_filter = (timecourse_w[None, :] @ stim_time_basis).squeeze(0)
        max_timecourse = torch.linalg.norm(timecourse_filter)

        timecourse_w = timecourse_w.div_(max_timecourse)
        timecourse_filter = timecourse_filter.div_(max_timecourse)

    spat_opt_module = NS_WNReg_SpatFitGLM(
        dim_spat,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        weight_wn,
        loss_callable,
        spat_filter_sparsity_lambda=l1_spat_sparse_lambda,
        group_sparse_reg_lambda=l21_group_sparse_lambda,
        init_feedback_w=initial_guess_feedback,
        init_coupling_w=initial_guess_coupling,
        init_coupling_norm=initial_guess_coupling_norm,
        init_bias=initial_guess_bias,
        debug_wn_only=debug_wn_only
    ).to(device)

    timecourse_opt_module = NS_WNReg_TimecourseFitGLM(
        n_basis_stim_time,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        weight_wn,
        loss_callable,
        group_sparse_reg_lambda=l21_group_sparse_lambda,
        stim_time_init_guess=timecourse_w,
        init_feedback_w=feedback_w,
        init_coupling_w=coupling_w,
        init_bias=bias,
        init_coupling_norm=coupling_filt_norm,
        debug_wn_only=debug_wn_only
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
            coupling_w=coupling_w,
            feedback_w=feedback_w,
            bias=bias,
            coupling_filt_norm=coupling_filt_norm
        )

        loss_spatial = _optim_FISTA(
            spat_opt_module,
            spat_solver_params,
            ns_stimulus_frames,
            ns_filtered_stim_time,
            ns_spikes_cell,
            ns_feedback_basis_conv,
            ns_coupling_basis_conv,
            wn_filtered_stim_time,
            wn_spikes_cell,
            wn_feedback_basis_conv,
            wn_coupling_basis_conv,
        )

        del ns_filtered_stim_time, wn_filtered_stim_time

        coupling_w, feedback_w, prev_iter_spatial_filter, bias, coupling_filt_norm = spat_opt_module.return_parameters_torch()

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
            coupling_w=coupling_w,
            feedback_w=feedback_w,
            bias=bias,
            coupling_filt_norm=coupling_filt_norm
        )

        loss_timecourse = _optim_FISTA(
            timecourse_opt_module,
            time_solver_params,
            ns_spatial_filter_applied,
            ns_stim_time_basis_conv,
            ns_spikes_cell,
            ns_feedback_basis_conv,
            ns_coupling_basis_conv,
            wn_spatial_filter_time_basis_applied,
            wn_spikes_cell,
            wn_feedback_basis_conv,
            wn_coupling_basis_conv,
        )

        del ns_spatial_filter_applied, wn_spatial_filter_applied, wn_spatial_filter_time_basis_applied

        coupling_w, feedback_w, timecourse_w, bias, coupling_filt_norm = timecourse_opt_module.return_parameters_torch()

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
    return loss_timecourse, (prev_iter_spatial_filter, timecourse_w, coupling_w, feedback_w, bias)
