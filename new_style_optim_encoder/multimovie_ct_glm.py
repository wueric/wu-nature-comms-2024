import numpy as np

import torch
import torch.nn as nn

from typing import Callable, Optional, Union, List, Tuple

from new_style_optim_encoder.glm_components import _movie_spatial_stimulus_contrib_gensig, \
    _multimovie_spatial_stimulus_contrib_gensig, _movie_timecourse_stimulus_contrib_gensig, \
    _multimovie_timecourse_stimulus_contrib_gensig, _movie_coupling_feedback_contrib_gensig, \
    _multimovie_coupling_feedback_contrib_gensig, _multimovie_feedback_contrib_gensig
from new_style_optim_encoder.glm_prox import _time_model_prox_project_variables, _spatial_model_prox_project_variables, \
    _feedback_only_spatial_model_prox_project_variables


class NS_Timecourse_GroupSparseLRCroppedCT_GLM(nn.Module):
    '''
    Non-multimovie version of the continuous time GLM

    This one uses gradient accumulation to do the optimization over
        multiple chunks of movies, rather than compute the gradient all
        at once. This should save a substantial amount of GPU memory
        and will make it more scalable on AWS
    '''

    def __init__(self,
                 n_basis_stim_time: int,
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 group_sparse_reg_lambda: float = 0.0,
                 stim_time_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.n_basis_stim_time = n_basis_stim_time
        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.loss_callable = loss_callable
        self.group_sparse_reg_lambda = group_sparse_reg_lambda

        # OPTIMIZATION VARIABLE 0
        self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                       requires_grad=True)
        nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1
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

        # OPTIMIZATION VARIABLE 3
        self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
        nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )
        self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                 requires_grad=True)
        nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   coupling_w: torch.Tensor,
                   feedback_w: torch.Tensor,
                   timecourse_w: torch.Tensor,
                   bias: torch.Tensor,
                   coupling_aux: torch.Tensor,
                   filt_movie: torch.Tensor,
                   binned_spikes: torch.Tensor,
                   filt_feedback: torch.Tensor,
                   filt_coupling: torch.Tensor) -> torch.Tensor:
        '''

        :param coupling_w:
        :param feedback_w:
        :param timecourse_w:
        :param bias:
        :param coupling_aux:
        :param filt_multimovie:
        :param cell_spike_multibin:
        :param filt_feedback_multibins:
        :param filt_coupling_multibins:
        :return:
        '''

        stimulus_contrib = _movie_timecourse_stimulus_contrib_gensig(timecourse_w, filt_movie)
        feedback_coupling_contrib = _movie_coupling_feedback_contrib_gensig(
            coupling_w, feedback_w, filt_coupling, filt_feedback)

        gen_sig = stimulus_contrib + feedback_coupling_contrib + bias
        spiking_loss = self.loss_callable(gen_sig[:-1], binned_spikes[1:])

        if self.group_sparse_reg_lambda != 0.0:  # FIXME be careful with how we weight the regularization term
            # with the spiking loss term when we do gradient accumulation
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
                               coupling_w: torch.Tensor,
                               feedback_w: torch.Tensor,
                               timecourse_w: torch.Tensor,
                               bias: torch.Tensor,
                               coupling_aux: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return _time_model_prox_project_variables(coupling_w,
                                                  feedback_w,
                                                  timecourse_w,
                                                  bias,
                                                  coupling_aux)

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
        self.coupling_w.data[:] = coord_desc_other.coupling_w.data[:]
        self.feedback_w.data[:] = coord_desc_other.feedback_w.data[:]
        self.bias.data[:] = coord_desc_other.bias.data[:]
        self.coupling_filter_norm.data[:] = coord_desc_other.coupling_filter_norm.data[:]

        self.zero_grad()

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            self.stim_time_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy()
        )


class NS_Spatial_GroupSparseLRCroppedCT_GLM(nn.Module):
    '''
    Non-multimovie version of the continuous time GLM

    This one uses gradient accumulation to do the optimization over
        multiple chunks of movies, rather than compute the gradient all
        at once. This should save a substantial amount of GPU memory
        and will make it more scalable on AWS
    '''

    def __init__(self,
                 n_basis_stim_spat: int,
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 group_sparse_reg_lambda: float = 0.0,
                 spatial_sparsity_l1_lambda: float = 0.0,
                 stim_spat_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 dtype: torch.dtype = torch.float32):
        '''

        :param spt_lin_filt_rank:
        :param stim_spat_basis:  shape (n_basis_stim_spat, n_pix)
        :param stim_time_basis: shape (n_basis_stim_time, n_bins_filter)
        :param n_basis_feedback:
        :param n_basis_coupling:
        :param n_coupled_cells:
        :param loss_fn:
        :param dtype:
        '''
        super().__init__()

        self.n_basis_stim_spat = n_basis_stim_spat
        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.loss_callable = loss_callable
        self.group_sparse_reg_lambda = group_sparse_reg_lambda
        self.spatial_sparsity_l1_lambda = spatial_sparsity_l1_lambda

        # OPT VARIABLE 0
        self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                       requires_grad=True)
        nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1
        self.feedback_w = nn.Parameter(torch.empty((1, n_basis_feedback), dtype=dtype),
                                       requires_grad=True)
        nn.init.uniform_(self.feedback_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 2, shape (1, n_basis_stim_spat)
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

        # OPTIMIZATION VARIABLE 3
        self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
        nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )
        self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                 requires_grad=True)
        nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   coupling_w: torch.Tensor,
                   feedback_w: torch.Tensor,
                   spat_filt_w: torch.Tensor,
                   bias: torch.Tensor,
                   coupling_aux: torch.Tensor,
                   filt_movie: torch.Tensor,
                   cell_spikes: torch.Tensor,
                   filt_feedback: torch.Tensor,
                   filt_coupling: torch.Tensor) -> torch.Tensor:

        stimulus_contrib = _movie_spatial_stimulus_contrib_gensig(spat_filt_w, filt_movie)
        feedback_coupling_contrib = _movie_coupling_feedback_contrib_gensig(coupling_w, feedback_w,
                                                                            filt_coupling, filt_feedback)
        gen_sig = stimulus_contrib + feedback_coupling_contrib + bias
        spiking_loss = self.loss_callable(gen_sig[:-1], cell_spikes[1:])

        if self.group_sparse_reg_lambda != 0.0:  # FIXME be careful with how the regularization term
            # is weighted when we do gradient accumulation
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
                               coupling_auxvar: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        '''
        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        Everything else we can just pass through

        :param args:
        :param kwargs:
        :return:
        '''

        return _spatial_model_prox_project_variables(coupling_filter_w,
                                                     feedback_filter_w,
                                                     spat_filter_w,
                                                     bias_w,
                                                     coupling_auxvar,
                                                     self.spatial_sparsity_l1_lambda)

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
        self.coupling_w.data[:] = coord_desc_other.coupling_w.data[:]
        self.feedback_w.data[:] = coord_desc_other.feedback_w.data[:]
        self.bias.data[:] = coord_desc_other.bias.data[:]
        self.coupling_filter_norm.data[:] = coord_desc_other.coupling_filter_norm.data[:]

        self.zero_grad()

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            self.stim_spat_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy()
        )


class NS_MM_Timecourse_GroupSparseLRCroppedCT_GLM(nn.Module):

    def __init__(self,
                 n_basis_stim_time: int,
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 group_sparse_reg_lambda: float = 0.0,
                 stim_time_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 multimovie_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.n_basis_stim_time = n_basis_stim_time
        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.loss_callable = loss_callable
        self.group_sparse_reg_lambda = group_sparse_reg_lambda

        self.use_multimovie_weights = False
        if multimovie_weights is not None:
            if isinstance(multimovie_weights, np.ndarray):
                self.register_buffer('multimovie_weights', torch.tensor(multimovie_weights, dtype=dtype))
            else:
                self.register_buffer('multimovie_weights', multimovie_weights.detach().clone())

        # OPT VARIABLE 0
        self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                       requires_grad=True)
        nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1
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

        # OPTIMIZATION VARIABLE 3
        self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
        nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )
        self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                 requires_grad=True)
        nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   coupling_w: torch.Tensor,
                   feedback_w: torch.Tensor,
                   timecourse_w: torch.Tensor,
                   bias: torch.Tensor,
                   coupling_aux: torch.Tensor,
                   filt_multimovie: List[torch.Tensor],
                   cell_spike_multibin: List[torch.Tensor],
                   filt_feedback_multibins: List[torch.Tensor],
                   filt_coupling_multibins: List[torch.Tensor]) -> torch.Tensor:
        '''

        :param coupling_w:
        :param feedback_w:
        :param timecourse_w:
        :param bias:
        :param coupling_aux:
        :param filt_multimovie:
        :param cell_spike_multibin:
        :param filt_feedback_multibins:
        :param filt_coupling_multibins:
        :return:
        '''

        stimulus_contrib = _multimovie_timecourse_stimulus_contrib_gensig(timecourse_w, filt_multimovie)
        feedback_coupling_contrib = _multimovie_coupling_feedback_contrib_gensig(
            coupling_w, feedback_w, filt_coupling_multibins, filt_feedback_multibins)
        gen_sig_list = [bias + x + y for x, y in zip(stimulus_contrib, feedback_coupling_contrib)]

        spiking_loss_per_movie = [self.loss_callable(gen_sig[:-1], binned_spikes[1:])
                                  for gen_sig, binned_spikes in zip(gen_sig_list, cell_spike_multibin)]

        if self.use_multimovie_weights:
            stacked_losses = torch.stack(spiking_loss_per_movie, dim=0)
            spiking_loss = torch.sum(self.multimovie_weights * stacked_losses)
        else:
            spiking_loss = torch.mean(torch.stack(spiking_loss_per_movie, dim=0))

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
                               coupling_w: torch.Tensor,
                               feedback_w: torch.Tensor,
                               timecourse_w: torch.Tensor,
                               bias: torch.Tensor,
                               coupling_aux: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        return _time_model_prox_project_variables(coupling_w, feedback_w, timecourse_w,
                                                  bias, coupling_aux)

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
        self.coupling_w.data[:] = coord_desc_other.coupling_w.data[:]
        self.feedback_w.data[:] = coord_desc_other.feedback_w.data[:]
        self.bias.data[:] = coord_desc_other.bias.data[:]
        self.coupling_filter_norm.data[:] = coord_desc_other.coupling_filter_norm.data[:]

        self.zero_grad()

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            self.stim_time_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy()
        )


class NS_MM_Spatial_GroupSparseLRCroppedCT_GLM(nn.Module):

    def __init__(self,
                 n_basis_stim_spat: int,
                 n_basis_feedback: int,
                 n_basis_coupling: int,
                 n_coupled_cells: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 group_sparse_reg_lambda: float = 0.0,
                 spatial_sparsity_l1_lambda: float = 0.0,
                 stim_spat_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 multimovie_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 dtype: torch.dtype = torch.float32):
        '''

        :param spt_lin_filt_rank:
        :param stim_spat_basis:  shape (n_basis_stim_spat, n_pix)
        :param stim_time_basis: shape (n_basis_stim_time, n_bins_filter)
        :param n_basis_feedback:
        :param n_basis_coupling:
        :param n_coupled_cells:
        :param loss_fn:
        :param dtype:
        '''
        super().__init__()

        self.n_basis_stim_spat = n_basis_stim_spat
        self.n_basis_feedback = n_basis_feedback
        self.n_basis_coupling = n_basis_coupling
        self.n_coupled_cells = n_coupled_cells

        self.loss_callable = loss_callable
        self.group_sparse_reg_lambda = group_sparse_reg_lambda
        self.spatial_sparsity_l1_lambda = spatial_sparsity_l1_lambda

        self.use_multimovie_weights = False
        if multimovie_weights is not None:
            if isinstance(multimovie_weights, np.ndarray):
                self.register_buffer('multimovie_weights', torch.tensor(multimovie_weights, dtype=dtype))
            else:
                self.register_buffer('multimovie_weights', multimovie_weights.detach().clone())

        # OPT VARIABLE 0
        self.coupling_w = nn.Parameter(torch.empty((n_coupled_cells, n_basis_coupling), dtype=dtype),
                                       requires_grad=True)
        nn.init.uniform_(self.coupling_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 1
        self.feedback_w = nn.Parameter(torch.empty((1, n_basis_feedback), dtype=dtype),
                                       requires_grad=True)
        nn.init.uniform_(self.feedback_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 2, shape (1, n_basis_stim_spat)
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

        # OPTIMIZATION VARIABLE 3
        self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
        nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 4 (this one is for the coupling filter group sparsity penalty)
        # shape (n_coupled_cells, )
        self.coupling_filter_norm = nn.Parameter(torch.empty((self.coupling_w.shape[0],), dtype=dtype),
                                                 requires_grad=True)
        nn.init.uniform_(self.coupling_filter_norm, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   coupling_w: torch.Tensor,
                   feedback_w: torch.Tensor,
                   spat_filt_w: torch.Tensor,
                   bias: torch.Tensor,
                   coupling_aux: torch.Tensor,
                   filt_multimovie: List[torch.Tensor],
                   cell_spike_multibin: List[torch.Tensor],
                   filt_feedback_multibins: List[torch.Tensor],
                   filt_coupling_multibins: List[torch.Tensor]) -> torch.Tensor:

        stimulus_contrib = _multimovie_spatial_stimulus_contrib_gensig(spat_filt_w, filt_multimovie)
        feedback_coupling_contrib = _multimovie_coupling_feedback_contrib_gensig(coupling_w, feedback_w,
                                                                                 filt_coupling_multibins,
                                                                                 filt_feedback_multibins)
        gen_sig_list = [bias + x + y for x, y in zip(stimulus_contrib, feedback_coupling_contrib)]

        spiking_loss_per_movie = [self.loss_callable(gen_sig[:-1], binned_spikes[1:])
                                  for gen_sig, binned_spikes in zip(gen_sig_list, cell_spike_multibin)]

        if self.use_multimovie_weights:
            stacked_losses = torch.stack(spiking_loss_per_movie, dim=0)
            spiking_loss = torch.sum(self.multimovie_weights * stacked_losses)
        else:
            spiking_loss = torch.mean(torch.stack(spiking_loss_per_movie, dim=0))

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
                               coupling_auxvar: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        '''
        The only not-smooth part of the penalty that we need to project is the
            neighboring cell coupling filter coefficients, and the neighboring cell
            coupling norm auxiliary variables.

        Everything else we can just pass through

        :param args:
        :param kwargs:
        :return:
        '''

        return _spatial_model_prox_project_variables(coupling_filter_w, feedback_filter_w, spat_filter_w,
                                                     bias_w, coupling_auxvar, self.spatial_sparsity_l1_lambda)

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
        self.coupling_w.data[:] = coord_desc_other.coupling_w.data[:]
        self.feedback_w.data[:] = coord_desc_other.feedback_w.data[:]
        self.bias.data[:] = coord_desc_other.bias.data[:]
        self.coupling_filter_norm.data[:] = coord_desc_other.coupling_filter_norm.data[:]

        self.zero_grad()

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.coupling_w.detach().cpu().numpy(),
            self.feedback_w.detach().cpu().numpy(),
            self.stim_spat_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy()
        )


class NS_MM_Timecourse_FB_Only_GLM(nn.Module):

    def __init__(self,
                 n_basis_stim_time: int,
                 n_basis_feedback: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 stim_time_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 multimovie_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.n_basis_stim_time = n_basis_stim_time
        self.n_basis_feedback = n_basis_feedback

        self.loss_callable = loss_callable

        self.use_multimovie_weights = False
        if multimovie_weights is not None:
            if isinstance(multimovie_weights, np.ndarray):
                self.register_buffer('multimovie_weights', torch.tensor(multimovie_weights, dtype=dtype))
            else:
                self.register_buffer('multimovie_weights', multimovie_weights.detach().clone())

        # OPTIMIZATION VARIABLE 1
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

        # OPTIMIZATION VARIABLE 3
        self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
        nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   feedback_w: torch.Tensor,
                   timecourse_w: torch.Tensor,
                   bias: torch.Tensor,
                   filt_multimovie: List[torch.Tensor],
                   cell_spike_multibin: List[torch.Tensor],
                   filt_feedback_multibins: List[torch.Tensor]) -> torch.Tensor:
        '''

        :param coupling_w:
        :param feedback_w:
        :param timecourse_w:
        :param bias:
        :param coupling_aux:
        :param filt_multimovie:
        :param cell_spike_multibin:
        :param filt_feedback_multibins:
        :param filt_coupling_multibins:
        :return:
        '''

        stimulus_contrib = _multimovie_timecourse_stimulus_contrib_gensig(timecourse_w, filt_multimovie)
        feedback_contrib_gensig = _multimovie_feedback_contrib_gensig(
            feedback_w, filt_feedback_multibins)
        gen_sig_list = [bias + x + y for x, y in zip(stimulus_contrib, feedback_contrib_gensig)]

        spiking_loss_per_movie = [self.loss_callable(gen_sig[:-1], binned_spikes[1:])
                                  for gen_sig, binned_spikes in zip(gen_sig_list, cell_spike_multibin)]

        if self.use_multimovie_weights:
            stacked_losses = torch.stack(spiking_loss_per_movie, dim=0)
            spiking_loss = torch.sum(self.multimovie_weights * stacked_losses)
        else:
            spiking_loss = torch.mean(torch.stack(spiking_loss_per_movie, dim=0))

        return spiking_loss

    def forward(self, *data_args) -> torch.Tensor:
        return self._eval_loss(*self.parameters(recurse=False),
                               *data_args)

    def make_loss_eval_callable(self, *data_args) \
            -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], float]:
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

    def clone_parameters_model(self, coord_desc_other) -> None:
        self.feedback_w.data[:] = coord_desc_other.feedback_w.data[:]
        self.bias.data[:] = coord_desc_other.bias.data[:]

        self.zero_grad()

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        return (
            self.feedback_w.detach().cpu().numpy(),
            self.stim_time_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy()
        )


class NS_MM_Spatial_FB_Only_GLM(nn.Module):

    def __init__(self,
                 n_basis_stim_spat: int,
                 n_basis_feedback: int,
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 spatial_sparsity_l1_lambda: float = 0.0,
                 stim_spat_init_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 multimovie_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.n_basis_stim_spat = n_basis_stim_spat
        self.n_basis_feedback = n_basis_feedback

        self.loss_callable = loss_callable
        self.spatial_sparsity_l1_lambda = spatial_sparsity_l1_lambda

        self.use_multimovie_weights = False
        if multimovie_weights is not None:
            if isinstance(multimovie_weights, np.ndarray):
                self.register_buffer('multimovie_weights', torch.tensor(multimovie_weights, dtype=dtype))
            else:
                self.register_buffer('multimovie_weights', multimovie_weights.detach().clone())

        # OPTIMIZATION VARIABLE 1
        self.feedback_w = nn.Parameter(torch.empty((1, n_basis_feedback), dtype=dtype),
                                       requires_grad=True)
        nn.init.uniform_(self.feedback_w, a=-1e-2, b=1e-2)

        # OPTIMIZATION VARIABLE 2, shape (1, n_basis_stim_spat)
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

        # OPTIMIZATION VARIABLE 3
        self.bias = nn.Parameter(torch.empty((1,), dtype=dtype), requires_grad=True)
        nn.init.uniform_(self.bias, a=-1e-2, b=1e-2)

    def _eval_loss(self,
                   feedback_w: torch.Tensor,
                   spat_filt_w: torch.Tensor,
                   bias: torch.Tensor,
                   filt_multimovie: List[torch.Tensor],
                   cell_spike_multibin: List[torch.Tensor],
                   filt_feedback_multibins: List[torch.Tensor]) -> torch.Tensor:

        stimulus_contrib = _multimovie_spatial_stimulus_contrib_gensig(spat_filt_w, filt_multimovie)
        feedback_contrib = _multimovie_feedback_contrib_gensig(feedback_w,
                                                               filt_feedback_multibins)
        gen_sig_list = [bias + x + y for x, y in zip(stimulus_contrib, feedback_contrib)]

        spiking_loss_per_movie = [self.loss_callable(gen_sig[:-1], binned_spikes[1:])
                                  for gen_sig, binned_spikes in zip(gen_sig_list, cell_spike_multibin)]

        if self.use_multimovie_weights:
            stacked_losses = torch.stack(spiking_loss_per_movie, dim=0)
            spiking_loss = torch.sum(self.multimovie_weights * stacked_losses)
        else:
            spiking_loss = torch.mean(torch.stack(spiking_loss_per_movie, dim=0))

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
                               feedback_filter_w: torch.Tensor,
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

        return _feedback_only_spatial_model_prox_project_variables(
            feedback_filter_w, spat_filter_w, bias_w, self.spatial_sparsity_l1_lambda)

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
        self.feedback_w.data[:] = coord_desc_other.feedback_w.data[:]
        self.bias.data[:] = coord_desc_other.bias.data[:]

        self.zero_grad()

    def return_parameters_np(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.feedback_w.detach().cpu().numpy(),
            self.stim_spat_w.detach().cpu().numpy(),
            self.bias.detach().cpu().numpy()
        )
