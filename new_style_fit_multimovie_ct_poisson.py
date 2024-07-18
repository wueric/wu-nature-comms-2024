import numpy as np
import torch
from torch import autocast
from torch.cuda.amp import GradScaler

from typing import List, Tuple, Callable, Iterator, Optional, Union

import lib.data_utils.dynamic_data_util as ddu
from convex_optim_base.prox_optim import ProxFISTASolverParams
from convex_optim_base.unconstrained_optim import FistaSolverParams
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lnp_precompute.ct_lnp_precompute import frame_rate_lnp_multidata_bin_spikes, \
    frame_rate_multidata_precompute_spatial_convolutions, frame_rate_multidata_precompute_temporal_convolutions
from lnp_precompute.wn_lnp_precompute import frame_rate_wn_lnp_multidata_bin_spikes, \
    frame_rate_wn_multidata_precompute_spatial_convolutions, frame_rate_wn_multidata_precompute_temporal_convolutions
from new_style_optim.prox_optim import NewStyleSingleProxFISTAOptim
from new_style_optim.accelerated_unconstrained_optim import NewStyleSingleFISTAOptim
from new_style_optim_encoder.multimovie_ct_poisson import NS_MM_Timecourse_FrameRateLNP, NS_MM_Spatial_FrameRateLNP


def lnp_optim_timecourse(timecourse_model: NS_MM_Timecourse_FrameRateLNP,
                         basis_filt_multi: List[torch.Tensor],
                         binned_spikes_cell_multi: List[torch.Tensor],
                         fista_params: FistaSolverParams) -> float:

    no_autocast_fn = timecourse_model.make_loss_eval_callable(
        basis_filt_multi,
        binned_spikes_cell_multi)

    def autocast_nograd_loss_eval(*args) -> float:
        with autocast('cuda'):
            return no_autocast_fn(*args)

    optimizer = NewStyleSingleFISTAOptim(
        timecourse_model.parameters(),
        lr=fista_params.initial_learning_rate,
        backtrack_beta=fista_params.backtracking_beta,
        eval_loss_callable_fn=autocast_nograd_loss_eval,
        verbose=True
    )

    scaler = GradScaler()

    if fista_params.max_iter > 0:
        for iter in range(fista_params.max_iter):
            optimizer.zero_grad()
            with autocast('cuda'):
                loss = timecourse_model(basis_filt_multi,
                                        binned_spikes_cell_multi)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        optimizer.finish()

    with torch.no_grad(), autocast('cuda'):
        return timecourse_model(basis_filt_multi,
                                binned_spikes_cell_multi).item()


def lnp_optim_spatfilt(spat_model: NS_MM_Spatial_FrameRateLNP,
                       basis_filt_multi: List[torch.Tensor],
                       binned_spikes_cell_multi: List[torch.Tensor],
                       fista_params: ProxFISTASolverParams) -> float:
    no_autocast_fn = spat_model.make_loss_eval_callable(
        basis_filt_multi,
        binned_spikes_cell_multi
    )

    def autocast_nograd_loss_eval(*args) -> float:
        with autocast('cuda'):
            return no_autocast_fn(*args)

    optimizer = NewStyleSingleProxFISTAOptim(spat_model.parameters(),
                                             lr=fista_params.initial_learning_rate,
                                             backtrack_beta=fista_params.backtracking_beta,
                                             eval_loss_callable_fn=autocast_nograd_loss_eval,
                                             prox_callable_fn=spat_model.prox_project_variables,
                                             verbose=True)

    scaler = GradScaler()

    for iter in range(fista_params.max_iter):
        optimizer.zero_grad()

        with autocast('cuda'):
            loss = spat_model(basis_filt_multi,
                              binned_spikes_cell_multi)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    optimizer.finish()

    with torch.no_grad(), autocast('cuda'):
        return spat_model(basis_filt_multi,
                          binned_spikes_cell_multi).item()


def new_style_frame_rate_wn_regularized_precompute_convs_and_fit_lnp(
        jittered_movie_blocks: List[ddu.LoadedBrownianMovieBlock],
        wn_movie_blocks: List[ddu.LoadedWNMovieBlock],
        wn_weight: float,
        stim_spat_basis_imshape: np.ndarray,
        stim_time_basis: np.ndarray,
        center_cell_wn: Tuple[str, int],
        cells_ordered: OrderedMatchedCellsStruct,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        spatial_sparsity_l1_lambda: float,
        n_alternating_iters: int,
        alternating_opt_eps: float,
        solver_params_iter: Tuple[
            Iterator[ProxFISTASolverParams], Iterator[FistaSolverParams]],
        device: torch.device,
        initial_guess_spat: Optional[Union[torch.Tensor, np.ndarray]] = None,
        initial_guess_timecourse: Optional[
            Union[torch.Tensor, np.ndarray]] = None,
        log_verbose_ascent: bool = False,
        movie_spike_dtype: torch.dtype = torch.float32,
        jitter_spike_times: float = 0.0) \
        -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    filt_height, filt_width = stim_spat_basis_imshape.shape[1:]
    stim_spat_basis = stim_spat_basis_imshape.reshape(stim_spat_basis_imshape.shape[0], -1)
    n_bins_timefilt = stim_time_basis.shape[1]
    print('n_bins_timefilt', n_bins_timefilt)

    # bin spikes, store on GPU
    # since this is just for LNP fitting, we don't need to do any
    # precomputations with the spike trains themselves
    ns_center_spikes_frame_rate_torch = frame_rate_lnp_multidata_bin_spikes(
        jittered_movie_blocks,
        center_cell_wn,
        cells_ordered,
        device,
        prec_dtype=movie_spike_dtype,
        jitter_spike_times=jitter_spike_times,
        trim_spikes_seq=(n_bins_timefilt - 1)
    )

    ns_weights_per_block = np.array([x.shape[0] for x in ns_center_spikes_frame_rate_torch])
    ns_weights_per_block = ns_weights_per_block / np.sum(ns_weights_per_block)

    wn_center_spikes_frame_rate_torch = frame_rate_wn_lnp_multidata_bin_spikes(
        wn_movie_blocks,
        center_cell_wn,
        device,
        jitter_time_amount=jitter_spike_times,
        prec_dtype=movie_spike_dtype,
        trim_spikes_seq=(n_bins_timefilt - 1)
    )

    wn_weights_per_block_ = np.array([x.shape[0] for x in wn_center_spikes_frame_rate_torch])
    wn_weights_per_block = wn_weight * (wn_weights_per_block_ / np.sum(wn_weights_per_block_))

    all_weights_per_block = np.concatenate([ns_weights_per_block, wn_weights_per_block])
    all_center_spikes_torch = [*ns_center_spikes_frame_rate_torch, *wn_center_spikes_frame_rate_torch]

    # Step 3: do the alternating optimization
    # (a) build the optimization modules
    n_basis_stim_spat, n_basis_stim_time = stim_spat_basis.shape[0], stim_time_basis.shape[0]

    with torch.no_grad():
        if initial_guess_timecourse is None:
            timecourse_basis_repr = torch.rand((1, n_basis_stim_time), device=device,
                                               dtype=torch.float32).mul_(2e-2).add_(- 1e-2)  # type: torch.Tensor
        else:
            timecourse_basis_repr = torch.tensor(initial_guess_timecourse,
                                                 device=device,
                                                 dtype=torch.float32)  # type: torch.Tensor

        if initial_guess_spat is None:
            spat_filt_basis_repr = torch.rand((1, n_basis_stim_spat), dtype=torch.float32,
                                              device=device).mul_(2e-2).add_(-1e-2)  # type: torch.Tensor
        else:
            spat_filt_basis_repr = torch.tensor(initial_guess_spat, device=device,
                                                dtype=torch.float32)  # type: torch.Tensor

    stim_spat_basis_torch = torch.tensor(stim_spat_basis, device=device, dtype=torch.float32)
    stim_time_basis_torch = torch.tensor(stim_time_basis, device=device, dtype=torch.float32)

    timecourse_model = NS_MM_Timecourse_FrameRateLNP(
        n_basis_stim_time,
        loss_callable,
        stim_time_init_guess=timecourse_basis_repr,
        multimovie_weights=all_weights_per_block
    ).to(device)

    spatial_model = NS_MM_Spatial_FrameRateLNP(
        n_basis_stim_spat,
        loss_callable,
        spatial_sparsity_l1_lambda=spatial_sparsity_l1_lambda,
        stim_spat_init_guess=initial_guess_spat,
        multimovie_weights=all_weights_per_block
    ).to(device)

    prev_iter_loss = float('inf')
    for iter, spat_solver_params, time_solver_params in \
            zip(range(n_alternating_iters), solver_params_iter[0], solver_params_iter[1]):

        # (1, n_basis_stim_time) @ (n_basis_stim_time, n_bins_filter)
        # -> (1, n_bins_filter) -> (n_bins_filter, )
        with torch.no_grad():
            timecourse_waveform = (timecourse_basis_repr @ stim_time_basis_torch).squeeze(0)
            timecourse_norm = torch.linalg.norm(timecourse_waveform)
            timecourse_waveform.div_(timecourse_norm)
            timecourse_basis_repr.div_(timecourse_norm)
            spat_filt_basis_repr.mul_(timecourse_norm)

        if iter != 0:
            # this updates the feedback, coupling, bias, and auxiliary variables
            # stimulus filters are tracked separately
            spatial_model.clone_parameters_model(timecourse_model)
            spatial_model.set_spat_filter(spat_filt_basis_repr)

        # precompute application of spatial basis with fixed timecourse on the Brownian movie
        ns_precomputed_spatial_convolutions = frame_rate_multidata_precompute_spatial_convolutions(
            jittered_movie_blocks,
            stim_spat_basis,
            timecourse_waveform.detach().cpu().numpy(),
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        # precompute application of the spatial basis with fixed timecourse on the WN movie
        wn_precomputed_spatial_convolutions = frame_rate_wn_multidata_precompute_spatial_convolutions(
            wn_movie_blocks,
            stim_spat_basis_imshape,
            timecourse_waveform.detach().cpu().numpy(),
            image_transform_lambda,
            device,
            prec_dtype=movie_spike_dtype
        )

        precomputed_spatial_convolutions = [*ns_precomputed_spatial_convolutions, *wn_precomputed_spatial_convolutions]

        _ = lnp_optim_spatfilt(spatial_model,
                               precomputed_spatial_convolutions,
                               all_center_spikes_torch,
                               spat_solver_params)

        # now do a clean up of precomputed_spatial_convolutions
        del ns_precomputed_spatial_convolutions, wn_precomputed_spatial_convolutions, precomputed_spatial_convolutions

        # this updates the feedback, coupling, bias, and auxiliary variables
        # stimulus filters are tracked separately
        timecourse_model.clone_parameters_model(spatial_model)
        timecourse_model.set_time_filter(timecourse_basis_repr)

        with torch.no_grad():
            # manually update the stimulus spatial filter
            # shape (1, n_basis_stim_spat)
            spat_filt_basis_repr = spatial_model.return_spat_filt_parameters()
            # shape (n_pixels, )
            spat_filt = (spat_filt_basis_repr @ stim_spat_basis_torch).squeeze(0)

        spat_filt_np = spat_filt.detach().cpu().numpy()

        # precompute application of the temporal basis with
        # known fixed spatial filter for jittered movie stimulus
        ns_precomputed_temp_convolutions = frame_rate_multidata_precompute_temporal_convolutions(
            jittered_movie_blocks,
            spat_filt_np,
            stim_time_basis,
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        # precompute application of the temporal basis with
        # known fixed spatial filter for WN stimulus
        wn_precomputed_temp_convolutions = frame_rate_wn_multidata_precompute_temporal_convolutions(
            wn_movie_blocks,
            spat_filt_np.reshape(filt_height, filt_width),
            stim_time_basis,
            image_transform_lambda,
            device,
            prec_dtype=movie_spike_dtype
        )

        all_precomputed_temp_convolutions = [*ns_precomputed_temp_convolutions, *wn_precomputed_temp_convolutions]

        iter_loss = lnp_optim_timecourse(timecourse_model,
                                     all_precomputed_temp_convolutions,
                                     all_center_spikes_torch,
                                     time_solver_params)

        # clean up of precomputed_temp_convolutions
        del all_precomputed_temp_convolutions, ns_precomputed_temp_convolutions, wn_precomputed_temp_convolutions

        # update timecourse parameters
        # shape (1, n_basis_stim_time)
        timecourse_basis_repr = timecourse_model.return_timecourse_params()

        delta_loss = prev_iter_loss - iter_loss
        prev_iter_loss = iter_loss

        if log_verbose_ascent:
            print("Coord. descent iter {0} loss {1}".format(iter, prev_iter_loss))

        if delta_loss < alternating_opt_eps:
            break

    # clean up GPU
    del all_center_spikes_torch, wn_center_spikes_frame_rate_torch, ns_center_spikes_frame_rate_torch
    del stim_spat_basis_torch, stim_time_basis_torch

    # return the parameters
    _, bias = timecourse_model.return_parameters_np()

    del timecourse_model, spatial_model

    spat_filt_basis_repr_np = spat_filt_basis_repr.detach().cpu().numpy()
    timecourse_basis_repr_np = timecourse_basis_repr.detach().cpu().numpy()

    torch.cuda.empty_cache()

    return prev_iter_loss, (spat_filt_basis_repr_np, timecourse_basis_repr_np, bias)
