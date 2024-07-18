from typing import List, Dict, Tuple, Callable, Iterator, Union, Optional

import numpy as np
import torch
from torch import autocast

import lib.data_utils.dynamic_data_util as ddu
from convex_optim_base.prox_optim import ProxFISTASolverParams
from convex_optim_base.unconstrained_optim import FistaSolverParams
from glm_precompute.ct_glm_precompute import multidata_precompute_spatial_convolutions, \
    multidata_precompute_temporal_convolutions, _count_coupled_cells, \
    multidata_bin_center_spikes_precompute_feedback_convs, multidata_bin_center_spikes_precompute_coupling_convs, \
    repeats_blocks_extract_center_spikes_precompute_feedback_convs, repeats_blocks_precompute_coupling_convs, \
    repeats_blocks_precompute_spatial_convolutions, repeats_blocks_precompute_temporal_convolutions
from glm_precompute.flashed_glm_precompute import flashed_ns_precompute_spatial_basis, \
    flashed_ns_bin_spikes_precompute_feedback_convs2, \
    flashed_ns_bin_spikes_precompute_coupling_convs2, flashed_ns_bin_spikes_precompute_timecourse_basis_conv2
from glm_precompute.wn_glm_precompute import multidata_wn_preapply_spatial_convolutions, \
    multidata_wn_preapply_temporal_convolutions, multimovie_wn_bin_spikes_precompute_feedback_convs, \
    multimovie_wn_bin_spikes_precompute_coupling_convs
from lib.data_utils.dynamic_data_util import LoadedBrownianMovieBlock, LoadedWNMovieBlock, LoadedFlashPatch
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from new_style_optim.accelerated_unconstrained_optim import _optim_unconstrained_FISTA
from new_style_optim.prox_optim import _optim_FISTA
from new_style_optim_encoder.joint_flashed_and_multimovie_glm import JointFlashedMultimovie_TimecourseFitGLM, \
    JointFlashedMultimovie_SpatFitGLM, JointFlashedMultimovie_SpatFit_FeedbackOnlyGLM, \
    JointFlashedMultimovie_TimecourseFit_FeedbackOnlyGLM
from new_style_optim_encoder.multimovie_ct_glm import NS_MM_Timecourse_GroupSparseLRCroppedCT_GLM, \
    NS_MM_Spatial_GroupSparseLRCroppedCT_GLM, NS_MM_Timecourse_FB_Only_GLM, NS_MM_Spatial_FB_Only_GLM


def new_style_precompute_convs_and_fit_glm(jittered_movie_blocks: List[LoadedBrownianMovieBlock],
                                           stim_spat_basis: np.ndarray,
                                           stim_time_basis: np.ndarray,
                                           feedback_basis: np.ndarray,
                                           coupling_basis: np.ndarray,
                                           center_cell_wn: Tuple[str, int],
                                           coupled_cells: Dict[str, List[int]],
                                           cells_ordered: OrderedMatchedCellsStruct,
                                           bin_width_samples: int,
                                           image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
                                           loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                           l21_group_sparse_lambda: float,
                                           spatial_sparsity_l1_lambda: float,
                                           n_alternating_iters: int,
                                           alternating_opt_eps: float,
                                           solver_params_iter: Tuple[
                                               Iterator[ProxFISTASolverParams], Iterator[ProxFISTASolverParams]],
                                           device: torch.device,
                                           initial_guess_spat: Optional[Union[torch.Tensor, np.ndarray]] = None,
                                           initial_guess_timecourse: Optional[
                                               Union[torch.Tensor, np.ndarray]] = None,
                                           trim_spikes_seq: int = 0,
                                           jitter_spike_times: float = 0.0,
                                           log_verbose_ascent: bool = False,
                                           movie_spike_dtype: torch.dtype = torch.float32) \
        -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    '''
    New-style optimization using torch.amp mixed precision training and external optimizer
        that inherits from torch.optim.Optimizer rather than our homegrown optimization framework

    This arrangement is capable of using torch.amp mixed precision training, which greatly reduces
        the memory consumption and enables training with much larger datasets and larger basis sets
        than previously possible.

    We also add optional joint training with white noise for regularization purposes

    :param jittered_movie_blocks:
    :param stim_spat_basis: shape (n_basis, n_pix)
    :param stim_time_basis:
    :param feedback_basis:
    :param coupling_basis:
    :param center_cell_wn:
    :param coupled_cells:
    :param cells_ordered:
    :param bin_width_samples:
    :param image_transform_lambda:
    :param loss_callable:
    :param l21_group_sparse_lambda:
    :param spatial_sparsity_l1_lambda:
    :param n_alternating_iters:
    :param alternating_opt_eps:
    :param solver_params_iter:
    :param device:
    :param fit_noncausal:
    :param initial_guess_spat:
    :param initial_guess_timecourse:
    :param trim_spikes_seq:
    :return:
    '''

    # Step 0: precompute spiking quantities for the NScenes
    # Also bin spikes, pre-compute spike train convolutions, and store those on GPU
    # Also compute frame overlaps, store those on GPU as well
    ns_timebins_all = ddu.multimovie_construct_natural_movies_timebins(jittered_movie_blocks,
                                                                       bin_width_samples)

    frame_sel_and_overlaps_all = ddu.multimovie_compute_interval_overlaps(jittered_movie_blocks,
                                                                          ns_timebins_all)

    center_spikes_all_torch, feedback_conv_all_torch = multidata_bin_center_spikes_precompute_feedback_convs(
        jittered_movie_blocks,
        ns_timebins_all,
        feedback_basis,
        center_cell_wn,
        cells_ordered,
        device,
        trim_spikes_seq=trim_spikes_seq,
        jitter_spike_times=jitter_spike_times,
        prec_dtype=movie_spike_dtype
    )

    coupling_conv_all_torch = multidata_bin_center_spikes_precompute_coupling_convs(
        jittered_movie_blocks,
        ns_timebins_all,
        coupling_basis,
        coupled_cells,
        cells_ordered,
        device,
        prec_dtype=movie_spike_dtype,
        jitter_spike_times=jitter_spike_times
    )

    # Step 3: do the alternating optimization
    # remember to clean up upsampled movies after each optimization problem
    # to save GPU space

    # (a) build the optimization modules
    n_basis_stim_spat, n_basis_stim_time = stim_spat_basis.shape[0], stim_time_basis.shape[0]
    n_basis_feedback = feedback_basis.shape[0]
    n_basis_coupling = coupling_basis.shape[0]
    n_coupled_cells = _count_coupled_cells(coupled_cells)

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

    multimovie_weights = np.array([x.shape[0] - 1 for x in center_spikes_all_torch])
    multimovie_weights = multimovie_weights / np.sum(multimovie_weights)

    timecourse_model = NS_MM_Timecourse_GroupSparseLRCroppedCT_GLM(
        n_basis_stim_time,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        loss_callable,
        l21_group_sparse_lambda,
        stim_time_init_guess=timecourse_basis_repr,
        multimovie_weights=multimovie_weights
    ).to(device)

    spatial_model = NS_MM_Spatial_GroupSparseLRCroppedCT_GLM(
        n_basis_stim_spat,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        loss_callable,
        l21_group_sparse_lambda,
        spatial_sparsity_l1_lambda=spatial_sparsity_l1_lambda,
        stim_spat_init_guess=initial_guess_spat,
        multimovie_weights=multimovie_weights
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

        # pre-compute the application of the spatial basis
        # with the fixed timecourse on the upsampled movie
        # and store the result on GPU
        precomputed_spatial_convolutions = multidata_precompute_spatial_convolutions(
            jittered_movie_blocks,
            frame_sel_and_overlaps_all,
            stim_spat_basis,
            timecourse_waveform.detach().cpu().numpy(),
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        _ = _optim_FISTA(spatial_model,
                         spat_solver_params,
                         precomputed_spatial_convolutions,
                         center_spikes_all_torch,
                         coupling_conv_all_torch,
                         feedback_conv_all_torch)

        # now do a clean up of precomputed_spatial_convolutions
        del precomputed_spatial_convolutions

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

        # precompute application of the temporal basis with
        # known fixed spatial filter
        precomputed_temp_convolutions = multidata_precompute_temporal_convolutions(
            jittered_movie_blocks,
            frame_sel_and_overlaps_all,
            spat_filt.detach().cpu().numpy(),
            stim_time_basis,
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        iter_loss = _optim_FISTA(timecourse_model,
                                 time_solver_params,
                                 precomputed_temp_convolutions,
                                 center_spikes_all_torch,
                                 coupling_conv_all_torch,
                                 feedback_conv_all_torch)

        # clean up of precomputed_temp_convolutions
        del precomputed_temp_convolutions

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
    del center_spikes_all_torch, feedback_conv_all_torch, coupling_conv_all_torch
    del stim_spat_basis_torch, stim_time_basis_torch

    # return the parameters
    coupling_filters_w, feedback_filter_w, _, bias = timecourse_model.return_parameters_np()

    del timecourse_model, spatial_model

    spat_filt_basis_repr_np = spat_filt_basis_repr.detach().cpu().numpy()
    timecourse_basis_repr_np = timecourse_basis_repr.detach().cpu().numpy()

    torch.cuda.empty_cache()

    return prev_iter_loss, (
        coupling_filters_w, feedback_filter_w, spat_filt_basis_repr_np, timecourse_basis_repr_np, bias)


def new_style_wn_regularized_precompute_convs_and_fit_glm(
        jittered_movie_blocks: List[LoadedBrownianMovieBlock],
        wn_movie_blocks: List[LoadedWNMovieBlock],
        wn_weight: float,
        stim_spat_basis_imshape: np.ndarray,
        stim_time_basis: np.ndarray,
        feedback_basis: np.ndarray,
        coupling_basis: np.ndarray,
        center_cell_wn: Tuple[str, int],
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        bin_width_samples: int,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        l21_group_sparse_lambda: float,
        spatial_sparsity_l1_lambda: float,
        n_alternating_iters: int,
        alternating_opt_eps: float,
        solver_params_iter: Tuple[
            Iterator[ProxFISTASolverParams], Iterator[ProxFISTASolverParams]],
        device: torch.device,
        initial_guess_spat: Optional[Union[torch.Tensor, np.ndarray]] = None,
        initial_guess_timecourse: Optional[
            Union[torch.Tensor, np.ndarray]] = None,
        trim_spikes_seq: int = 0,
        log_verbose_ascent: bool = False,
        movie_spike_dtype: torch.dtype = torch.float32,
        jitter_spike_times: float = 0.0) \
        -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    '''
    New-style optimization using torch.amp mixed precision training and external optimizer
        that inherits from torch.optim.Optimizer rather than our homegrown optimization framework

    This arrangement is capable of using torch.amp mixed precision training, which greatly reduces
        the memory consumption and enables training with much larger datasets and larger basis sets
        than previously possible.

    We also joint training with white noise for regularization purposes.

    How do we make a coherent unified framework for fitting the combined
        jittered movie / white noise GLMs? We have the following constraints:

        * The jittered movies are very slow to load from disk, and so we should
            load them once at the beginning (i.e. outside this function) and then
            reuse them whenever possible
        * The white noise frames can be generated on the fly inside this function
            or can be generated outside of this function. We might prefer to do the
            white noise frame generation, cropping, and BW conversion outside this function to
            minimize clutter in this function
        * JitteredMovieTrainingData manages the timing information and the underlying
            Vision dataset for the training partition of the jittered movies. Managing the
            training and test partitions of the jittered movies is a little bit more complicated
            than for the white noise since we have discrete images being shown, and because the
            jittered movies have a block trial structure.

            Better idea: merge the preloaded frames into a JitteredMovieData object
                and then make that object responsible for spitting out the right frames
                and transition times

            We need an equivalent data structure for managing timing and the Vision dataset
            for the white noise datarun. We introduce a new class called LoadedWNMovies
            which should be able to manage timing, the Vision dataset

    :param movie_datasets:
    :param movie_patches_by_dataset:
    :param stim_spat_basis_imshape: shape (n_basis, height, width)
    :param stim_time_basis:
    :param feedback_basis:
    :param coupling_basis:
    :param center_cell_wn:
    :param coupled_cells:
    :param cells_ordered:
    :param bin_width_samples:
    :param image_transform_lambda:
    :param loss_callable:
    :param l21_group_sparse_lambda:
    :param spatial_sparsity_l1_lambda:
    :param n_alternating_iters:
    :param alternating_opt_eps:
    :param solver_params_iter:
    :param device:
    :param initial_guess_spat:
    :param initial_guess_timecourse:
    :param trim_spikes_seq:
    :return:
    '''

    filt_height, filt_width = stim_spat_basis_imshape.shape[1:]
    stim_spat_basis = stim_spat_basis_imshape.reshape(stim_spat_basis_imshape.shape[0], -1)

    # Step 0: precompute spiking quantities for the NScenes
    # Also bin spikes, pre-compute spike train convolutions, and store those on GPU
    # Also compute frame overlaps, store those on GPU as well
    ns_timebins_all = ddu.multimovie_construct_natural_movies_timebins(jittered_movie_blocks,
                                                                       bin_width_samples)

    ns_frame_sel_and_overlaps_all = ddu.multimovie_compute_interval_overlaps(jittered_movie_blocks,
                                                                             ns_timebins_all)

    ns_center_spikes_all_torch, ns_feedback_conv_all_torch = multidata_bin_center_spikes_precompute_feedback_convs(
        jittered_movie_blocks,
        ns_timebins_all,
        feedback_basis,
        center_cell_wn,
        cells_ordered,
        device,
        trim_spikes_seq=trim_spikes_seq,
        jitter_spike_times=jitter_spike_times,
        prec_dtype=movie_spike_dtype
    )

    ns_coupling_conv_all_torch = multidata_bin_center_spikes_precompute_coupling_convs(
        jittered_movie_blocks,
        ns_timebins_all,
        coupling_basis,
        coupled_cells,
        cells_ordered,
        device,
        prec_dtype=movie_spike_dtype,
        jitter_spike_times=jitter_spike_times
    )

    ns_weights_per_block = np.array([x.shape[0] - 1 for x in ns_center_spikes_all_torch])
    ns_weights_per_block = ns_weights_per_block / np.sum(ns_weights_per_block)

    ###############################################################
    # now do the same for the white noise
    wn_timebins_all = ddu.multimovie_construct_wn_timebins(wn_movie_blocks,
                                                           bin_width_samples)

    wn_sel_overlaps = ddu.multimovie_compute_interval_overlaps_for_wn(wn_movie_blocks,
                                                                      wn_timebins_all)

    wn_center_spikes_all_torch, wn_feedback_conv_all_torch = multimovie_wn_bin_spikes_precompute_feedback_convs(
        wn_movie_blocks,
        wn_timebins_all,
        center_cell_wn,
        feedback_basis,
        device,
        jitter_time_amount=jitter_spike_times,
        prec_dtype=movie_spike_dtype,
        trim_spikes_seq=trim_spikes_seq
    )

    wn_coupling_conv_all_torch = multimovie_wn_bin_spikes_precompute_coupling_convs(
        wn_movie_blocks,
        wn_timebins_all,
        coupled_cells,
        cells_ordered,
        coupling_basis,
        device,
        jitter_time_amount=jitter_spike_times,
        prec_dtype=movie_spike_dtype
    )

    wn_weights_per_block_ = np.array([x.shape[0] - 1 for x in wn_center_spikes_all_torch])
    wn_weights_per_block = wn_weight * (wn_weights_per_block_ / np.sum(wn_weights_per_block_))

    all_center_spikes_torch = [*ns_center_spikes_all_torch, *wn_center_spikes_all_torch]
    all_feedback_conv_torch = [*ns_feedback_conv_all_torch, *wn_feedback_conv_all_torch]
    all_coupling_conv_torch = [*ns_coupling_conv_all_torch, *wn_coupling_conv_all_torch]

    all_weights_per_block = np.concatenate([ns_weights_per_block, wn_weights_per_block])

    # Step 3: do the alternating optimization
    # remember to clean up upsampled movies after each optimization problem
    # to save GPU space

    # (a) build the optimization modules
    n_basis_stim_spat, n_basis_stim_time = stim_spat_basis.shape[0], stim_time_basis.shape[0]
    n_basis_feedback = feedback_basis.shape[0]
    n_basis_coupling = coupling_basis.shape[0]
    n_coupled_cells = _count_coupled_cells(coupled_cells)

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

    timecourse_model = NS_MM_Timecourse_GroupSparseLRCroppedCT_GLM(
        n_basis_stim_time,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        loss_callable,
        l21_group_sparse_lambda,
        stim_time_init_guess=timecourse_basis_repr,
        multimovie_weights=all_weights_per_block
    ).to(device)

    spatial_model = NS_MM_Spatial_GroupSparseLRCroppedCT_GLM(
        n_basis_stim_spat,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        loss_callable,
        l21_group_sparse_lambda,
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

        # pre-compute the application of the spatial basis
        # with the fixed timecourse on the upsampled jittered movie
        # and store the result on GPU
        ns_precomputed_spatial_convolutions = multidata_precompute_spatial_convolutions(
            jittered_movie_blocks,
            ns_frame_sel_and_overlaps_all,
            stim_spat_basis,
            timecourse_waveform.detach().cpu().numpy(),
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        # pre-compute the application of the spatial basis
        # with the fixed timecourse on the upsampled WN movie
        # and store the results on GPU
        wn_precomputed_spatial_convolutions = multidata_wn_preapply_spatial_convolutions(
            wn_movie_blocks,
            wn_sel_overlaps,
            stim_spat_basis_imshape,
            timecourse_waveform.detach().cpu().numpy(),
            image_transform_lambda,
            device,
            prec_dtype=movie_spike_dtype
        )

        precomputed_spatial_convolutions = [*ns_precomputed_spatial_convolutions, *wn_precomputed_spatial_convolutions]

        _ = _optim_FISTA(spatial_model,
                         spat_solver_params,
                         precomputed_spatial_convolutions,
                         all_center_spikes_torch,
                         all_feedback_conv_torch,
                         all_coupling_conv_torch)

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
        ns_precomputed_temp_convolutions = multidata_precompute_temporal_convolutions(
            jittered_movie_blocks,
            ns_frame_sel_and_overlaps_all,
            spat_filt_np,
            stim_time_basis,
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        # precompute application of the temporal basis with
        # known fixed spatial filter for WN stimulus
        wn_precomputed_temp_convolutions = multidata_wn_preapply_temporal_convolutions(
            wn_movie_blocks,
            wn_sel_overlaps,
            spat_filt_np.reshape(filt_height, filt_width),
            stim_time_basis,
            image_transform_lambda,
            device,
            prec_dtype=movie_spike_dtype
        )

        all_precomputed_temp_convolutions = [*ns_precomputed_temp_convolutions, *wn_precomputed_temp_convolutions]

        iter_loss = _optim_FISTA(timecourse_model,
                                 time_solver_params,
                                 all_precomputed_temp_convolutions,
                                 all_center_spikes_torch,
                                 all_feedback_conv_torch,
                                 all_coupling_conv_torch)

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
    del all_center_spikes_torch, all_coupling_conv_torch, all_feedback_conv_torch
    del ns_center_spikes_all_torch, ns_coupling_conv_all_torch, ns_feedback_conv_all_torch
    del wn_center_spikes_all_torch, wn_coupling_conv_all_torch, wn_feedback_conv_all_torch
    del stim_spat_basis_torch, stim_time_basis_torch

    # return the parameters
    coupling_filters_w, feedback_filter_w, _, bias = timecourse_model.return_parameters_np()

    del timecourse_model, spatial_model

    spat_filt_basis_repr_np = spat_filt_basis_repr.detach().cpu().numpy()
    timecourse_basis_repr_np = timecourse_basis_repr.detach().cpu().numpy()

    torch.cuda.empty_cache()

    return prev_iter_loss, (
        coupling_filters_w, feedback_filter_w, spat_filt_basis_repr_np, timecourse_basis_repr_np, bias)


def new_style_joint_flashed_jittered_precompute_convs_and_fit_glm(
        jittered_movie_blocks: List[LoadedBrownianMovieBlock],
        flashed_stimulus_block: List[LoadedFlashPatch],
        flashed_image_relative_weight: float,
        wn_movie_blocks: List[LoadedWNMovieBlock],
        wn_weight: float,
        stim_spat_basis_imshape: np.ndarray,
        stim_time_basis: np.ndarray,
        feedback_basis: np.ndarray,
        coupling_basis: np.ndarray,
        center_cell_wn: Tuple[str, int],
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        bin_width_samples: int,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        l21_group_sparse_lambda: float,
        spatial_sparsity_l1_lambda: float,
        n_alternating_iters: int,
        alternating_opt_eps: float,
        solver_params_iter: Tuple[
            Iterator[ProxFISTASolverParams], Iterator[ProxFISTASolverParams]],
        device: torch.device,
        initial_guess_spat: Optional[Union[torch.Tensor, np.ndarray]] = None,
        initial_guess_timecourse: Optional[
            Union[torch.Tensor, np.ndarray]] = None,
        trim_spikes_seq: int = 0,
        log_verbose_ascent: bool = False,
        movie_spike_dtype: torch.dtype = torch.float32,
        jitter_spike_times: float = 0.0) \
        -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    filt_height, filt_width = stim_spat_basis_imshape.shape[1:]
    stim_spat_basis = stim_spat_basis_imshape.reshape(stim_spat_basis_imshape.shape[0], -1)

    # precompute quantitites for the flashed natural images
    flashed_spikes, flashed_feedback_basis_conv = flashed_ns_bin_spikes_precompute_feedback_convs2(
        flashed_stimulus_block,
        feedback_basis,
        center_cell_wn,
        cells_ordered,
        device,
        jitter_time_amount=jitter_spike_times,
        prec_dtype=movie_spike_dtype,
        trim_spikes_seq=trim_spikes_seq
    )

    flashed_couple_basis_conv = flashed_ns_bin_spikes_precompute_coupling_convs2(
        flashed_stimulus_block,
        coupling_basis,
        coupled_cells,
        cells_ordered,
        device,
        jitter_time_amount=jitter_spike_times,
        prec_dtype=movie_spike_dtype
    )

    flashed_timecourse_basis_conv = flashed_ns_bin_spikes_precompute_timecourse_basis_conv2(
        flashed_stimulus_block,
        stim_time_basis,
        device,
        prec_dtype=movie_spike_dtype
    )

    flashed_spat_basis_filt = flashed_ns_precompute_spatial_basis(
        flashed_stimulus_block,
        stim_spat_basis,
        image_transform_lambda,
        device,
        prec_dtype=movie_spike_dtype)

    # then precompute quantities for the jittered stimuli
    movie_timebins_all = ddu.multimovie_construct_natural_movies_timebins(jittered_movie_blocks,
                                                                          bin_width_samples)

    movie_frame_sel_and_overlaps_all = ddu.multimovie_compute_interval_overlaps(jittered_movie_blocks,
                                                                                movie_timebins_all)

    movie_center_spikes_all_torch, movie_feedback_conv_all_torch = multidata_bin_center_spikes_precompute_feedback_convs(
        jittered_movie_blocks,
        movie_timebins_all,
        feedback_basis,
        center_cell_wn,
        cells_ordered,
        device,
        trim_spikes_seq=trim_spikes_seq,
        jitter_spike_times=jitter_spike_times,
        prec_dtype=movie_spike_dtype
    )

    movie_coupling_conv_all_torch = multidata_bin_center_spikes_precompute_coupling_convs(
        jittered_movie_blocks,
        movie_timebins_all,
        coupling_basis,
        coupled_cells,
        cells_ordered,
        device,
        prec_dtype=movie_spike_dtype,
        jitter_spike_times=jitter_spike_times
    )

    # FIXME check this stuff
    ns_weights_per_block = np.array([x.shape[0] - 1 for x in movie_center_spikes_all_torch])
    ns_weights_per_block = ns_weights_per_block / np.sum(ns_weights_per_block)

    ################################################################
    wn_time_bins_all = ddu.multimovie_construct_wn_timebins(wn_movie_blocks,
                                                            bin_width_samples)

    wn_sel_overlaps = ddu.multimovie_compute_interval_overlaps_for_wn(wn_movie_blocks,
                                                                      wn_time_bins_all)

    wn_center_spikes_all_torch, wn_feedback_conv_all_torch = multimovie_wn_bin_spikes_precompute_feedback_convs(
        wn_movie_blocks,
        wn_time_bins_all,
        center_cell_wn,
        feedback_basis,
        device,
        jitter_time_amount=jitter_spike_times,
        prec_dtype=movie_spike_dtype,
        trim_spikes_seq=trim_spikes_seq
    )

    wn_coupling_conv_all_torch = multimovie_wn_bin_spikes_precompute_coupling_convs(
        wn_movie_blocks,
        wn_time_bins_all,
        coupled_cells,
        cells_ordered,
        coupling_basis,
        device,
        jitter_time_amount=jitter_spike_times,
        prec_dtype=movie_spike_dtype
    )

    # FIXME also check this stuff
    wn_weights_per_block_ = np.array([x.shape[0] - 1 for x in wn_center_spikes_all_torch])
    wn_weights_per_block = wn_weight * (wn_weights_per_block_ / np.sum(wn_weights_per_block_))

    movie_all_center_spikes_torch = [*movie_center_spikes_all_torch, *wn_center_spikes_all_torch]
    movie_all_feedback_conv_torch = [*movie_feedback_conv_all_torch, *wn_feedback_conv_all_torch]
    movie_all_coupling_conv_torch = [*movie_coupling_conv_all_torch, *wn_coupling_conv_all_torch]

    movie_all_weights_per_block = np.concatenate([ns_weights_per_block, wn_weights_per_block])

    # Step 3: do the alternating optimization
    # remember to clean up upsampled movies after each optimization problem
    # to save GPU space

    # (a) build the optimization modules
    n_basis_stim_spat, n_basis_stim_time = stim_spat_basis.shape[0], stim_time_basis.shape[0]
    n_basis_feedback = feedback_basis.shape[0]
    n_basis_coupling = coupling_basis.shape[0]
    n_coupled_cells = _count_coupled_cells(coupled_cells)

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

    timecourse_model = JointFlashedMultimovie_TimecourseFitGLM(
        n_basis_stim_time,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        loss_callable,
        group_sparse_reg_lambda=l21_group_sparse_lambda,
        stim_time_init_guess=timecourse_basis_repr,
        multimovie_weights=movie_all_weights_per_block,
        flashed_weight=flashed_image_relative_weight,
    ).to(device)

    spatial_model = JointFlashedMultimovie_SpatFitGLM(
        n_basis_stim_spat,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        loss_callable,
        group_sparse_reg_lambda=l21_group_sparse_lambda,
        spatial_sparsity_l1_lambda=spatial_sparsity_l1_lambda,
        multimovie_weights=movie_all_weights_per_block,
        flashed_weight=flashed_image_relative_weight
    ).to(device)

    prev_iter_loss = float('inf')
    for iter, spat_solver_params, time_solver_params in \
            zip(range(n_alternating_iters), solver_params_iter[0], solver_params_iter[1]):

        # (1, n_basis_stim_time) @ (n_basis_stim_time, n_bins_filter)
        # -> (1, n_bins_filter) -> (n_bins_filter, )
        with torch.no_grad(), autocast('cuda'):
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

        # pre-compute the application of the fixed timecourse
        # on the time component of the flashed stimulus
        with torch.no_grad(), autocast('cuda'):
            # shape (1, n_basis_time) @ (n_basis_time, n_bins - n_bins_filter + 1)
            # -> (1, n_bins - n_bins_filter + 1) -> (n_bins - n_bins_filter + 1, )
            flashed_filtered_stim_time = (timecourse_basis_repr @ flashed_timecourse_basis_conv).squeeze(0)

        # pre-compute the application of the spatial basis
        # with the fixed timecourse on the upsampled jittered movie
        # and store the result on GPU
        movie_precomputed_spatial_convolutions = multidata_precompute_spatial_convolutions(
            jittered_movie_blocks,
            movie_frame_sel_and_overlaps_all,
            stim_spat_basis,
            timecourse_waveform.detach().cpu().numpy(),
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        # pre-compute the application of the spatial basis
        # with the fixed timecourse on the upsampled WN movie
        # and store the results on GPU
        wn_precomputed_spatial_convolutions = multidata_wn_preapply_spatial_convolutions(
            wn_movie_blocks,
            wn_sel_overlaps,
            stim_spat_basis_imshape,
            timecourse_waveform.detach().cpu().numpy(),
            image_transform_lambda,
            device,
            prec_dtype=movie_spike_dtype
        )

        all_movie_precomputed_spatial_convolutions = [*movie_precomputed_spatial_convolutions,
                                                      *wn_precomputed_spatial_convolutions]

        _ = _optim_FISTA(spatial_model,
                         spat_solver_params,
                         all_movie_precomputed_spatial_convolutions,
                         movie_all_center_spikes_torch,
                         movie_all_feedback_conv_torch,
                         movie_all_coupling_conv_torch,
                         flashed_spat_basis_filt,
                         flashed_filtered_stim_time,
                         flashed_spikes,
                         flashed_feedback_basis_conv,
                         flashed_couple_basis_conv)

        # now do a clean up of precomputed_spatial_convolutions
        del movie_precomputed_spatial_convolutions, wn_precomputed_spatial_convolutions, \
            all_movie_precomputed_spatial_convolutions

        # this updates the feedback, coupling, bias, and auxiliary variables
        # stimulus filters are tracked separately
        timecourse_model.clone_parameters_model(spatial_model)
        timecourse_model.set_time_filter(timecourse_basis_repr)

        with torch.no_grad(), autocast('cuda'):
            # manually update the stimulus spatial filter
            # shape (1, n_basis_stim_spat)
            spat_filt_basis_repr = spatial_model.return_spat_filt_parameters()
            # shape (n_pixels, )
            spat_filt = (spat_filt_basis_repr @ stim_spat_basis_torch).squeeze(0)

        spat_filt_np = spat_filt.detach().cpu().numpy()

        # precompute application of the temporal basis with
        # known fixed spatial filter for jittered movie stimulus
        movie_precomputed_temp_convolutions = multidata_precompute_temporal_convolutions(
            jittered_movie_blocks,
            movie_frame_sel_and_overlaps_all,
            spat_filt_np,
            stim_time_basis,
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        # precompute application of the temporal basis with
        # known fixed spatial filter for WN stimulus
        wn_precomputed_temp_convolutions = multidata_wn_preapply_temporal_convolutions(
            wn_movie_blocks,
            wn_sel_overlaps,
            spat_filt_np.reshape(filt_height, filt_width),
            stim_time_basis,
            image_transform_lambda,
            device,
            prec_dtype=movie_spike_dtype
        )

        # precompute application of temporal basis wtih
        # known fixed spatial filter for the flashed stimulus
        with torch.no_grad(), autocast('cuda'):
            # shape (batch, n_spat_basis) @ (n_spat_basis, 1)
            # -> (batch, 1) -> (batch, )
            flashed_spatial_filter_applied = (flashed_spat_basis_filt @ spat_filt_basis_repr.T).squeeze(1)

        all_movie_precomputed_temp_convolutions = [*movie_precomputed_temp_convolutions,
                                                   *wn_precomputed_temp_convolutions]

        iter_loss = _optim_FISTA(timecourse_model,
                                 time_solver_params,
                                 all_movie_precomputed_temp_convolutions,
                                 movie_all_center_spikes_torch,
                                 movie_all_feedback_conv_torch,
                                 movie_all_coupling_conv_torch,
                                 flashed_spatial_filter_applied,
                                 flashed_timecourse_basis_conv,
                                 flashed_spikes,
                                 flashed_feedback_basis_conv,
                                 flashed_couple_basis_conv)

        # clean up of precomputed_temp_convolutions
        del all_movie_precomputed_temp_convolutions, movie_precomputed_temp_convolutions, \
            wn_precomputed_temp_convolutions

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
    del movie_all_center_spikes_torch, movie_all_feedback_conv_torch, movie_all_coupling_conv_torch
    del movie_center_spikes_all_torch, movie_feedback_conv_all_torch, movie_coupling_conv_all_torch
    del wn_center_spikes_all_torch, wn_coupling_conv_all_torch, wn_feedback_conv_all_torch
    del stim_spat_basis_torch, stim_time_basis_torch

    # return the parameters
    coupling_filters_w, feedback_filter_w, _, bias = timecourse_model.return_parameters_np()

    del timecourse_model, spatial_model

    spat_filt_basis_repr_np = spat_filt_basis_repr.detach().cpu().numpy()
    timecourse_basis_repr_np = timecourse_basis_repr.detach().cpu().numpy()

    torch.cuda.empty_cache()

    return prev_iter_loss, (
        coupling_filters_w, feedback_filter_w, spat_filt_basis_repr_np, timecourse_basis_repr_np, bias)


def new_style_wn_regularized_precompute_convs_and_fit_fb_only_glm(jittered_movie_blocks: List[LoadedBrownianMovieBlock],
                                                                  wn_movie_blocks: List[LoadedWNMovieBlock],
                                                                  wn_weight: float,
                                                                  stim_spat_basis_imshape: np.ndarray,
                                                                  stim_time_basis: np.ndarray,
                                                                  feedback_basis: np.ndarray,
                                                                  center_cell_wn: Tuple[str, int],
                                                                  cells_ordered: OrderedMatchedCellsStruct,
                                                                  bin_width_samples: int,
                                                                  image_transform_lambda: Callable[
                                                                      [torch.Tensor], torch.Tensor],
                                                                  loss_callable: Callable[
                                                                      [torch.Tensor, torch.Tensor], torch.Tensor],
                                                                  spatial_sparsity_l1_lambda: float,
                                                                  n_alternating_iters: int,
                                                                  alternating_opt_eps: float,
                                                                  solver_params_iter: Tuple[
                                                                      Iterator[ProxFISTASolverParams], Iterator[
                                                                          FistaSolverParams]],
                                                                  device: torch.device,
                                                                  initial_guess_spat: Optional[
                                                                      Union[torch.Tensor, np.ndarray]] = None,
                                                                  initial_guess_timecourse: Optional[
                                                                      Union[torch.Tensor, np.ndarray]] = None,
                                                                  trim_spikes_seq: int = 0,
                                                                  log_verbose_ascent: bool = False,
                                                                  movie_spike_dtype: torch.dtype = torch.float32,
                                                                  jitter_spike_times: float = 0.0) \
        -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    filt_height, filt_width = stim_spat_basis_imshape.shape[1:]
    stim_spat_basis = stim_spat_basis_imshape.reshape(stim_spat_basis_imshape.shape[0], -1)

    ###############################################################
    # Step 0: precompute spiking quantities for the NScenes
    # Also bin spikes, pre-compute spike train convolutions, and store those on GPU
    # Also compute frame overlaps, store those on GPU as well
    ns_timebins_all = ddu.multimovie_construct_natural_movies_timebins(jittered_movie_blocks,
                                                                       bin_width_samples)

    ns_frame_sel_and_overlaps_all = ddu.multimovie_compute_interval_overlaps(jittered_movie_blocks,
                                                                             ns_timebins_all)

    ns_center_spikes_all_torch, ns_feedback_conv_all_torch = multidata_bin_center_spikes_precompute_feedback_convs(
        jittered_movie_blocks,
        ns_timebins_all,
        feedback_basis,
        center_cell_wn,
        cells_ordered,
        device,
        trim_spikes_seq=trim_spikes_seq,
        jitter_spike_times=jitter_spike_times,
        prec_dtype=movie_spike_dtype
    )

    ns_weights_per_block = np.array([x.shape[0] - 1 for x in ns_center_spikes_all_torch])
    ns_weights_per_block = ns_weights_per_block / np.sum(ns_weights_per_block)

    ###############################################################
    # now do the same for the white noise
    wn_timebins_all = ddu.multimovie_construct_wn_timebins(wn_movie_blocks,
                                                           bin_width_samples)

    wn_sel_overlaps = ddu.multimovie_compute_interval_overlaps_for_wn(wn_movie_blocks,
                                                                      wn_timebins_all)

    wn_center_spikes_all_torch, wn_feedback_conv_all_torch = multimovie_wn_bin_spikes_precompute_feedback_convs(
        wn_movie_blocks,
        wn_timebins_all,
        center_cell_wn,
        feedback_basis,
        device,
        jitter_time_amount=jitter_spike_times,
        prec_dtype=movie_spike_dtype,
        trim_spikes_seq=trim_spikes_seq
    )

    wn_weights_per_block_ = np.array([x.shape[0] - 1 for x in wn_center_spikes_all_torch])
    wn_weights_per_block = wn_weight * (wn_weights_per_block_ / np.sum(wn_weights_per_block_))

    all_center_spikes_torch = [*ns_center_spikes_all_torch, *wn_center_spikes_all_torch]
    all_feedback_conv_torch = [*ns_feedback_conv_all_torch, *wn_feedback_conv_all_torch]

    all_weights_per_block = np.concatenate([ns_weights_per_block, wn_weights_per_block])

    # Step 3: do the alternating optimization
    # remember to clean up upsampled movies after each optimization problem
    # to save GPU space

    # (a) build the optimization modules
    n_basis_stim_spat, n_basis_stim_time = stim_spat_basis.shape[0], stim_time_basis.shape[0]
    n_basis_feedback = feedback_basis.shape[0]

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

    timecourse_model = NS_MM_Timecourse_FB_Only_GLM(
        n_basis_stim_time,
        n_basis_feedback,
        loss_callable,
        stim_time_init_guess=timecourse_basis_repr,
        multimovie_weights=all_weights_per_block
    ).to(device)

    spatial_model = NS_MM_Spatial_FB_Only_GLM(
        n_basis_stim_spat,
        n_basis_feedback,
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

        # pre-compute the application of the spatial basis
        # with the fixed timecourse on the upsampled jittered movie
        # and store the result on GPU
        ns_precomputed_spatial_convolutions = multidata_precompute_spatial_convolutions(
            jittered_movie_blocks,
            ns_frame_sel_and_overlaps_all,
            stim_spat_basis,
            timecourse_waveform.detach().cpu().numpy(),
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        # pre-compute the application of the spatial basis
        # with the fixed timecourse on the upsampled WN movie
        # and store the results on GPU
        wn_precomputed_spatial_convolutions = multidata_wn_preapply_spatial_convolutions(
            wn_movie_blocks,
            wn_sel_overlaps,
            stim_spat_basis_imshape,
            timecourse_waveform.detach().cpu().numpy(),
            image_transform_lambda,
            device,
            prec_dtype=movie_spike_dtype
        )

        precomputed_spatial_convolutions = [*ns_precomputed_spatial_convolutions, *wn_precomputed_spatial_convolutions]

        _ = _optim_FISTA(spatial_model,
                         spat_solver_params,
                         precomputed_spatial_convolutions,
                         all_center_spikes_torch,
                         all_feedback_conv_torch)

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
        ns_precomputed_temp_convolutions = multidata_precompute_temporal_convolutions(
            jittered_movie_blocks,
            ns_frame_sel_and_overlaps_all,
            spat_filt_np,
            stim_time_basis,
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        # precompute application of the temporal basis with
        # known fixed spatial filter for WN stimulus
        wn_precomputed_temp_convolutions = multidata_wn_preapply_temporal_convolutions(
            wn_movie_blocks,
            wn_sel_overlaps,
            spat_filt_np.reshape(filt_height, filt_width),
            stim_time_basis,
            image_transform_lambda,
            device,
            prec_dtype=movie_spike_dtype
        )

        all_precomputed_temp_convolutions = [*ns_precomputed_temp_convolutions, *wn_precomputed_temp_convolutions]

        iter_loss = _optim_unconstrained_FISTA(timecourse_model,
                                               time_solver_params,
                                               all_precomputed_temp_convolutions,
                                               all_center_spikes_torch,
                                               all_feedback_conv_torch)

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
    del all_center_spikes_torch, all_feedback_conv_torch
    del ns_center_spikes_all_torch, ns_feedback_conv_all_torch
    del wn_center_spikes_all_torch, wn_feedback_conv_all_torch
    del stim_spat_basis_torch, stim_time_basis_torch

    # return the parameters
    feedback_filter_w, _, bias = timecourse_model.return_parameters_np()

    del timecourse_model, spatial_model

    spat_filt_basis_repr_np = spat_filt_basis_repr.detach().cpu().numpy()
    timecourse_basis_repr_np = timecourse_basis_repr.detach().cpu().numpy()

    torch.cuda.empty_cache()

    return prev_iter_loss, (
        feedback_filter_w, spat_filt_basis_repr_np, timecourse_basis_repr_np, bias)


def new_style_joint_flashed_jittered_precompute_convs_and_fit_FB_only_glm(
        jittered_movie_blocks: List[LoadedBrownianMovieBlock],
        flashed_stimulus_block: List[LoadedFlashPatch],
        flashed_image_relative_weight: float,
        wn_movie_blocks: List[LoadedWNMovieBlock],
        wn_weight: float,
        stim_spat_basis_imshape: np.ndarray,
        stim_time_basis: np.ndarray,
        feedback_basis: np.ndarray,
        center_cell_wn: Tuple[str, int],
        cells_ordered: OrderedMatchedCellsStruct,
        bin_width_samples: int,
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
        trim_spikes_seq: int = 0,
        log_verbose_ascent: bool = False,
        movie_spike_dtype: torch.dtype = torch.float32,
        jitter_spike_times: float = 0.0) \
        -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    '''

    :param jittered_movie_blocks:
    :param flashed_stimulus_block:
    :param flashed_image_relative_weight:
    :param wn_movie_blocks:
    :param wn_weight:
    :param stim_spat_basis_imshape:
    :param stim_time_basis:
    :param feedback_basis:
    :param center_cell_wn:
    :param cells_ordered:
    :param bin_width_samples:
    :param image_transform_lambda:
    :param loss_callable:
    :param spatial_sparsity_l1_lambda:
    :param n_alternating_iters:
    :param alternating_opt_eps:
    :param solver_params_iter:
    :param device:
    :param initial_guess_spat:
    :param initial_guess_timecourse:
    :param trim_spikes_seq:
    :param log_verbose_ascent:
    :param movie_spike_dtype:
    :param jitter_spike_times:
    :return:
    '''

    filt_height, filt_width = stim_spat_basis_imshape.shape[1:]
    stim_spat_basis = stim_spat_basis_imshape.reshape(stim_spat_basis_imshape.shape[0], -1)

    # precompute quantitites for the flashed natural images
    flashed_spikes, flashed_feedback_basis_conv = flashed_ns_bin_spikes_precompute_feedback_convs2(
        flashed_stimulus_block,
        feedback_basis,
        center_cell_wn,
        cells_ordered,
        device,
        jitter_time_amount=jitter_spike_times,
        prec_dtype=movie_spike_dtype,
        trim_spikes_seq=trim_spikes_seq
    )

    flashed_timecourse_basis_conv = flashed_ns_bin_spikes_precompute_timecourse_basis_conv2(
        flashed_stimulus_block,
        stim_time_basis,
        device,
        prec_dtype=movie_spike_dtype
    )

    flashed_spat_basis_filt = flashed_ns_precompute_spatial_basis(
        flashed_stimulus_block,
        stim_spat_basis,
        image_transform_lambda,
        device,
        prec_dtype=movie_spike_dtype)

    # then precompute quantities for the jittered stimuli
    movie_timebins_all = ddu.multimovie_construct_natural_movies_timebins(jittered_movie_blocks,
                                                                          bin_width_samples)

    movie_frame_sel_and_overlaps_all = ddu.multimovie_compute_interval_overlaps(jittered_movie_blocks,
                                                                                movie_timebins_all)

    movie_center_spikes_all_torch, movie_feedback_conv_all_torch = multidata_bin_center_spikes_precompute_feedback_convs(
        jittered_movie_blocks,
        movie_timebins_all,
        feedback_basis,
        center_cell_wn,
        cells_ordered,
        device,
        trim_spikes_seq=trim_spikes_seq,
        jitter_spike_times=jitter_spike_times,
        prec_dtype=movie_spike_dtype
    )

    # FIXME check this stuff
    ns_weights_per_block = np.array([x.shape[0] - 1 for x in movie_center_spikes_all_torch])
    ns_weights_per_block = ns_weights_per_block / np.sum(ns_weights_per_block)

    ################################################################
    wn_time_bins_all = ddu.multimovie_construct_wn_timebins(wn_movie_blocks,
                                                            bin_width_samples)

    wn_sel_overlaps = ddu.multimovie_compute_interval_overlaps_for_wn(wn_movie_blocks,
                                                                      wn_time_bins_all)

    wn_center_spikes_all_torch, wn_feedback_conv_all_torch = multimovie_wn_bin_spikes_precompute_feedback_convs(
        wn_movie_blocks,
        wn_time_bins_all,
        center_cell_wn,
        feedback_basis,
        device,
        jitter_time_amount=jitter_spike_times,
        prec_dtype=movie_spike_dtype,
        trim_spikes_seq=trim_spikes_seq
    )

    # FIXME also check this stuff
    wn_weights_per_block_ = np.array([x.shape[0] - 1 for x in wn_center_spikes_all_torch])
    wn_weights_per_block = wn_weight * (wn_weights_per_block_ / np.sum(wn_weights_per_block_))

    movie_all_center_spikes_torch = [*movie_center_spikes_all_torch, *wn_center_spikes_all_torch]
    movie_all_feedback_conv_torch = [*movie_feedback_conv_all_torch, *wn_feedback_conv_all_torch]

    movie_all_weights_per_block = np.concatenate([ns_weights_per_block, wn_weights_per_block])

    # Step 3: do the alternating optimization
    # remember to clean up upsampled movies after each optimization problem
    # to save GPU space

    # (a) build the optimization modules
    n_basis_stim_spat, n_basis_stim_time = stim_spat_basis.shape[0], stim_time_basis.shape[0]
    n_basis_feedback = feedback_basis.shape[0]

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

    timecourse_model = JointFlashedMultimovie_TimecourseFit_FeedbackOnlyGLM(
        n_basis_stim_time,
        n_basis_feedback,
        loss_callable,
        stim_time_init_guess=timecourse_basis_repr,
        multimovie_weights=movie_all_weights_per_block,
        flashed_weight=flashed_image_relative_weight,
    ).to(device)

    spatial_model = JointFlashedMultimovie_SpatFit_FeedbackOnlyGLM(
        n_basis_stim_spat,
        n_basis_feedback,
        loss_callable,
        spatial_sparsity_l1_lambda=spatial_sparsity_l1_lambda,
        multimovie_weights=movie_all_weights_per_block,
        flashed_weight=flashed_image_relative_weight
    ).to(device)

    prev_iter_loss = float('inf')
    for iter, spat_solver_params, time_solver_params in \
            zip(range(n_alternating_iters), solver_params_iter[0], solver_params_iter[1]):

        # (1, n_basis_stim_time) @ (n_basis_stim_time, n_bins_filter)
        # -> (1, n_bins_filter) -> (n_bins_filter, )
        with torch.no_grad(), autocast('cuda'):
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

        # pre-compute the application of the fixed timecourse
        # on the time component of the flashed stimulus
        with torch.no_grad(), autocast('cuda'):
            # shape (1, n_basis_time) @ (n_basis_time, n_bins - n_bins_filter + 1)
            # -> (1, n_bins - n_bins_filter + 1) -> (n_bins - n_bins_filter + 1, )
            flashed_filtered_stim_time = (timecourse_basis_repr @ flashed_timecourse_basis_conv).squeeze(0)

        # pre-compute the application of the spatial basis
        # with the fixed timecourse on the upsampled jittered movie
        # and store the result on GPU
        movie_precomputed_spatial_convolutions = multidata_precompute_spatial_convolutions(
            jittered_movie_blocks,
            movie_frame_sel_and_overlaps_all,
            stim_spat_basis,
            timecourse_waveform.detach().cpu().numpy(),
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        # pre-compute the application of the spatial basis
        # with the fixed timecourse on the upsampled WN movie
        # and store the results on GPU
        wn_precomputed_spatial_convolutions = multidata_wn_preapply_spatial_convolutions(
            wn_movie_blocks,
            wn_sel_overlaps,
            stim_spat_basis_imshape,
            timecourse_waveform.detach().cpu().numpy(),
            image_transform_lambda,
            device,
            prec_dtype=movie_spike_dtype
        )

        all_movie_precomputed_spatial_convolutions = [*movie_precomputed_spatial_convolutions,
                                                      *wn_precomputed_spatial_convolutions]

        _ = _optim_FISTA(spatial_model,
                         spat_solver_params,
                         all_movie_precomputed_spatial_convolutions,
                         movie_all_center_spikes_torch,
                         movie_all_feedback_conv_torch,
                         flashed_spat_basis_filt,
                         flashed_filtered_stim_time,
                         flashed_spikes,
                         flashed_feedback_basis_conv)

        # now do a clean up of precomputed_spatial_convolutions
        del movie_precomputed_spatial_convolutions, wn_precomputed_spatial_convolutions, \
            all_movie_precomputed_spatial_convolutions

        # this updates the feedback, coupling, bias, and auxiliary variables
        # stimulus filters are tracked separately
        timecourse_model.clone_parameters_model(spatial_model)
        timecourse_model.set_time_filter(timecourse_basis_repr)

        with torch.no_grad(), autocast('cuda'):
            # manually update the stimulus spatial filter
            # shape (1, n_basis_stim_spat)
            spat_filt_basis_repr = spatial_model.return_spat_filt_parameters()
            # shape (n_pixels, )
            spat_filt = (spat_filt_basis_repr @ stim_spat_basis_torch).squeeze(0)

        spat_filt_np = spat_filt.detach().cpu().numpy()

        # precompute application of the temporal basis with
        # known fixed spatial filter for jittered movie stimulus
        movie_precomputed_temp_convolutions = multidata_precompute_temporal_convolutions(
            jittered_movie_blocks,
            movie_frame_sel_and_overlaps_all,
            spat_filt_np,
            stim_time_basis,
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        # precompute application of the temporal basis with
        # known fixed spatial filter for WN stimulus
        wn_precomputed_temp_convolutions = multidata_wn_preapply_temporal_convolutions(
            wn_movie_blocks,
            wn_sel_overlaps,
            spat_filt_np.reshape(filt_height, filt_width),
            stim_time_basis,
            image_transform_lambda,
            device,
            prec_dtype=movie_spike_dtype
        )

        # precompute application of temporal basis wtih
        # known fixed spatial filter for the flashed stimulus
        with torch.no_grad(), autocast('cuda'):
            # shape (batch, n_spat_basis) @ (n_spat_basis, 1)
            # -> (batch, 1) -> (batch, )
            flashed_spatial_filter_applied = (flashed_spat_basis_filt @ spat_filt_basis_repr.T).squeeze(1)

        all_movie_precomputed_temp_convolutions = [*movie_precomputed_temp_convolutions,
                                                   *wn_precomputed_temp_convolutions]

        iter_loss = _optim_unconstrained_FISTA(timecourse_model,
                                               time_solver_params,
                                               all_movie_precomputed_temp_convolutions,
                                               movie_all_center_spikes_torch,
                                               movie_all_feedback_conv_torch,
                                               flashed_spatial_filter_applied,
                                               flashed_timecourse_basis_conv,
                                               flashed_spikes,
                                               flashed_feedback_basis_conv)

        # clean up of precomputed_temp_convolutions
        del all_movie_precomputed_temp_convolutions, movie_precomputed_temp_convolutions, \
            wn_precomputed_temp_convolutions

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
    del movie_all_center_spikes_torch, movie_all_feedback_conv_torch
    del movie_center_spikes_all_torch, movie_feedback_conv_all_torch
    del wn_center_spikes_all_torch, wn_feedback_conv_all_torch
    del stim_spat_basis_torch, stim_time_basis_torch

    # return the parameters
    feedback_filter_w, _, bias = timecourse_model.return_parameters_np()

    del timecourse_model, spatial_model

    spat_filt_basis_repr_np = spat_filt_basis_repr.detach().cpu().numpy()
    timecourse_basis_repr_np = timecourse_basis_repr.detach().cpu().numpy()

    torch.cuda.empty_cache()

    return prev_iter_loss, (
        feedback_filter_w, spat_filt_basis_repr_np, timecourse_basis_repr_np, bias)


def new_style_repeats_precompute_convs_and_fit_glm(
        repeats_training_blocks: List[ddu.RepeatBrownianTrainingBlock],
        stim_spat_basis_imshape: np.ndarray,
        stim_time_basis: np.ndarray,
        feedback_basis: np.ndarray,
        coupling_basis: np.ndarray,
        center_cell_wn: Tuple[str, int],
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        stimulus_cropping_bounds: Tuple[Tuple[int, int], Tuple[int, int]],
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        l21_group_sparse_lambda: float,
        spatial_sparsity_l1_lambda: float,
        n_alternating_iters: int,
        alternating_opt_eps: float,
        solver_params_iter: Tuple[
            Iterator[ProxFISTASolverParams], Iterator[ProxFISTASolverParams]],
        device: torch.device,
        initial_guess_spat: Optional[Union[torch.Tensor, np.ndarray]] = None,
        initial_guess_timecourse: Optional[
            Union[torch.Tensor, np.ndarray]] = None,
        trim_spikes_seq: int = 0,
        log_verbose_ascent: bool = False,
        movie_spike_dtype: torch.dtype = torch.float32) \
        -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    '''

    :param repeats_training_blocks:
    :param stim_spat_basis_imshape:
    :param stim_time_basis:
    :param feedback_basis:
    :param coupling_basis:
    :param center_cell_wn:
    :param coupled_cells:
    :param cells_ordered:
    :param stimulus_cropping_bounds:
    :param image_transform_lambda:
    :param loss_callable:
    :param l21_group_sparse_lambda:
    :param spatial_sparsity_l1_lambda:
    :param n_alternating_iters:
    :param alternating_opt_eps:
    :param solver_params_iter:
    :param device:
    :param initial_guess_spat:
    :param initial_guess_timecourse:
    :param trim_spikes_seq:
    :param log_verbose_ascent:
    :param movie_spike_dtype:
    :return:
    '''

    ns_frame_sel_and_overlaps_all = ddu.repeat_training_compute_interval_overlaps(repeats_training_blocks)

    ns_center_spikes_all_torch, ns_feedback_conv_all_torch = repeats_blocks_extract_center_spikes_precompute_feedback_convs(
        repeats_training_blocks,
        feedback_basis,
        center_cell_wn,
        cells_ordered,
        device,
        trim_spikes_seq=trim_spikes_seq,
        prec_dtype=movie_spike_dtype
    )

    ns_coupling_conv_all_torch = repeats_blocks_precompute_coupling_convs(
        repeats_training_blocks,
        coupling_basis,
        coupled_cells,
        cells_ordered,
        device,
        prec_dtype=movie_spike_dtype
    )

    ns_weights_per_block = np.array([x.shape[0] - 1 for x in ns_center_spikes_all_torch])
    ns_weights_per_block = ns_weights_per_block / np.sum(ns_weights_per_block)

    # do the alternating optimization
    # remember to clean up upsampled movies after each optimization problem
    # to save GPU space

    # (a) build the optimization modules
    n_basis_stim_spat, n_basis_stim_time = stim_spat_basis_imshape.shape[0], stim_time_basis.shape[0]
    n_basis_feedback = feedback_basis.shape[0]
    n_basis_coupling = coupling_basis.shape[0]
    n_coupled_cells = _count_coupled_cells(coupled_cells)

    # shape (batch, n_pix)
    stim_spat_basis = stim_spat_basis_imshape.reshape(n_basis_stim_spat, -1)

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

    timecourse_model = NS_MM_Timecourse_GroupSparseLRCroppedCT_GLM(
        n_basis_stim_time,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        loss_callable,
        l21_group_sparse_lambda,
        stim_time_init_guess=timecourse_basis_repr,
        multimovie_weights=ns_weights_per_block
    ).to(device)

    spatial_model = NS_MM_Spatial_GroupSparseLRCroppedCT_GLM(
        n_basis_stim_spat,
        n_basis_feedback,
        n_basis_coupling,
        n_coupled_cells,
        loss_callable,
        l21_group_sparse_lambda,
        spatial_sparsity_l1_lambda=spatial_sparsity_l1_lambda,
        stim_spat_init_guess=initial_guess_spat,
        multimovie_weights=ns_weights_per_block
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

        # pre-compute the application of the spatial basis
        # with the fixed timecourse on the upsapmled jittered movie
        # and put the results on GPU
        ns_precomputed_spatial_convolutions = repeats_blocks_precompute_spatial_convolutions(
            repeats_training_blocks,
            ns_frame_sel_and_overlaps_all,
            stim_spat_basis,
            timecourse_waveform.detach().cpu().numpy(),
            stimulus_cropping_bounds,
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        _ = _optim_FISTA(spatial_model,
                         spat_solver_params,
                         ns_precomputed_spatial_convolutions,
                         ns_center_spikes_all_torch,
                         ns_feedback_conv_all_torch,
                         ns_coupling_conv_all_torch)

        del ns_precomputed_spatial_convolutions

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

        ns_precomputed_temp_convolutions = repeats_blocks_precompute_temporal_convolutions(
            repeats_training_blocks,
            ns_frame_sel_and_overlaps_all,
            spat_filt_np,
            stim_time_basis,
            stimulus_cropping_bounds,
            image_transform_lambda,
            device,
            dtype=movie_spike_dtype
        )

        iter_loss = _optim_FISTA(timecourse_model,
                                 time_solver_params,
                                 ns_precomputed_temp_convolutions,
                                 ns_center_spikes_all_torch,
                                 ns_feedback_conv_all_torch,
                                 ns_coupling_conv_all_torch)

        del ns_precomputed_temp_convolutions

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
    del ns_center_spikes_all_torch, ns_feedback_conv_all_torch, ns_coupling_conv_all_torch
    del stim_spat_basis_torch, stim_time_basis_torch

    # return the parameters
    coupling_filters_w, feedback_filter_w, _, bias = timecourse_model.return_parameters_np()

    del timecourse_model, spatial_model

    spat_filt_basis_repr_np = spat_filt_basis_repr.detach().cpu().numpy()
    timecourse_basis_repr_np = timecourse_basis_repr.detach().cpu().numpy()

    torch.cuda.empty_cache()

    return prev_iter_loss, (
        coupling_filters_w, feedback_filter_w, spat_filt_basis_repr_np, timecourse_basis_repr_np, bias)
