from typing import Callable, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from movie_upsampling import compute_interval_overlaps, flat_sparse_upsample_transpose_cuda, flat_sparse_upsample_cuda

import lib.data_utils.dynamic_data_util as ddu
from glm_precompute.ct_glm_precompute import precompute_feedback_convolutions, precompute_coupling_convolutions
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.movie_processing import multiresolution_spatial_basis_application


def preapply_spatial_basis_to_wn(
        cropped_wn_bw_stimulus: np.ndarray,
        stimulus_rescale_lambda: Callable[[torch.Tensor], torch.Tensor],
        spatial_basis_imshape: np.ndarray,
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32):
    '''

    :param cropped_wn_bw_stimulus: shape (n_frames, height, width)
    :param stimulus_rescale_lambda:
    :param spatial_basis_imshape:
    :param device:
    :param prec_dtype:
    :return:
    '''

    with torch.no_grad():
        wn_cropped_stimulus_bw_torch_unscaled = torch.tensor(cropped_wn_bw_stimulus,
                                                             dtype=torch.float32, device=device)
        wn_cropped_stimulus_bw_torch = stimulus_rescale_lambda(wn_cropped_stimulus_bw_torch_unscaled).to(prec_dtype)
        spat_spline_basis_imshape_torch = torch.tensor(spatial_basis_imshape, dtype=prec_dtype, device=device)

        # shape (n_frames, n_spat_basis)
        mres_apply_basis_shape = multiresolution_spatial_basis_application(
            wn_cropped_stimulus_bw_torch,
            spat_spline_basis_imshape_torch).to(prec_dtype)

        del wn_cropped_stimulus_bw_torch, wn_cropped_stimulus_bw_torch_unscaled
        del spat_spline_basis_imshape_torch

    return mres_apply_basis_shape


def preapply_spatial_basis_to_wn_and_temporally_upsample(
        cropped_wn_bw_stimulus: np.ndarray,
        wn_frame_transition_times: np.ndarray,
        wn_bin_edges: np.ndarray,
        stimulus_rescale_lambda: Callable[[torch.Tensor], torch.Tensor],
        spatial_basis_imshape: np.ndarray,
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32):
    '''

    :param cropped_wn_bw_stimulus: shape (n_frames, height, width)
    :param wn_frame_transition_times: shape (n_frames + 1, )
    :param wn_bin_edges: shape (n_bins + 1, )
    :param spatial_basis_imshape: shape (n_basis, basis_height, basis_width)
    :param prec_dtype:
    :return:
    '''

    sel_a, weight_a = compute_interval_overlaps(wn_frame_transition_times, wn_bin_edges)

    with torch.no_grad():
        # shape (n_frames, n_spat_basis)
        mres_apply_basis_shape = preapply_spatial_basis_to_wn(
            cropped_wn_bw_stimulus,
            stimulus_rescale_lambda,
            spatial_basis_imshape,
            device,
            prec_dtype=prec_dtype
        )

        # shape (n_bins, 2)
        selection_ix_torch = torch.tensor(sel_a, dtype=torch.long, device=device)
        overlap_bin_torch = torch.tensor(weight_a, dtype=prec_dtype, device=device)

        del sel_a, weight_a

        # shape (n_spat_basis, n_bins)
        wn_movie_upsampled_flat_torch = flat_sparse_upsample_transpose_cuda(mres_apply_basis_shape,
                                                                            selection_ix_torch,
                                                                            overlap_bin_torch)

        del mres_apply_basis_shape
        del selection_ix_torch, overlap_bin_torch

        return wn_movie_upsampled_flat_torch


def wn_preapply_temporal_convolutions(wn_movie_block: ddu.LoadedWNMovieBlock,
                                      bin_interval_sel_overlap: Tuple[np.ndarray, np.ndarray],
                                      fixed_spatial_filter_imshape: np.ndarray,
                                      timecourse_basis: np.ndarray,
                                      stimulus_rescale_lambda: Callable[[torch.Tensor], torch.Tensor],
                                      device: torch.device,
                                      prec_dtype: torch.dtype = torch.float32,
                                      store_intermediate_on_cpu: bool = False) -> torch.Tensor:
    '''

    :param wn_movie_block:
    :param bin_interval_sel_overlap:
    :param fixed_spatial_filter_imshape: shape (height, width)
    :param timecourse_basis:
    :param stimulus_rescale_lambda:
    :param device:
    :param prec_dtype:
    :return:
    '''
    with torch.no_grad():
        wn_cropped_stimulus_bw_torch_unscaled = torch.tensor(wn_movie_block.stimulus_frame_patches_wn_resolution,
                                                             dtype=prec_dtype, device=device)  # FIXME was torch.float32
        # not sure if change breaks AMP or not
        wn_cropped_stimulus_bw_torch = stimulus_rescale_lambda(wn_cropped_stimulus_bw_torch_unscaled).to(prec_dtype)
        spatial_filter_imshape_torch = torch.tensor(fixed_spatial_filter_imshape, dtype=prec_dtype, device=device)

        # shape (n_frames, )
        mres_apply_basis_shape = multiresolution_spatial_basis_application(
            wn_cropped_stimulus_bw_torch,
            spatial_filter_imshape_torch[None, :, :]).squeeze(1)

        del wn_cropped_stimulus_bw_torch, wn_cropped_stimulus_bw_torch_unscaled
        del spatial_filter_imshape_torch

        # shape (n_bins, 2)
        sel_a, weight_a = bin_interval_sel_overlap
        selection_ix_torch = torch.tensor(sel_a, dtype=torch.long, device=device)
        overlap_bin_torch = torch.tensor(weight_a, dtype=prec_dtype, device=device)

        # shape (n_bins, )
        upsampled_timedomain = flat_sparse_upsample_cuda(mres_apply_basis_shape[:, None],
                                                         selection_ix_torch,
                                                         overlap_bin_torch).squeeze(1)

        timecourse_basis_torch = torch.tensor(timecourse_basis, dtype=prec_dtype, device=device)

        # shape (n_basis_timecourse, n_bins - n_bins_filter + 1)
        conv_output = F.conv1d(upsampled_timedomain[None, None, :],
                               timecourse_basis_torch[:, None, :]).squeeze(0)

        del timecourse_basis_torch

        if store_intermediate_on_cpu:
            return conv_output.detach().cpu()
        return conv_output


def multidata_wn_preapply_temporal_convolutions(
        wn_movie_blocks: List[ddu.LoadedWNMovieBlock],
        bin_interval_sel_overlaps: List[Tuple[np.ndarray, np.ndarray]],
        fixed_spatial_filter_imshape: np.ndarray,
        timecourse_basis: np.ndarray,
        stimulus_rescale_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32,
        store_intermediate_on_cpu: bool = False) -> List[torch.Tensor]:
    return [wn_preapply_temporal_convolutions(a, b, fixed_spatial_filter_imshape, timecourse_basis,
                                              stimulus_rescale_lambda, device, prec_dtype=prec_dtype,
                                              store_intermediate_on_cpu=store_intermediate_on_cpu)
            for a, b, in zip(wn_movie_blocks, bin_interval_sel_overlaps)]


def wn_preapply_spatial_convolutions(wn_movie_block: ddu.LoadedWNMovieBlock,
                                     bin_interval_sel_overlap: Tuple[np.ndarray, np.ndarray],
                                     spatial_basis_imshape: np.ndarray,
                                     fixed_timecourse: np.ndarray,
                                     stimulus_rescale_lambda: Callable[[torch.Tensor], torch.Tensor],
                                     device: torch.device,
                                     prec_dtype: torch.dtype = torch.float32,
                                     store_intermediate_on_cpu: bool = False) -> torch.Tensor:
    with torch.no_grad():
        wn_cropped_stimulus_bw_torch_unscaled = torch.tensor(wn_movie_block.stimulus_frame_patches_wn_resolution,
                                                             dtype=prec_dtype, device=device)  # FIXME was torch.float32
        # don't know whether the change breaks AMP or not
        wn_cropped_stimulus_bw_torch = stimulus_rescale_lambda(wn_cropped_stimulus_bw_torch_unscaled).to(prec_dtype)
        spat_spline_basis_imshape_torch = torch.tensor(spatial_basis_imshape, dtype=prec_dtype, device=device)

        # shape (n_frames, n_spat_basis)
        mres_apply_basis_shape = multiresolution_spatial_basis_application(
            wn_cropped_stimulus_bw_torch,
            spat_spline_basis_imshape_torch).to(prec_dtype)

        del wn_cropped_stimulus_bw_torch, wn_cropped_stimulus_bw_torch_unscaled
        del spat_spline_basis_imshape_torch

        # shape (n_bins, 2)
        sel_a, weight_a = bin_interval_sel_overlap
        selection_ix_torch = torch.tensor(sel_a, dtype=torch.long, device=device)
        overlap_bin_torch = torch.tensor(weight_a, dtype=prec_dtype, device=device)

        # shape (n_spat_basis, n_bins)
        wn_movie_upsampled_flat_torch = flat_sparse_upsample_transpose_cuda(mres_apply_basis_shape,
                                                                            selection_ix_torch,
                                                                            overlap_bin_torch)
        torch.cuda.synchronize()

        # apply the time-domain convolution
        timecourse_filter_torch = torch.tensor(fixed_timecourse, dtype=prec_dtype, device=device)
        wn_upsampled_timecourse_convolved = F.conv1d(
            wn_movie_upsampled_flat_torch[:, None, :],
            timecourse_filter_torch[None, None, :]).squeeze(1)

        del wn_movie_upsampled_flat_torch
        del mres_apply_basis_shape
        del selection_ix_torch, overlap_bin_torch

        if store_intermediate_on_cpu:
            return wn_upsampled_timecourse_convolved.detach().cpu()
        return wn_upsampled_timecourse_convolved


def multidata_wn_preapply_spatial_convolutions(wn_movie_block: List[ddu.LoadedWNMovieBlock],
                                               bin_interval_sel_overlaps: List[Tuple[np.ndarray, np.ndarray]],
                                               spatial_basis_imshape: np.ndarray,
                                               fixed_timecourse: np.ndarray,
                                               stimulus_rescale_lambda: Callable[[torch.Tensor], torch.Tensor],
                                               device: torch.device,
                                               prec_dtype: torch.dtype = torch.float32,
                                               store_intermediate_on_cpu: bool = False) \
        -> List[torch.Tensor]:
    return [
        wn_preapply_spatial_convolutions(
            a, b, spatial_basis_imshape, fixed_timecourse, stimulus_rescale_lambda, device,
            prec_dtype=prec_dtype, store_intermediate_on_cpu=store_intermediate_on_cpu) for (a, b) in
        zip(wn_movie_block, bin_interval_sel_overlaps)]


def wn_bin_spikes_precompute_feedback_convs(
        wn_movie_block: ddu.LoadedWNMovieBlock,
        spike_time_bins: np.ndarray,
        center_cell_wn: Tuple[str, int],
        feedback_basis: np.ndarray,
        device: torch.device,
        jitter_time_amount: float = 0.0,
        prec_dtype: torch.dtype = torch.float32,
        trim_spikes_seq: int = 0) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''

    :param wn_movie_block:
    :param spike_time_bins: spike bin edges, shape (n_bins + 1, )
    :param center_cell_wn:
    :param feedback_basis: feedback basis, shape (n_feedback_basis, n_bins_filter)
    :param device:
    :param jitter_time_amount: float, default=0.0, standard deviation of Gaussian (in units of electrical samples)
        to jitter the recorded spike times by
    :param prec_dtype:
    :param trim_spikes_seq: how many samples to trim off of the beginning of the
        binned center cell spikes, default 0 corresponding to no trim
    :return:
    '''
    center_cell_type, center_cell_id = center_cell_wn
    wn_center_cell_spikes = ddu.movie_bin_spikes_multiple_cells2(
        wn_movie_block.vision_dataset,
        [[center_cell_id, ], ],
        spike_time_bins,
        jitter_time_amount=jitter_time_amount).squeeze(0)

    with torch.no_grad():
        feedback_basis_torch = torch.tensor(feedback_basis, device=device, dtype=prec_dtype)

        # shape (n_bins, )
        wn_center_spikes_torch = torch.tensor(wn_center_cell_spikes, device=device, dtype=prec_dtype)

        del wn_center_cell_spikes

        wn_feedback_basis_applied = precompute_feedback_convolutions(wn_center_spikes_torch,
                                                                     feedback_basis_torch)
        del feedback_basis_torch

        if trim_spikes_seq != 0:
            wn_center_spikes_torch = wn_center_spikes_torch[trim_spikes_seq:].contiguous()

    return wn_center_spikes_torch, wn_feedback_basis_applied


def multimovie_wn_bin_spikes_precompute_feedback_convs(
        wn_movie_blocks: List[ddu.LoadedWNMovieBlock],
        multimovie_spike_time_bins: List[np.ndarray],
        center_cell_wn: Tuple[str, int],
        feedback_basis: np.ndarray,
        device: torch.device,
        jitter_time_amount: float = 0.0,
        prec_dtype: torch.dtype = torch.float32,
        trim_spikes_seq: int = 0) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    '''

    :param wn_movie_blocks:
    :param multimovie_spike_time_bins:
    :param center_cell_wn:
    :param feedback_basis:
    :param device:
    :param jitter_time_amount:
    :param prec_dtype:
    :param trim_spikes_seq:
    :return:
    '''

    combined = [wn_bin_spikes_precompute_feedback_convs(a, b, center_cell_wn, feedback_basis, device,
                                                        jitter_time_amount=jitter_time_amount,
                                                        prec_dtype=prec_dtype, trim_spikes_seq=trim_spikes_seq)
                for a, b in zip(wn_movie_blocks, multimovie_spike_time_bins)]

    center_spikes_list = [a[0] for a in combined]
    fb_convolved_list = [a[1] for a in combined]

    return center_spikes_list, fb_convolved_list


def wn_bin_spikes_precompute_coupling_convs(
        wn_movie_block: ddu.LoadedWNMovieBlock,
        spike_time_bins: np.ndarray,
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        coupling_basis: np.ndarray,
        device: torch.device,
        jitter_time_amount: float = 0.0,
        prec_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    '''

    :param wn_movie_block:
    :param spike_time_bins:
    :param coupled_cells:
    :param cells_ordered:
    :param coupling_basis:
    :param device:
    :param jitter_time_amount:
    :param prec_dtype:
    :return:
    '''

    wn_coupled_spikes = ddu.timebin_wn_movie_coupled_cell_spikes(wn_movie_block, spike_time_bins, coupled_cells,
                                                                 cells_ordered, jitter_time_amount=jitter_time_amount)

    with torch.no_grad():
        coupling_basis_torch = torch.tensor(coupling_basis, device=device, dtype=prec_dtype)

        # shape (n_coupled_cells, n_bins)
        wn_coupling_spikes_torch = torch.tensor(wn_coupled_spikes, device=device, dtype=prec_dtype)

        del wn_coupled_spikes

        wn_coupling_basis_applied = precompute_coupling_convolutions(wn_coupling_spikes_torch,
                                                                     coupling_basis_torch)

        del wn_coupling_spikes_torch, coupling_basis_torch

    return wn_coupling_basis_applied


def multimovie_wn_bin_spikes_precompute_coupling_convs(
        wn_movie_blocks: List[ddu.LoadedWNMovieBlock],
        multimovie_spike_time_bins: List[np.ndarray],
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        coupling_basis: np.ndarray,
        device: torch.device,
        jitter_time_amount: float = 0.0,
        prec_dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    return [wn_bin_spikes_precompute_coupling_convs(a, b, coupled_cells, cells_ordered, coupling_basis, device,
                                                    jitter_time_amount=jitter_time_amount, prec_dtype=prec_dtype)
            for a, b, in zip(wn_movie_blocks, multimovie_spike_time_bins)]
