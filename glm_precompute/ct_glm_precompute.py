from typing import List, Tuple, Dict, Callable

import numpy as np
import torch
from torch.nn import functional as F

from lib.data_utils import dynamic_data_util as ddu
from lib.data_utils.dynamic_data_util import LoadedBrownianMovieBlock, interpolate_frame_transition_times2
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct

from movie_upsampling import compute_interval_overlaps, movie_sparse_upsample_transpose_cuda, \
    flat_sparse_upsample_transpose_cuda


def multidata_bin_center_spikes(
        jittered_movie_blocks: List[ddu.LoadedBrownianMovieBlock],
        bin_times_per_block: List[np.ndarray],
        center_cell_wn: Tuple[str, int],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32,
        jitter_spike_times: float = 0.0) -> List[torch.Tensor]:
    center_spike_list = []
    for jittered_movie_block, spike_bin_edges in zip(jittered_movie_blocks, bin_times_per_block):
        binned_center_spikes = ddu.timebin_natural_movies_center_cell_spikes(
            jittered_movie_block.vision_dataset,
            jittered_movie_block.vision_name,
            cells_ordered,
            center_cell_wn,
            spike_bin_edges,
            jitter_spike_times=jitter_spike_times
        )

        center_spikes_torch = torch.tensor(binned_center_spikes, dtype=prec_dtype, device=device)
        center_spike_list.append(center_spikes_torch)

    return center_spike_list


def multidata_bin_coupling_spikes(
        jittered_movie_blocks: List[LoadedBrownianMovieBlock],
        bin_times_per_block: List[np.ndarray],
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        prec_dtype=torch.float32,
        jitter_spike_times: float = 0.0) -> List[torch.Tensor]:
    coupling_list = []

    for jittered_movie_block, spike_bin_edges in zip(jittered_movie_blocks, bin_times_per_block):
        binned_coupled_spikes = ddu.timebin_natural_movies_coupled_cell_spikes(
            jittered_movie_block.vision_dataset,
            jittered_movie_block.vision_name,
            cells_ordered,
            coupled_cells,
            spike_bin_edges,
            jitter_spike_times=jitter_spike_times
        )

        coupled_spikes_torch = torch.tensor(binned_coupled_spikes, dtype=prec_dtype, device=device)
        coupling_list.append(coupled_spikes_torch)

    return coupling_list


def multidata_bin_center_spikes_precompute_feedback_convs(
        jittered_movie_blocks: List[LoadedBrownianMovieBlock],
        bin_times_per_block: List[np.ndarray],
        feedback_basis: np.ndarray,
        center_cell_wn: Tuple[str, int],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        trim_spikes_seq: int = 0,
        prec_dtype: torch.dtype = torch.float32,
        jitter_spike_times: float = 0.0) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    '''
    Bins spikes for the center cell, and precomputes convolutions of the center
        cell spike train with the feedback filter basis.

    Stores all results of the computation on GPU

    :param jittered_movie_blocks:
    :param bin_times_per_block: spike bin edges, for each block in jittered_movie_blocks
        Length of this must match jittered_movie_blocks
    :param feedback_basis: feedback basis, shape (n_feedback_basis, n_bins_filter)
    :param center_cell_wn:
    :param cells_ordered:
    :param device:
    :param trim_spikes_seq: how many samples to trim off of the beginning of the
        binned center cell spikes, default 0 corresponding to no trim
    :param prec_dtype:
    :param jitter_spike_times: float, default=0.0, standard deviation of Gaussian (in units of electrical samples)
        to jitter the recorded spike times by
    :return:
    '''

    with torch.no_grad():
        feedback_basis_torch = torch.tensor(feedback_basis, dtype=prec_dtype, device=device)

        center_spike_list = []  # type: List[torch.Tensor]
        filtered_feedback_list = []  # type: List[torch.Tensor]
        for jittered_movie_block, spike_bin_edges in zip(jittered_movie_blocks, bin_times_per_block):

            binned_center_spikes = ddu.timebin_natural_movies_center_cell_spikes(
                jittered_movie_block.vision_dataset,
                jittered_movie_block.vision_name,
                cells_ordered,
                center_cell_wn,
                spike_bin_edges,
                jitter_spike_times=jitter_spike_times
            )

            center_spikes_torch = torch.tensor(binned_center_spikes, dtype=prec_dtype, device=device).squeeze(0)
            feedback_basis_applied = precompute_feedback_convolutions(center_spikes_torch,
                                                                      feedback_basis_torch)

            if trim_spikes_seq != 0:
                center_spikes_torch_trimmed = center_spikes_torch[trim_spikes_seq:].contiguous()
                del center_spikes_torch
                center_spike_list.append(center_spikes_torch_trimmed)
            else:
                center_spike_list.append(center_spikes_torch)

            filtered_feedback_list.append(feedback_basis_applied)

            torch.cuda.empty_cache()

        del feedback_basis_torch

    return center_spike_list, filtered_feedback_list


def multidata_bin_center_spikes_precompute_coupling_convs(
        jittered_movie_blocks: List[LoadedBrownianMovieBlock],
        bin_times_per_block: List[np.ndarray],
        coupling_basis: np.ndarray,
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32,
        jitter_spike_times: float = 0.0) -> List[torch.Tensor]:
    '''
    Bins spikes for the coupled cells, and precomputes convolutions of the coupled
        cell spike trains with the coupling filter basis.

    Returns only the results of the convolutions.

    Stores all results of the computation on GPU

    :param jittered_movie_blocks:
    :param bin_times_per_block: spike bin edges, for each block in jittered_movie_blocks
        Length of this must match jittered_movie_blocks
    :param coupling_basis: (np.ndarray) coupling_basis, shape (n_coupling_basis, n_bins_filter)
    :param coupled_cells:
    :param cells_ordered:
    :param device:
    :param prec_dtype:
    :param jitter_spike_times: float, default=0.0, standard deviation of Gaussian (in units of electrical samples)
        to jitter the recorded spike times by
    :return:
    '''

    with torch.no_grad():
        coupling_basis_torch = torch.tensor(coupling_basis, dtype=prec_dtype, device=device)

        filtered_coupling_list = []
        for jittered_movie_block, spike_bin_edges in zip(jittered_movie_blocks, bin_times_per_block):
            binned_coupled_spikes = ddu.timebin_natural_movies_coupled_cell_spikes(
                jittered_movie_block.vision_dataset,
                jittered_movie_block.vision_name,
                cells_ordered,
                coupled_cells,
                spike_bin_edges,
                jitter_spike_times=jitter_spike_times
            )

            coupled_spikes_torch = torch.tensor(binned_coupled_spikes, dtype=prec_dtype, device=device)
            coupling_basis_applied = precompute_coupling_convolutions(coupled_spikes_torch,
                                                                      coupling_basis_torch)
            del coupled_spikes_torch

            filtered_coupling_list.append(coupling_basis_applied)

            torch.cuda.empty_cache()

        del coupling_basis_torch

    return filtered_coupling_list


def _list_precompute_spat_filter_apply(
        jittered_movie_patches: List[torch.Tensor],
        bin_interval_sel_overlap: List[Tuple[torch.Tensor, torch.Tensor]],
        stim_spat_filter_torch: torch.Tensor,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
    '''

    :param jittered_movie_patches:
    :param bin_interval_sel_overlap:
    :param stim_spat_filter_torch:
    :param image_transform_lambda:
    :param device:
    :return:
    '''

    ret_list = []
    with torch.no_grad():
        for patch_frames_torch, (selection_ix_torch, overlap_bin_torch) in \
            zip(jittered_movie_patches, bin_interval_sel_overlap):

            # was torch.float32
            movie_no_upsample_torch = image_transform_lambda(patch_frames_torch)

            # shape (n_frames, n_pixels)
            movie_no_upsample_flat = movie_no_upsample_torch.reshape(movie_no_upsample_torch.shape[0], -1)

            # shape (n_frames, n_pixels) @ (n_pixels, 1)
            # -> (n_frames, 1)
            spat_basis_applied_no_upsample = movie_no_upsample_flat @ stim_spat_filter_torch[:, None]

            del movie_no_upsample_flat, movie_no_upsample_torch, patch_frames_torch

            spat_basis_applied = flat_sparse_upsample_transpose_cuda(spat_basis_applied_no_upsample,
                                                                     selection_ix_torch,
                                                                     overlap_bin_torch).squeeze(1)

            del spat_basis_applied_no_upsample

            ret_list.append(spat_basis_applied)

        torch.cuda.empty_cache()

    return ret_list


def multidata_precompute_spat_filter_apply(
        jittered_movie_blocks: List[LoadedBrownianMovieBlock],
        bin_interval_sel_overlap: List[Tuple[np.ndarray, np.ndarray]],
        stim_spat_filter: np.ndarray,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    '''
    Upsamples/transposes the movie temporally, and then applies the spatial stimulus basis

    :param block_sequence: (List[SynchronizedNSBrownianSection]) time-synchronized sections of
        the jittered natural movie dataset and stimulus
    :param movie_patches_no_upsample: (List[np.ndarray]), each has shape (n_frames_stim, height, width)
        Note that n_frames_stim could be different for each entry in the list
    :param bin_interval_sel_overlap: (List[Tuple[np.ndarray, np.ndarray]]) # FIXME
    :param stim_spat_basis: (np.ndarray) stimulus spatial basis vectors, shape (n_pixels, )
    :param image_transform_lambda: function that transforms the range of the stimulus. Called
        prior to performing any convolutions
    :param device:
    :return: List[torch.Tensor], each entry with shape (n_basis_stim_spat, n_bins)
        corresponding to the pre-application of each stimulus basis vector
    '''

    # shape (n_pixels, )
    stim_spat_filter_torch = torch.tensor(stim_spat_filter, dtype=dtype, device=device)

    # each has shape (n_frames, height, width)
    patches_list_all_torch = [
        torch.tensor(jittered_movie_block.stimulus_frame_patches, dtype=dtype, device=device)
        for jittered_movie_block in jittered_movie_blocks
    ]

    # each tensor in each pair has shape (n_bins, 2)
    overlap_sel_list_torch = [
        (torch.tensor(sel_ix, dtype=torch.long, device=device), torch.tensor(overlap_weight, dtype=dtype, device=device))
        for sel_ix, overlap_weight in bin_interval_sel_overlap
    ]

    ret_list = _list_precompute_spat_filter_apply(
        patches_list_all_torch,
        overlap_sel_list_torch,
        stim_spat_filter_torch,
        image_transform_lambda,
    )

    del stim_spat_filter_torch, patches_list_all_torch, overlap_sel_list_torch

    return ret_list


def multidata_precompute_spat_basis_mul_only(
        jittered_movie_blocks: List[LoadedBrownianMovieBlock],
        bin_interval_sel_overlap: List[Tuple[np.ndarray, np.ndarray]],
        stim_spat_basis: np.ndarray,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device) -> List[torch.Tensor]:
    '''
    Upsamples/transposes the movie temporally, and then applies the spatial stimulus basis

    :param block_sequence: (List[SynchronizedNSBrownianSection]) time-synchronized sections of
        the jittered natural movie dataset and stimulus
    :param movie_patches_no_upsample: (List[np.ndarray]), each has shape (n_frames_stim, height, width)
        Note that n_frames_stim could be different for each entry in the list
    :param bin_interval_sel_overlap: (List[Tuple[np.ndarray, np.ndarray]]) # FIXME
    :param stim_spat_basis: (np.ndarray) stimulus spatial basis vectors, shape (n_basis_spat, n_pixels)
    :param image_transform_lambda: function that transforms the range of the stimulus. Called
        prior to performing any convolutions
    :param device:
    :return: List[torch.Tensor], each entry with shape (n_basis_stim_spat, n_bins)
        corresponding to the pre-application of each stimulus basis vector
    '''

    stim_spat_basis_torch = torch.tensor(stim_spat_basis, dtype=torch.float32, device=device)

    preconv_spatial_basis = []  # type: List[torch.Tensor]
    for jittered_movie_block, sel_overlap in zip(jittered_movie_blocks, bin_interval_sel_overlap):
        # get the selection and overlap arrays
        selection_ix_arr, overlap_arr = sel_overlap

        with torch.no_grad():
            # shape (n_frames, height, width)
            patch_frames_torch = torch.tensor(jittered_movie_block.stimulus_frame_patches,
                                              dtype=torch.float32, device=device)
            movie_no_upsample_torch = image_transform_lambda(patch_frames_torch)

            # shape (n_frames, n_pixels)
            movie_no_upsample_flat = movie_no_upsample_torch.reshape(movie_no_upsample_torch.shape[0], -1)

            # shape (n_frames, n_pixels) @ (n_pixels, n_basis_spat)
            # -> (n_frames, n_basis_spat)
            spat_basis_applied_no_upsample = movie_no_upsample_flat @ stim_spat_basis_torch.T

            del movie_no_upsample_flat, movie_no_upsample_torch, patch_frames_torch

            # shape have shape (n_bins, 2)
            selection_ix_torch = torch.tensor(selection_ix_arr, dtype=torch.long, device=device)
            overlap_bin_torch = torch.tensor(overlap_arr, dtype=torch.float32, device=device)

            spat_basis_applied = flat_sparse_upsample_transpose_cuda(spat_basis_applied_no_upsample,
                                                                     selection_ix_torch,
                                                                     overlap_bin_torch)

            del spat_basis_applied_no_upsample, selection_ix_torch, overlap_bin_torch

        torch.cuda.empty_cache()

        preconv_spatial_basis.append(spat_basis_applied)

    del stim_spat_basis_torch

    return preconv_spatial_basis


def _list_precompute_spatial_convolutions(
        cropped_frames_list_torch: List[torch.Tensor],
        bin_sel_overlap_torch: List[Tuple[torch.Tensor, torch.Tensor]],
        stim_spat_basis_torch: torch.Tensor,
        fixed_timecourse_torch: torch.Tensor,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
    with torch.no_grad():
        preconv_spatial_basis = []
        for cropped_frames_torch, (selection_ix_torch, overlap_bin_torch) in \
                zip(cropped_frames_list_torch, bin_sel_overlap_torch):
            #################################################################################
            # Within this block is the more GPU memory-efficient implementation
            # Order of operations is
            # (1) Apply spatial basis to the non-time-upsampled stimulus. Discard the original stimulus
            #     from GPU
            # (2) Upsample in time, remove the non-upsampled version from GPU
            # (3) Apply the time-domain convolutions, remove the pre-convolved from GPU

            # shape (n_frames, height, width)
            movie_no_upsample_torch = image_transform_lambda(cropped_frames_torch)

            # Step (1), apply the spatial basis
            # shape (n_frames, n_pixels)
            movie_no_upsample_flat_torch = movie_no_upsample_torch.reshape(movie_no_upsample_torch.shape[0], -1)

            # shape (n_frames, n_pixels) @ (n_pixels, n_basis_spat)
            # -> (n_frames, n_basis_spat)
            spatial_basis_applied_no_upsample = movie_no_upsample_flat_torch @ stim_spat_basis_torch.T

            del movie_no_upsample_flat_torch, movie_no_upsample_torch

            # Step (2), Upsample in time
            # shape (n_basis_spat, n_bins)
            spat_basis_applied_upsampled = flat_sparse_upsample_transpose_cuda(spatial_basis_applied_no_upsample,
                                                                               selection_ix_torch,
                                                                               overlap_bin_torch)

            del spatial_basis_applied_no_upsample, selection_ix_torch, overlap_bin_torch

            # Step (3), apply the convolution in time
            # shape (n_basis_spat, n_bins - n_bins_filter + 1)
            basis_applied = F.conv1d(spat_basis_applied_upsampled[:, None, :],
                                     fixed_timecourse_torch[None, None, :]).squeeze(1)

            del spat_basis_applied_upsampled

            preconv_spatial_basis.append(basis_applied)

            torch.cuda.empty_cache()

    return preconv_spatial_basis


def multidata_precompute_spatial_convolutions(jittered_movie_blocks: List[LoadedBrownianMovieBlock],
                                              bin_interval_sel_overlap: List[Tuple[np.ndarray, np.ndarray]],
                                              stim_spat_basis: np.ndarray,
                                              fixed_timecourse: np.ndarray,
                                              image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
                                              device: torch.device,
                                              dtype: torch.dtype = torch.float32) \
        -> List[torch.Tensor]:
    '''
    Upsamples/transposes the movie temporally, and then precomputes the stimulus convolutions for
        a known fixed timecourse and all of the spatial basis vectors

    :param block_sequence: (List[LoadedBrownianMovieBlock]) time-synchronized sections of
        the jittered natural movie dataset and stimulus
    :param movie_patches_no_upsample: (List[np.ndarray]), each has shape (n_frames_stim, height, width)
        Note that n_frames_stim could be different for each entry in the list
    :param bin_interval_sel_overlap: (List[Tuple[np.ndarray, np.ndarray]]) # FIXME
    :param stim_spat_basis: (np.ndarray) stimulus spatial basis vectors, shape (n_basis_spat, n_pixels)
    :param fixed_timecourse: (np.ndarray) fixed timecourse, shape (1, n_bins_filter)
    :param image_transform_lambda: function that transforms the range of the stimulus. Called
        prior to performing any convolutions
    :param device:
    :param store_intermediate_on_cpu: whether to store the precomputed tensors on CPU rather than GPU
        to save GPU memory (useful for training on AWS where the cheap GPUs don't fit all of the data)
    :return: List[torch.Tensor], each entry with shape (n_basis_stim_spat, n_bins - n_bins_filter + 1)
        corresponding to the pre-application of each stimulus basis vector with a fixed timecourse
    '''

    with torch.no_grad():
        stim_spat_basis_torch = torch.tensor(stim_spat_basis, dtype=dtype, device=device)
        fixed_timecourse_torch = torch.tensor(fixed_timecourse, dtype=dtype, device=device)

        bin_interval_sel_overlap_torch = [
            (torch.tensor(a, dtype=torch.long, device=device), torch.tensor(b, dtype=dtype, device=device))
            for a, b, in bin_interval_sel_overlap
        ]

        patch_frames_all_torch = [
            torch.tensor(x.stimulus_frame_patches, dtype=dtype, device=device)
            for x in jittered_movie_blocks
        ]

    temp_return_vals = _list_precompute_spatial_convolutions(
        patch_frames_all_torch,
        bin_interval_sel_overlap_torch,
        stim_spat_basis_torch,
        fixed_timecourse_torch,
        image_transform_lambda
    )

    del stim_spat_basis_torch, fixed_timecourse_torch, bin_interval_sel_overlap_torch, patch_frames_all_torch

    return temp_return_vals


def _list_precompute_temporal_convolutions(
        cropped_frames_list_torch: List[torch.Tensor],
        bin_sel_overlap_torch: List[Tuple[torch.Tensor, torch.Tensor]],
        fixed_spat_filter_torch: torch.Tensor,
        stim_timecourse_basis_torch: torch.Tensor,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
    '''

    :param cropped_frames_list_torch: each entry has shape (n_frames, height, width)
    :param bin_sel_overlap_torch: each entry ahs shape (n_bins, 2) and (n_bins, 2)
        First element in each entry is the selection indices tensor, with dtype torch.LongTensor
        Second element in each entry is the weights tensor, which is a floating point tensor
    :param fixed_spat_filter_torch: fixed spatial basis, shape (n_pixels, )
    :param stim_timecourse_basis_torch: timecourse basis, shape (n_basis_timecourse, n_bins_filter)
    :return:
    '''

    preconv_timecourse_basis = []
    for patch_frames_torch, (selection_ix_torch, overlap_bin_torch) in \
            zip(cropped_frames_list_torch, bin_sel_overlap_torch):
        with torch.no_grad():
            # shape (n_frames, height, width)
            movie_no_upsample_torch = image_transform_lambda(patch_frames_torch)

            del patch_frames_torch

            # shape (n_frames, n_pixels)
            movie_flat_no_upsample_torch = movie_no_upsample_torch.reshape(movie_no_upsample_torch.shape[0], -1)

            # shape (n_frames, n_pixels) @ (n_pixels, 1)
            # -> (n_frames, 1)
            basis_applied_no_upsample = movie_flat_no_upsample_torch @ fixed_spat_filter_torch[:, None]

            del movie_no_upsample_torch, movie_flat_no_upsample_torch

            # shape (n_basis, n_bins)
            spatial_basis_applied_upsampled = flat_sparse_upsample_transpose_cuda(basis_applied_no_upsample,
                                                                                  selection_ix_torch,
                                                                                  overlap_bin_torch)
            del basis_applied_no_upsample, selection_ix_torch, overlap_bin_torch

            basis_applied = F.conv1d(spatial_basis_applied_upsampled[:, None, :],
                                     stim_timecourse_basis_torch[:, None, :]).squeeze(0)

            del spatial_basis_applied_upsampled

        torch.cuda.empty_cache()
        preconv_timecourse_basis.append(basis_applied)

    return preconv_timecourse_basis


def multidata_precompute_temporal_convolutions(jittered_movie_blocks: List[LoadedBrownianMovieBlock],
                                               bin_interval_sel_overlap: List[Tuple[np.ndarray, np.ndarray]],
                                               fixed_spat_filter: np.ndarray,
                                               stim_timecourse_basis: np.ndarray,
                                               image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
                                               device: torch.device,
                                               dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    '''
    Upsamples/transposes the movie temporally, and then precomputes the stimulus convolution with a known
        fixed spatial filter and all of the timecourse basis vectors

    :param block_sequence: (List[SynchronizedNSBrownianSection]) time-synchronized sections of
        the jittered natural movie dataset and stimulus
    :param movie_patches_no_upsample: (List[np.ndarray]), each has shape (n_frames_stim, height, width)
        Note that n_frames_stim could be different for each entry in the list
    :param bin_interval_sel_overlap: (List[Tuple[np.ndarray, np.ndarray]]), # FIXME
    :param fixed_spat_filter: (np.ndarray) fixed spatial basis, shape (n_pixels, )
    :param stim_timecourse_basis: (np.ndarray) timecourse basis, shape (n_basis_timecourse, n_bins_filter)
    :param image_transform_lambda: function that transforms the range of the stimulus. Called
        prior to performing any convolutions
    :param device:
    :param store_intermediate_on_cpu: whether to store the precomputed tensors on CPU rather than GPU
        to save GPU memory (useful for training on AWS where the cheap GPUs don't fit all of the data)
    :return: List[torch.Tensor], each entry with shape (n_basis_timecourse, n_bins - n_bins_filter + 1)
        corresponding to the pre-application of each timecourse basis vector with a fixed spatial filter
    '''

    with torch.no_grad():
        fixed_spat_filter_torch = torch.tensor(fixed_spat_filter, dtype=dtype, device=device)
        stim_timecourse_basis_torch = torch.tensor(stim_timecourse_basis, dtype=dtype, device=device)

        patch_frames_all_torch = [
            torch.tensor(x.stimulus_frame_patches, dtype=dtype, device=device) for x in jittered_movie_blocks
        ]

        sel_overlap_all_torch = [
            (torch.tensor(sel_ix_arr, dtype=torch.long, device=device),
             torch.tensor(overlap_weight, dtype=dtype, device=device))
            for sel_ix_arr, overlap_weight in bin_interval_sel_overlap
        ]

        ret_list = _list_precompute_temporal_convolutions(
            patch_frames_all_torch,
            sel_overlap_all_torch,
            fixed_spat_filter_torch,
            stim_timecourse_basis_torch,
            image_transform_lambda
        )

        del patch_frames_all_torch, sel_overlap_all_torch, fixed_spat_filter_torch, stim_timecourse_basis_torch
        return ret_list


def _count_coupled_cells(coupled_cells: Dict[str, List[int]]) -> int:
    return sum([len(val) for val in coupled_cells.values()])


def precompute_feedback_convolutions(feedback_spike_train: torch.Tensor,
                                     feedback_filt_basis: torch.Tensor) -> torch.Tensor:
    '''

    :param feedback_spike_train: (n_bins, )
    :param feedback_filt_basis: (n_basis_feedback, n_bins_filter)
    :return: shape (1, n_basis_feedback, n_bins - n_bins_filter + 1)
    '''

    # shape (1, n_basis_feedback, n_bins - n_bins_filter + 1)
    # -> (1, n_basis_feedback, n_bins - n_bins_filter + 1)
    return F.conv1d(feedback_spike_train[None, None, :],
                    feedback_filt_basis[:, None, :])


def precompute_coupling_convolutions(coupling_spike_train: torch.Tensor,
                                     coupling_filt_basis: torch.Tensor) -> torch.Tensor:
    '''

    :param coupling_spike_train: (n_coupled_cells, n_bins)
    :param coupling_filt_basis: (n_basis_coupling, n_bins_filter)
    :return: (n_coupled_cells, n_basis_coupling, n_bins_filter)
    '''

    # shape (n_coupled_cells, n_basis_coupling, n_bins_filter)
    return F.conv1d(coupling_spike_train[:, None, :],
                    coupling_filt_basis[:, None, :])


def repeats_blocks_extract_center_spikes(
        repeats_training_blocks: List[ddu.RepeatBrownianTrainingBlock],
        center_cell_wn: Tuple[str, int],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    ret_list = []
    with torch.no_grad():
        for repeats_training_block in repeats_training_blocks:
            center_spikes = ddu.extract_center_spike_train_from_repeat_training_block(
                repeats_training_block, center_cell_wn, cells_ordered
            )
            ret_list.append(torch.tensor(center_spikes, device=device, dtype=dtype))

    return ret_list


def repeats_blocks_extract_center_spikes_precompute_feedback_convs(
        repeats_training_blocks: List[ddu.RepeatBrownianTrainingBlock],
        feedback_basis: np.ndarray,
        center_cell_wn: Tuple[str, int],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        trim_spikes_seq: int = 0,
        prec_dtype: torch.dtype = torch.float32) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    '''
    Extracts spikes for the center cell out of each mini repeat training block, and precomputes
        the convolutions of the center cell spike train with the feedback filter basis

    Moves all of the output tensors to GPU as well

    :param repeats_training_blocks:
    :param feedback_basis: feedback basis, shape (n_feedback_basis, n_bins_filter)
    :param center_cell_wn:
    :param cells_ordered:
    :param device:
    :param trim_spikes_seq: how many samples to trim off of the beginning of the
        binned center cell spikes, default 0 corresponding to no trim
    :param prec_dtype:
    :return:
    '''

    with torch.no_grad():

        feedback_basis_torch = torch.tensor(feedback_basis, device=device, dtype=prec_dtype)

        center_spikes_list = []
        filtered_feedback_list = []

        for repeat_training_block in repeats_training_blocks:

            center_spikes = ddu.extract_center_spike_train_from_repeat_training_block(
                repeat_training_block, center_cell_wn, cells_ordered
            )

            center_spikes_torch = torch.tensor(center_spikes, device=device, dtype=prec_dtype)
            feedback_basis_applied = precompute_feedback_convolutions(
                center_spikes_torch, feedback_basis_torch
            )

            if trim_spikes_seq != 0:
                center_spikes_torch_trimmed = center_spikes_torch[trim_spikes_seq:].contiguous()
                del center_spikes_torch
                center_spikes_list.append(center_spikes_torch_trimmed)
            else:
                center_spikes_list.append(center_spikes_torch)

            filtered_feedback_list.append(feedback_basis_applied)

            torch.cuda.empty_cache()

        del feedback_basis_torch

    return center_spikes_list, filtered_feedback_list


def repeats_blocks_extract_coupling_spikes(
        repeats_training_blocks: List[ddu.RepeatBrownianTrainingBlock],
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:

    ret_list = []
    with torch.no_grad():
        for repeats_training_block in repeats_training_blocks:
            coupled_spikes = ddu.extract_coupled_spike_trains_from_repeat_training_block(
                repeats_training_block, coupled_cells, cells_ordered
            )
            ret_list.append(torch.tensor(coupled_spikes, dtype=prec_dtype, device=device))
    return ret_list


def repeats_blocks_precompute_spatial_filter_apply(
        repeats_training_blocks: List[ddu.RepeatBrownianTrainingBlock],
        bin_interval_sel_overlap: List[Tuple[np.ndarray, np.ndarray]],
        stim_spat_filter: np.ndarray,
        stimulus_cropping_bounds: Tuple[Tuple[int, int], Tuple[int, int]],
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    '''
    Upsamples/transposes the movie temporally, and applies the spatial stimulus filter

    :param repeats_training_blocks:
    :param bin_interval_sel_overlap:
    :param stim_spat_filter:
    :param image_transform_lambda:
    :param device:
    :return:
    '''

    bounds_h, bounds_w = stimulus_cropping_bounds
    h_low, h_high = bounds_h
    w_low, w_high = bounds_w

    with torch.no_grad():
        stim_spat_filter_torch = torch.tensor(stim_spat_filter, dtype=dtype, device=device)
        bin_interval_sel_overlap_torch = [
            (torch.tensor(a, dtype=torch.long, device=device), torch.tensor(b, dtype=dtype, device=device))
            for a, b, in bin_interval_sel_overlap
        ]

        patch_frames_all_torch = [
            torch.tensor(x.stimulus_movie[:, h_low:h_high, w_low:w_high], dtype=dtype, device=device)
            for x in repeats_training_blocks
        ]

    ret_list = _list_precompute_spat_filter_apply(
        patch_frames_all_torch,
        bin_interval_sel_overlap_torch,
        stim_spat_filter_torch,
        image_transform_lambda
    )

    del stim_spat_filter_torch, patch_frames_all_torch, bin_interval_sel_overlap_torch

    return ret_list


def repeats_blocks_precompute_coupling_convs(
        repeats_training_blocks: List[ddu.RepeatBrownianTrainingBlock],
        coupling_basis: np.ndarray,
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    '''
    Extracts the coupled cell spike trains from the repeats, and precomputes coupling
        basis filters

    Returns only the results of the convolution on GPU

    :param repeats_training_blocks:
    :param coupling_basis: (np.ndarray) coupling_basis, shape (n_coupling_basis, n_bins_filter)
    :param coupled_cells:
    :param cells_ordered:
    :param device:
    :param prec_dtype:
    :return:
    '''

    with torch.no_grad():
        coupling_basis_torch = torch.tensor(coupling_basis, dtype=prec_dtype, device=device)
        filtered_coupling_list = []
        for repeat_training_block in repeats_training_blocks:
            coupled_spikes = ddu.extract_coupled_spike_trains_from_repeat_training_block(
                repeat_training_block, coupled_cells, cells_ordered
            )

            coupled_spikes_torch = torch.tensor(coupled_spikes, dtype=prec_dtype, device=device)
            coupling_basis_applied = precompute_coupling_convolutions(coupled_spikes_torch,
                                                                      coupling_basis_torch)
            del coupled_spikes_torch

            filtered_coupling_list.append(coupling_basis_applied)

            torch.cuda.empty_cache()
        del coupling_basis_torch
    return filtered_coupling_list


def repeats_blocks_precompute_spatial_convolutions(
        repeats_training_blocks: List[ddu.RepeatBrownianTrainingBlock],
        bin_interval_sel_overlap: List[Tuple[np.ndarray, np.ndarray]],
        stim_spat_basis: np.ndarray,
        fixed_timecourse: np.ndarray,
        stimulus_cropping_bounds: Tuple[Tuple[int, int], Tuple[int, int]],
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    '''

    :param repeats_training_blocks:
    :param bin_interval_sel_overlap:
    :param stim_spat_basis:
    :param fixed_timecourse:
    :param stimulus_cropping_slice:
    :param image_transform_lambda:
    :param device:
    :return:
    '''

    bounds_h, bounds_w = stimulus_cropping_bounds
    h_low, h_high = bounds_h
    w_low, w_high = bounds_w

    with torch.no_grad():
        stim_spat_basis_torch = torch.tensor(stim_spat_basis, dtype=dtype, device=device)
        fixed_timecourse_torch = torch.tensor(fixed_timecourse, dtype=dtype, device=device)

        bin_interval_sel_overlap_torch = [
            (torch.tensor(a, dtype=torch.long, device=device), torch.tensor(b, dtype=dtype, device=device))
            for a, b, in bin_interval_sel_overlap
        ]

        patch_frames_all_torch = [
            torch.tensor(x.stimulus_movie[:, h_low:h_high, w_low:w_high], dtype=dtype, device=device)
            for x in repeats_training_blocks
        ]

    temp_return_vals = _list_precompute_spatial_convolutions(
        patch_frames_all_torch,
        bin_interval_sel_overlap_torch,
        stim_spat_basis_torch,
        fixed_timecourse_torch,
        image_transform_lambda
    )

    del stim_spat_basis_torch, fixed_timecourse_torch, bin_interval_sel_overlap_torch, patch_frames_all_torch

    return temp_return_vals


def repeats_blocks_precompute_temporal_convolutions(
        repeats_training_blocks: List[ddu.RepeatBrownianTrainingBlock],
        bin_interval_sel_overlap: List[Tuple[np.ndarray, np.ndarray]],
        fixed_spat_filter: np.ndarray,
        stim_timecourse_basis: np.ndarray,
        stimulus_cropping_bounds: Tuple[Tuple[int, int], Tuple[int, int]],
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    bounds_h, bounds_w = stimulus_cropping_bounds
    h_low, h_high = bounds_h
    w_low, w_high = bounds_w

    with torch.no_grad():
        fixed_spat_filter_torch = torch.tensor(fixed_spat_filter, dtype=dtype, device=device)
        stim_timecourse_basis_torch = torch.tensor(stim_timecourse_basis, dtype=dtype, device=device)

        bin_interval_sel_overlap_torch = [
            (torch.tensor(a, dtype=torch.long, device=device), torch.tensor(b, dtype=dtype, device=device))
            for a, b, in bin_interval_sel_overlap
        ]

        patch_frames_all_torch = [
            torch.tensor(x.stimulus_movie[:, h_low:h_high, w_low:w_high], dtype=dtype, device=device)
            for x in repeats_training_blocks
        ]

        temp_return_values = _list_precompute_temporal_convolutions(
            patch_frames_all_torch,
            bin_interval_sel_overlap_torch,
            fixed_spat_filter_torch,
            stim_timecourse_basis_torch,
            image_transform_lambda
        )

        del fixed_spat_filter_torch, stim_timecourse_basis_torch, bin_interval_sel_overlap_torch, patch_frames_all_torch

    return temp_return_values
