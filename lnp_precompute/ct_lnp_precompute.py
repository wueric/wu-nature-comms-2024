import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Tuple, Dict, Callable

import lib.data_utils.dynamic_data_util as ddu
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct

FrameRateLNPSpikeTrains = List[torch.Tensor]


def frame_rate_lnp_multidata_bin_spikes(
        jittered_movie_blocks: List[ddu.LoadedBrownianMovieBlock],
        center_cell_wn: Tuple[str, int],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32,
        jitter_spike_times: float = 0.0,
        trim_spikes_seq: int = 0) -> FrameRateLNPSpikeTrains:
    center_spike_list = []  # type: List[torch.Tensor]
    for jittered_movie_block in jittered_movie_blocks:
        _, binned_center_spikes, _ = ddu.framerate_timebin_natural_movies_subset_cells(
            jittered_movie_block.vision_dataset,
            jittered_movie_block.vision_name,
            jittered_movie_block.timing_synchro,
            cells_ordered,
            center_cell_wn,
            {},
            jitter_spike_times=jitter_spike_times
        )

        binned_center_spikes = binned_center_spikes.squeeze(0)

        if trim_spikes_seq != 0:
            binned_center_spikes = binned_center_spikes[trim_spikes_seq:]

        center_spikes_torch = torch.tensor(binned_center_spikes, dtype=prec_dtype, device=device)

        center_spike_list.append(center_spikes_torch)

    return center_spike_list


def frame_rate_multidata_precompute_spatial_convolutions(
        jittered_movie_blocks: List[ddu.LoadedBrownianMovieBlock],
        stim_spat_basis: np.ndarray,
        fixed_timecourse: np.ndarray,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    '''
    Transposes the movie temporally, and then precomputes the stimulus convolutions
        for known fixed timecourse and all of the spatial basis vectors

    Assumes that the movie does not need to be upsampled, since we are operating
        at frame rate (i.e. the movie that we loaded from disk is at frame rate)

    :param jittered_movie_blocks:
    :param stim_spat_basis:
    :param fixed_timecourse:
    :param image_transform_lambda:
    :param device:
    :param dtype:
    :return:
    '''

    stim_spat_basis_torch = torch.tensor(stim_spat_basis, dtype=dtype, device=device)
    fixed_timecourse_torch = torch.tensor(fixed_timecourse, dtype=dtype, device=device)

    preconv_spatial_basis = []  # type: List[torch.Tensor]
    for jittered_movie_block in jittered_movie_blocks:
        with torch.no_grad():
            # shape (n_frames, height, width)
            patch_frames_torch = torch.tensor(jittered_movie_block.stimulus_frame_patches,
                                              dtype=dtype, device=device)

            # shape (n_frames, height, width)
            movie_torch = image_transform_lambda(patch_frames_torch)

            del patch_frames_torch

            # Step (1), apply the spatial basis
            # shape (n_frames, n_pixels)
            movie_flat_torch = movie_torch.reshape(movie_torch.shape[0], -1)

            # shape (n_basis_spat, n_pixels) @ (n_pixels, n_frames)
            # -> (n_basis_spat, n_frames)
            spatial_basis_applied = stim_spat_basis_torch @ movie_flat_torch.T

            del movie_flat_torch, movie_torch

            # Step (3), apply the convolution in time
            # shape (n_basis_spat, n_bins - n_bins_filter + 1)
            basis_applied = F.conv1d(spatial_basis_applied[:, None, :],
                                     fixed_timecourse_torch[None, None, :]).squeeze(1)

            del spatial_basis_applied

        preconv_spatial_basis.append(basis_applied)

    del stim_spat_basis_torch, fixed_timecourse_torch
    return preconv_spatial_basis


def frame_rate_multidata_precompute_temporal_convolutions(
        jittered_movie_blocks: List[ddu.LoadedBrownianMovieBlock],
        fixed_spat_filter: np.ndarray,
        stim_timecourse_basis: np.ndarray,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    '''
    Transposes the movie temporally, and then precomputes the stimulus convolutions
        for known fixed spatial filter and all of the timecourse basis vectors

    Assumes that the movie does not need to be upsampled, since we are operating
        at frame rate (i.e. the movie that we loaded from disk is at frame rate)

    :param jittered_movie_blocks:
    :param fixed_spat_filter:
    :param stim_timecourse_basis:
    :param image_transform_lambda:
    :param device:
    :param dtype:
    :return:
    '''

    fixed_spat_filter_torch = torch.tensor(fixed_spat_filter, dtype=dtype, device=device)
    stim_timecourse_basis_torch = torch.tensor(stim_timecourse_basis, dtype=dtype, device=device)

    preconv_timecourse_basis = []  # type: List[torch.Tensor]
    for jittered_movie_block in jittered_movie_blocks:

        with torch.no_grad():

            # shape (n_frames, height, width)
            patch_frames_torch = torch.tensor(jittered_movie_block.stimulus_frame_patches,
                                              dtype=dtype, device=device)
            movie_torch = image_transform_lambda(patch_frames_torch)  # should be an in-place op

            del patch_frames_torch

            # shape (n_frames, n_pixels)
            movie_flat_torch = movie_torch.reshape(movie_torch.shape[0], -1)

            # shape (n_frames, n_pixels) @ (n_pixels, 1)
            # -> (n_frames, 1) -> (1, n_frames)
            basis_applied_no_upsample = (movie_flat_torch @ fixed_spat_filter_torch[:, None]).T

            basis_applied = F.conv1d(basis_applied_no_upsample[:, None, :],
                                     stim_timecourse_basis_torch[:, None, :]).squeeze(0)

            del basis_applied_no_upsample, movie_torch, movie_flat_torch

            preconv_timecourse_basis.append(basis_applied)

    return preconv_timecourse_basis


def frame_rate_multidata_precompute_spat_filter_apply(
        jittered_movie_blocks: List[ddu.LoadedBrownianMovieBlock],
        stim_spat_filter: np.ndarray,
        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device) -> List[torch.Tensor]:

    # shape (n_pixels, )
    stim_spat_filter_torch = torch.tensor(stim_spat_filter, dtype=torch.float32, device=device)

    preconv_spatial_basis = []  # type: List[torch.Tensor]
    for jittered_movie_block in jittered_movie_blocks:

        with torch.no_grad():
            # shape (n_frames, height, width)
            patch_frames_torch = torch.tensor(jittered_movie_block.stimulus_frame_patches,
                                              dtype=torch.float32, device=device) # FIXME might break AMP
            # was torch.float32
            movie_no_upsample_torch = image_transform_lambda(patch_frames_torch)

            # shape (n_frames, n_pixels)
            movie_no_upsample_flat = movie_no_upsample_torch.reshape(movie_no_upsample_torch.shape[0], -1)

            # shape (n_frames, n_pixels) @ (n_pixels, 1)
            # -> (n_frames, 1)
            spat_basis_applied_no_upsample = movie_no_upsample_flat @ stim_spat_filter_torch[:, None]

            del patch_frames_torch, movie_no_upsample_torch, movie_no_upsample_flat

            preconv_spatial_basis.append(spat_basis_applied_no_upsample.T)

    return preconv_spatial_basis

