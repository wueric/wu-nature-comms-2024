import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Dict, Tuple, Callable

import lib.data_utils.dynamic_data_util as ddu
from lib.data_utils.movie_processing import multiresolution_spatial_basis_application


def frame_rate_wn_lnp_multidata_bin_spikes(
        wn_movie_blocks: List[ddu.LoadedWNMovieBlock],
        center_cell_wn: Tuple[str, int],
        device: torch.device,
        jitter_time_amount: float = 0.0,
        trim_spikes_seq: int = 0,
        prec_dtype: torch.dtype = torch.float32) \
        -> List[torch.Tensor]:
    output_list = []
    center_cell_type, center_cell_id = center_cell_wn
    for wn_movie_block in wn_movie_blocks:
        wn_bin_edges = wn_movie_block.timing_synchro
        binned_spikes_wn = ddu.movie_bin_spikes_multiple_cells2(wn_movie_block.vision_dataset,
                                                                [[center_cell_id, ]],
                                                                wn_bin_edges,
                                                                jitter_time_amount=jitter_time_amount)
        wn_center_cell_spikes = binned_spikes_wn.squeeze(0)
        if trim_spikes_seq != 0:
            wn_center_cell_spikes = wn_center_cell_spikes[trim_spikes_seq:]

        binned_spikes_torch = torch.tensor(wn_center_cell_spikes, dtype=prec_dtype,
                                           device=device)
        output_list.append(binned_spikes_torch)
    return output_list


def frame_rate_wn_preapply_spatial_convolutions(wn_movie_block: ddu.LoadedWNMovieBlock,
                                                spatial_basis_imshape: np.ndarray,
                                                fixed_timecourse: np.ndarray,
                                                stimulus_rescale_lambda: Callable[[torch.Tensor], torch.Tensor],
                                                device: torch.device,
                                                prec_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    with torch.no_grad():
        wn_cropped_stimulus_bw_unscaled_torch = torch.tensor(wn_movie_block.stimulus_frame_patches_wn_resolution,
                                                             dtype=prec_dtype, device=device)

        wn_cropped_stimulus_bw_torch = stimulus_rescale_lambda(wn_cropped_stimulus_bw_unscaled_torch).to(prec_dtype)
        spat_spline_basis_imshape_torch = torch.tensor(spatial_basis_imshape, dtype=prec_dtype, device=device)

        # shape (n_frames, n_spat_basis)
        # -> (n_spat_basis, n_frames)
        mres_apply_basis_shape = multiresolution_spatial_basis_application(
            wn_cropped_stimulus_bw_torch,
            spat_spline_basis_imshape_torch).to(prec_dtype).T

        del wn_cropped_stimulus_bw_torch, wn_cropped_stimulus_bw_unscaled_torch
        del spat_spline_basis_imshape_torch

        # apply the time-domain convolution
        timecourse_filter_torch = torch.tensor(fixed_timecourse, dtype=prec_dtype, device=device)
        wn_upsampled_timecourse_convolved = F.conv1d(
            mres_apply_basis_shape[:, None, :],
            timecourse_filter_torch[None, None, :]).squeeze(1)

        del mres_apply_basis_shape

    return wn_upsampled_timecourse_convolved


def frame_rate_wn_multidata_precompute_spatial_convolutions(
        wn_movie_block: List[ddu.LoadedWNMovieBlock],
        spatial_basis_imshape: np.ndarray,
        fixed_timecourse: np.ndarray,
        stimulus_rescale_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    return [
        frame_rate_wn_preapply_spatial_convolutions(
            a, spatial_basis_imshape, fixed_timecourse, stimulus_rescale_lambda, device,
            prec_dtype=prec_dtype
        ) for a in wn_movie_block
    ]


def frame_rate_wn_preapply_temporal_convolutions(
        wn_movie_block: ddu.LoadedWNMovieBlock,
        fixed_spatial_filter_imshape: np.ndarray,
        timecourse_basis: np.ndarray,
        stimulus_rescale_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32) -> torch.Tensor:
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

        timecourse_basis_torch = torch.tensor(timecourse_basis, dtype=prec_dtype, device=device)

        # shape (n_basis_timecourse, n_bins - n_bins_filter + 1)
        conv_output = F.conv1d(mres_apply_basis_shape[None, None, :],
                               timecourse_basis_torch[:, None, :]).squeeze(0)

        del timecourse_basis_torch, mres_apply_basis_shape

    return conv_output


def frame_rate_wn_multidata_precompute_temporal_convolutions(
        wn_movie_blocks: List[ddu.LoadedWNMovieBlock],
        fixed_spatial_filter_imshape: np.ndarray,
        timecourse_basis: np.ndarray,
        stimulus_rescale_lambda: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    return [
        frame_rate_wn_preapply_temporal_convolutions(
            a, fixed_spatial_filter_imshape, timecourse_basis,
            stimulus_rescale_lambda, device, prec_dtype=prec_dtype)
        for a in wn_movie_blocks
    ]
