from typing import Tuple, List, Dict, Callable

import numpy as np
import torch
from torch.nn import functional as F

from lib.data_utils import dynamic_data_util as ddu
from lib.data_utils.dynamic_data_util import LoadedFlashPatch, PartitionType
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct


def _flashed_ns_preapply_spatial_basis(batched_flashed_ns_flat: np.ndarray,
                                       spat_basis_flat: np.ndarray,
                                       device: torch.device,
                                       prec_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    '''

    :param batched_flashed_ns_flat: shape (batch, n_pix)
    :param spat_basis_flat: shape (n_basis, n_pix)
    :param prec_dtype:
    :return: shape (batch, n_basis)
    '''

    with torch.no_grad():
        ns_stimulus_spatial_torch = torch.tensor(batched_flashed_ns_flat, dtype=prec_dtype, device=device)
        ns_stimulus_spat_basis_torch = torch.tensor(spat_basis_flat, dtype=prec_dtype, device=device)

        # shape (batch, n_pix) @ (n_pix, n_basis) -> (batch, n_basis)
        spat_filt_applied = ns_stimulus_spatial_torch @ ns_stimulus_spat_basis_torch.T

        del ns_stimulus_spatial_torch, ns_stimulus_spat_basis_torch

        return spat_filt_applied


def precompute_timecourse_feedback_basis_convs(
        stimulus_time: torch.Tensor,
        stim_time_filt_basis: torch.Tensor,
        spikes_cell: torch.Tensor,
        feedback_filt_basis: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''

    :param stimulus_time: shape (n_bins, ), the time component of the stimulus
    :param stim_time_filt_basis: (n_basis_time_filt, n_bins_filt), stimulus temporal basis set
    :param spikes_cell: (batch, n_bins), spikes for the cell being fit, for every trial
    :param feedback_filt_basis: (n_basis_feedback, n_bins_filt), feedback basis
    :return:
    '''

    with torch.no_grad():
        # shape (1, 1, n_bins)
        # -> (n_basis_time_filt, n_bins - n_bins_filter + 1)
        time_domain_basis_applied = F.conv1d(stimulus_time[None, None, :],
                                             stim_time_filt_basis[:, None, :]).squeeze(0)

        # shape (batch, n_basis_feedback, n_bins - n_bins_filter + 1)
        feedback_basis_applied = F.conv1d(spikes_cell[:, None, :],
                                          feedback_filt_basis[:, None, :])

        return time_domain_basis_applied, feedback_basis_applied


def precompute_timecourse_basis_convs(
        stimulus_time: torch.Tensor,
        stim_time_filt_basis: torch.Tensor) -> torch.Tensor:
    '''

    :param stimulus_time: shape (n_bins, ), the time component of the stimulus
    :param stim_time_filt_basis: (n_basis_time_filt, n_bins_filt), stimulus temporal basis set
    :return:
    '''

    with torch.no_grad():
        # shape (1, 1, n_bins)
        # -> (n_basis_time_filt, n_bins - n_bins_filter + 1)
        time_domain_basis_applied = F.conv1d(stimulus_time[None, None, :],
                                             stim_time_filt_basis[:, None, :]).squeeze(0)
        return time_domain_basis_applied


def precompute_feedback_basis_convs(
        spikes_cell: torch.Tensor,
        feedback_filt_basis: torch.Tensor) -> torch.Tensor:
    '''

    :param spikes_cell: (batch, n_bins), spikes for the cell being fit, for every trial
    :param feedback_filt_basis: (n_basis_feedback, n_bins_filt), feedback basis
    :return:
    '''
    with torch.no_grad():
        # shape (batch, n_basis_feedback, n_bins - n_bins_filter + 1)
        feedback_basis_applied = F.conv1d(spikes_cell[:, None, :],
                                          feedback_filt_basis[:, None, :])

        return feedback_basis_applied


def precompute_flashed_coupling_basis_convs(
        spikes_coupled_cells: torch.Tensor,
        coupling_filt_basis: torch.Tensor) -> torch.Tensor:
    '''

    :param spikes_coupled_cells: (batch, n_coupled_cells, n_bins),  spikes for all of the
        coupled cells, for every trial
    :param coupling_filt_basis: (n_basis_coupling, n_bins_filt), coupling basis
    :return:
    '''

    with torch.no_grad():
        batch, n_coupled_cells, _ = spikes_coupled_cells.shape
        temp_flattened_coupled_spikes = spikes_coupled_cells.reshape((batch * n_coupled_cells, -1))

        coupling_basis_applied_temp = F.conv1d(temp_flattened_coupled_spikes[:, None, :],
                                               coupling_filt_basis[:, None, :])

        coupling_basis_applied = coupling_basis_applied_temp.reshape(batch, n_coupled_cells,
                                                                     coupling_filt_basis.shape[0], -1)

        return coupling_basis_applied


def flashed_raw_spikes_precompute_feedback_basis_convs(
        binned_spikes: np.ndarray,
        feedback_basis: np.ndarray,
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32,
        trim_spikes_seq: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Moves center cell spikes to GPU, and precomputes the feedback convolutions

    Returns both the center cell spikes and the feedback convolution outputs, all on GPU

    :param binned_spikes: shape (batch, n_bins)
    :param feedback_basis:
    :param device:
    :param prec_dtype:
    :return:
    '''

    ns_feedback_spikes_torch = torch.tensor(binned_spikes, dtype=prec_dtype, device=device)
    feedback_basis_torch = torch.tensor(feedback_basis, dtype=prec_dtype, device=device)
    feedback_ba = precompute_feedback_basis_convs(
        ns_feedback_spikes_torch, feedback_basis_torch)
    del feedback_basis_torch

    if trim_spikes_seq != 0:
        ns_feedback_spikes_torch = ns_feedback_spikes_torch[:, trim_spikes_seq:].contiguous()

    return ns_feedback_spikes_torch, feedback_ba


def flashed_raw_spikes_precompute_coupling_basis_convs(
        batched_coupled_spikes: np.ndarray,
        coupling_basis: np.ndarray,
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    '''

    :param batched_coupled_spike:
    :param coupling_basis:
    :param device:
    :param prec_dtype:
    :return:
    '''
    ns_coupling_spikes_torch = torch.tensor(batched_coupled_spikes, dtype=prec_dtype, device=device)
    coupling_basis_torch = torch.tensor(coupling_basis, dtype=prec_dtype, device=device)
    coupling_basis_applied = precompute_flashed_coupling_basis_convs(
        ns_coupling_spikes_torch, coupling_basis_torch)
    del ns_coupling_spikes_torch, coupling_basis_torch
    return coupling_basis_applied


def flashed_raw_precompute_spatial_basis(cropped_patch: np.ndarray,
                                         spat_basis_flat: np.ndarray,
                                         image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
                                         device: torch.device,
                                         prec_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # shape (batch, n_pixels)
    flat_patch = cropped_patch.reshape(cropped_patch.shape[0], -1)

    with torch.no_grad():
        ns_stimulus_spatial_torch = image_transform_lambda(
            torch.tensor(flat_patch, dtype=prec_dtype, device=device))
        ns_stimulus_spat_basis_torch = torch.tensor(spat_basis_flat, dtype=prec_dtype, device=device)

        # shape (batch, n_pix) @ (n_pix, n_basis) -> (batch, n_basis)
        spat_filt_applied = ns_stimulus_spatial_torch @ ns_stimulus_spat_basis_torch.T

        del ns_stimulus_spatial_torch, ns_stimulus_spat_basis_torch

        return spat_filt_applied


def flashed_raw_precompute_timecourse_basis_conv(
        stimulus_time_component,
        timecourse_basis: np.ndarray,
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32) -> torch.Tensor:

    ns_time_component_torch = torch.tensor(stimulus_time_component, dtype=prec_dtype, device=device)
    timecourse_basis_torch = torch.tensor(timecourse_basis, dtype=prec_dtype, device=device)
    timecourse_ba = precompute_timecourse_basis_convs(
        ns_time_component_torch,
        timecourse_basis_torch,
    )

    del timecourse_basis_torch, ns_time_component_torch

    return timecourse_ba




def flashed_ns_bin_spikes_precompute_timecourse_basis_conv2(
        loaded_nscenes_list: List[LoadedFlashPatch],
        timecourse_basis: np.ndarray,
        device: torch.device,
        prec_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    ns_time_component_torch = torch.tensor(loaded_nscenes_list[0].stimulus_time_component,
                                           dtype=prec_dtype, device=device)

    timecourse_basis_torch = torch.tensor(timecourse_basis, dtype=prec_dtype, device=device)
    timecourse_ba = precompute_timecourse_basis_convs(
        ns_time_component_torch,
        timecourse_basis_torch,
    )

    del timecourse_basis_torch, ns_time_component_torch

    return timecourse_ba


def flashed_ns_bin_spikes_only(
        loaded_nscenes_list: List[LoadedFlashPatch],
        center_cell_wn: Tuple[str, int],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        jitter_time_amount: float = 0.0,
        prec_dtype: torch.dtype = torch.float32,
        trim_spikes_seq: int = 0) -> torch.Tensor:
    center_cell_type, center_cell_id = center_cell_wn

    fit_cell_binned = ddu.timebin_load_single_partition_trials_subset_cells(
        cells_ordered,
        {center_cell_type: [center_cell_id, ]},
        loaded_nscenes_list,
        return_stacked_array=True,
        jitter_time_amount=jitter_time_amount
    )[:, 0, :]

    if trim_spikes_seq != 0:
        fit_cell_binned = fit_cell_binned[:, trim_spikes_seq:]

    ns_feedback_spikes_torch = torch.tensor(fit_cell_binned, dtype=prec_dtype, device=device)

    del fit_cell_binned

    return ns_feedback_spikes_torch


def flashed_ns_bin_spikes_precompute_feedback_convs2(
        loaded_nscenes_list: List[LoadedFlashPatch],
        feedback_basis: np.ndarray,
        center_cell_wn: Tuple[str, int],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        jitter_time_amount: float = 0.0,
        prec_dtype: torch.dtype = torch.float32,
        trim_spikes_seq: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    center_cell_type, center_cell_id = center_cell_wn

    fit_cell_binned = ddu.timebin_load_single_partition_trials_subset_cells(
        cells_ordered,
        {center_cell_type: [center_cell_id, ]},
        loaded_nscenes_list,
        return_stacked_array=True,
        jitter_time_amount=jitter_time_amount
    )[:, 0, :]

    ns_feedback_spikes_torch, feedback_ba = flashed_raw_spikes_precompute_feedback_basis_convs(
        fit_cell_binned,
        feedback_basis,
        device,
        prec_dtype=prec_dtype,
        trim_spikes_seq=trim_spikes_seq
    )

    # save RAM
    del fit_cell_binned

    return ns_feedback_spikes_torch, feedback_ba


def flashed_ns_bin_spikes_precompute_coupling_convs2(
        loaded_nscenes_list: List[LoadedFlashPatch],
        coupling_basis: np.ndarray,
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        device: torch.device,
        jitter_time_amount: float = 0.0,
        prec_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    '''

    :param loaded_nscenes_list:
    :param coupling_basis:
    :param coupled_cells:
    :param cells_ordered:
    :param device:
    :param jitter_time_amount:
    :param prec_dtype:
    :return:
    '''
    coupled_cells_binned = ddu.timebin_load_single_partition_trials_subset_cells(
        cells_ordered,
        coupled_cells,
        loaded_nscenes_list,
        return_stacked_array=True,
        jitter_time_amount=jitter_time_amount
    )

    to_return = flashed_raw_spikes_precompute_coupling_basis_convs(
        coupled_cells_binned,
        coupling_basis,
        device,
        prec_dtype=prec_dtype
    )

    del coupled_cells_binned

    return to_return


def flashed_ns_precompute_spatial_basis(loaded_nscenes_patch_list: List[LoadedFlashPatch],
                                        spat_basis_flat: np.ndarray,
                                        image_transform_lambda: Callable[[torch.Tensor], torch.Tensor],
                                        device: torch.device,
                                        prec_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    '''

    :param loaded_nscenes_list:
    :param dataset_partition:
    :param spat_basis_flat: shape (n_basis, n_pix)
    :param prec_dtype:
    :return: shape (batch, n_basis)
    '''

    orig_stimulus_cat = np.concatenate([x.frames_cached for x in loaded_nscenes_patch_list], axis=0)
    return flashed_raw_precompute_spatial_basis(orig_stimulus_cat,
                                                spat_basis_flat,
                                                image_transform_lambda,
                                                device,
                                                prec_dtype=prec_dtype)
