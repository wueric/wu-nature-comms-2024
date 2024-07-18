import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import numpy as np

from typing import Dict, List, Tuple, Callable, Union, Optional, Any

from denoise_inverse_alg.flashed_recons_glm_components import flashed_recons_compute_feedback_exp_arg, \
    flashed_recons_compute_coupling_exp_arg, flashed_recons_compute_timecourse_component
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import CroppedSTABoundingBox, get_max_bounding_box_dimension
from denoise_inverse_alg.hqs_alg import HQS_X_Problem, BatchParallel_HQS_X_Problem
from convex_optim_base.optim_base import SingleUnconstrainedProblem, BatchParallelUnconstrainedProblem
from optimization_encoder.trial_glm import FittedGLMFamily, FittedGLM, FittedFBOnlyGLMFamily

from simple_priors.gaussian_prior import ConvPatch1FGaussianPrior

from dataclasses import dataclass
from abc import ABCMeta, abstractmethod


def _count_max_coupled_cells(fit_glms: Dict[str, FittedGLMFamily]) -> int:
    '''
    Counts the maximum number of coupled cells over every cell that has
        been fit (traverses over the cell types)

    :param fit_glms: fitted GLM families for each cell type, str key is cell type
    :return: int, maximum number of coupled cells for ANY model that has been fit
    '''

    max_coupled_cells = 0
    for ct, glm_family in fit_glms.items():
        for cell_id, fitted_glm in glm_family.fitted_models.items():
            max_coupled_cells = max(max_coupled_cells,
                                    fitted_glm.coupling_cells_weights[0].shape[0])

    return max_coupled_cells


def reinflate_spatial_filter(deflated_filter: np.ndarray,
                             cropping_bbox: CroppedSTABoundingBox,
                             downsample_factor: int = 1,
                             crop_width_low: int = 0,
                             crop_height_low: int = 0,
                             crop_width_high: int = 0,
                             crop_height_high: int = 0) -> np.ndarray:
    '''
    Inflates (uncrops) the GLM spatial filters back to the full size image
        to form the full-scale reconstruction filter

    Takes care of cropping, rescaling, etc.

    In the edge cell case, where the STA bounding box goes outside the stimulus
        area, the logic in the model fitting code is to place the (0, 0) idx
        of the reduced size bounding box in the (0, 0) idx of the full size box

        Note that in this edge cell case, deflated_filter has the full size height
        and width, since the model would have been fit with padding.

    :param deflated_filter: cropped filter, unflattened (i.e. rectangular matrix, not a vector)
        shape (height, width)
    :param cropping_bbox: CroppedSTABoundingBox, cropping bounding box used to do the initial crop
    :param raw_height: int, height of the output reconstruction filter
    :param raw_width: int, width of the output recosnstruction filter
    :param downsample_factor: int, downsampling factor (going from original raw image to the target
        reconstruction image)
    :param crop_width_low: int, number of pixels to crop (prior to downsampling)
    :param crop_height_low: int, number of pixels to crop (prior to downsampling)
    :param crop_width_high: int, number of pixels to crop (prior to downsampling)
    :param crop_height_high: int, number of pixels to crop (prior to downsampling)
    :return:
    '''

    raw_height, raw_width = cropping_bbox.compute_post_crop_downsample_shape(crop_hlow=crop_height_low,
                                                                             crop_hhigh=crop_height_high,
                                                                             crop_wlow=crop_width_low,
                                                                             crop_whigh=crop_width_high,
                                                                             downsample_factor=downsample_factor)

    inflated_filter = np.zeros((raw_height, raw_width), dtype=np.float32)
    deflate_grab_slice = cropping_bbox.make_selection_sliceobj(crop_hlow=crop_height_low,
                                                               crop_hhigh=crop_height_high,
                                                               crop_wlow=crop_width_low,
                                                               crop_whigh=crop_width_high,
                                                               downsample_factor=downsample_factor)
    inflate_place_slice = cropping_bbox.make_precropped_sliceobj(crop_hlow=crop_height_low,
                                                                 crop_hhigh=crop_height_high,
                                                                 crop_wlow=crop_width_low,
                                                                 crop_whigh=crop_width_high,
                                                                 downsample_factor=downsample_factor)
    inflated_filter[inflate_place_slice] = deflated_filter[deflate_grab_slice]
    return inflated_filter


@dataclass
class CompactModel:
    '''
    Used as an intermediate representation of the parameters of a single
        cell GLM for spot-checking the model, and for
        packing the single-cell model into the full-size multi-cell model
    '''
    spatial_filter: np.ndarray  # shape (height, width), same dimensions as the spatial basis
    timecourse_filter: np.ndarray  # shape (n_bins, )
    feedback_filter: np.ndarray  # shape (n_bins, )

    coupling_params: Tuple[np.ndarray, np.ndarray]
    # first array is the cell id, integer-valued, shape (n_cells, )
    # array has shape (n_cells, n_bins) which are the coupling filters
    # for the respective cells

    bias: np.ndarray

    stimulus_bounding_box: CroppedSTABoundingBox


@dataclass
class FullResolutionCompactModel:
    '''
    Used as an intermediate representation of the parameters of a single
        cell GLM for spot-checking the model

    Note that the model here is fitted on the full resolution stimulus
        with no spatial basis
    '''

    spatial_filter: np.ndarray  # shape (height, width), same shape as the stimulus
    timecourse_filter: np.ndarray  # shape (n_bins, )
    feedback_filter: np.ndarray  # shape (n_bins, )

    coupling_params: Union[Tuple[np.ndarray, np.ndarray], None]
    # first array is the cell id, integer-valued, shape (n_cells, )
    # array has shape (n_cells, n_bins) which are the coupling filters
    # for the respective cells

    bias: np.ndarray


def _extract_cropped_glm_params(glm_family: Union[FittedGLMFamily, FittedFBOnlyGLMFamily],
                                cell_id: int,
                                bounding_box: CroppedSTABoundingBox,
                                full_height: int,
                                full_width: int,
                                downsample_factor: int = 1,
                                crop_width_low: int = 0,
                                crop_width_high: int = 0,
                                crop_height_low: int = 0,
                                crop_height_high: int = 0) -> FullResolutionCompactModel:
    is_feedback_only = isinstance(glm_family, FittedFBOnlyGLMFamily)

    fitted_glm = glm_family.fitted_models[cell_id]

    # shape (n_feedback_basis, n_bins_filter)
    feedback_basis = glm_family.feedback_basis

    # shape (n_timecourse_basis, n_bins_filter)
    timecourse_basis = glm_family.timecourse_basis

    # shape (1, n_basis_stim_time)
    timecourse_weights = fitted_glm.timecourse_weights

    # shape (1, n_basis_stim_time) @ (n_timecourse_basis, n_bins_filter)
    # -> (1, n_bins_filter) -> (n_bins_filter, )
    timecourse_filter = (timecourse_weights @ timecourse_basis).squeeze(0)

    # shape (1, n_basis_feedback)
    feedback_weights = fitted_glm.feedback_weights

    # shape (1, n_basis_feedback) @ (n_basis_feedback, n_bins_filter)
    # -> (1, n_bins_filter) -> (n_bins_filter, )
    feedback_filter = (feedback_weights @ feedback_basis).squeeze(0)

    full_spatial_filter = np.zeros((full_height, full_width), dtype=np.float32)

    putback_slice_obj_h, putback_slice_obj_w = bounding_box.make_precropped_sliceobj(
        crop_hlow=crop_height_low,
        crop_hhigh=crop_height_high,
        crop_wlow=crop_width_low,
        crop_whigh=crop_width_high,
        downsample_factor=downsample_factor)

    full_spatial_filter[putback_slice_obj_h, putback_slice_obj_w] = fitted_glm.spatial_weights

    bias = fitted_glm.spatial_bias

    coupling_model_params = None
    if not is_feedback_only:
        # shape (n_coupling_basis, n_bins_filter)
        coupling_basis = glm_family.coupling_basis

        # coupling_weights shape (n_coupled_cells, n_coupling_basis)
        # coupling_ids shape (n_coupled_cells, ); these are reference dataset cell ids
        coupling_weights, coupling_ids = fitted_glm.coupling_cells_weights

        # (n_coupled_cells, n_coupling_basis) @ (n_coupling_basis, n_bins_filter)
        # -> (n_coupled_cells, n_bins_filter)
        coupling_filters_cell = coupling_weights @ coupling_basis

        coupling_model_params = (coupling_ids, coupling_filters_cell)

    return FullResolutionCompactModel(
        full_spatial_filter,
        timecourse_filter,
        feedback_filter,
        coupling_model_params,
        bias
    )


def _extract_full_res_glm_params(glm_family: FittedGLMFamily,
                                 cell_id: int) -> FullResolutionCompactModel:
    fitted_glm = glm_family.fitted_models[cell_id]

    # shape (n_coupling_basis, n_bins_filter)
    coupling_basis = glm_family.coupling_basis

    # shape (n_feedback_basis, n_bins_filter)
    feedback_basis = glm_family.feedback_basis

    # shape (n_timecourse_basis, n_bins_filter)
    timecourse_basis = glm_family.timecourse_basis

    # coupling_weights shape (n_coupled_cells, n_coupling_basis)
    # coupling_ids shape (n_coupled_cells, ); these are reference dataset cell ids
    coupling_weights, coupling_ids = fitted_glm.coupling_cells_weights

    # (n_coupled_cells, n_coupling_basis) @ (n_coupling_basis, n_bins_filter)
    # -> (n_coupled_cells, n_bins_filter)
    coupling_filters_cell = coupling_weights @ coupling_basis

    # shape (1, n_basis_stim_time)
    timecourse_weights = fitted_glm.timecourse_weights

    # shape (1, n_basis_stim_time) @ (n_timecourse_basis, n_bins_filter)
    # -> (1, n_bins_filter) -> (n_bins_filter, )
    timecourse_filter = (timecourse_weights @ timecourse_basis).squeeze(0)

    # shape (1, n_basis_feedback)
    feedback_weights = fitted_glm.feedback_weights

    # shape (1, n_basis_feedback) @ (n_basis_feedback, n_bins_filter)
    # -> (1, n_bins_filter) -> (n_bins_filter, )
    feedback_filter = (feedback_weights @ feedback_basis).squeeze(0)

    # shape (height, width)
    full_spatial_filter = fitted_glm.spatial_weights

    bias = fitted_glm.spatial_bias

    return FullResolutionCompactModel(
        full_spatial_filter,
        timecourse_filter,
        feedback_filter,
        (coupling_ids, coupling_filters_cell),
        bias
    )


def _extract_glm_params(glm_family: FittedGLMFamily,
                        cell_id: int,
                        model_fitted_patch_shape: Tuple[int, int],
                        stimulus_bounding_box) \
        -> CompactModel:
    fitted_glm = glm_family.fitted_models[cell_id]

    # shape (n_coupling_basis, n_bins_filter)
    coupling_basis = glm_family.coupling_basis

    # shape (n_feedback_basis, n_bins_filter)
    feedback_basis = glm_family.feedback_basis

    # shape (n_timecourse_basis, n_bins_filter)
    timecourse_basis = glm_family.timecourse_basis

    # shape (n_pixels, n_spatial_basis)
    spatial_basis = glm_family.spatial_basis

    # coupling_weights shape (n_coupled_cells, n_coupling_basis)
    # coupling_ids shape (n_coupled_cells, ); these are reference dataset cell ids
    coupling_weights, coupling_ids = fitted_glm.coupling_cells_weights

    # (n_coupled_cells, n_coupling_basis) @ (n_coupling_basis, n_bins_filter)
    # -> (n_coupled_cells, n_bins_filter)
    coupling_filters_cell = coupling_weights @ coupling_basis

    # shape (1, n_basis_stim_time)
    timecourse_weights = fitted_glm.timecourse_weights

    # shape (1, n_basis_stim_time) @ (n_timecourse_basis, n_bins_filter)
    # -> (1, n_bins_filter) -> (n_bins_filter, )
    timecourse_filter = (timecourse_weights @ timecourse_basis).squeeze(0)

    # shape (1, n_basis_feedback)
    feedback_weights = fitted_glm.feedback_weights

    # shape (1, n_basis_feedback) @ (n_basis_feedback, n_bins_filter)
    # -> (1, n_bins_filter) -> (n_bins_filter, )
    feedback_filter = (feedback_weights @ feedback_basis).squeeze(0)

    # shape (1, n_basis_stim_spat)
    spatial_weights = fitted_glm.spatial_weights

    # shape (1, n_basis_stim_spat) @ (n_spatial_basis, n_pixels)
    # -> (1, n_pixels) -> (n_pixels, ) -> (height, width)
    patch_spatial_filter = (spatial_basis @ spatial_weights.T).squeeze(1).reshape(model_fitted_patch_shape)

    bias = fitted_glm.spatial_bias

    return CompactModel(
        patch_spatial_filter,
        timecourse_filter,
        feedback_filter,
        (coupling_ids, coupling_filters_cell),
        bias,
        stimulus_bounding_box
    )


@dataclass
class PackedGLMTensors:
    spatial_filters: np.ndarray  # shape (n_cells, height, width)
    timecourse_filters: np.ndarray  # shape (n_cells, n_bins_filter)
    feedback_filters: np.ndarray  # shape (n_cells, n_bins_filter)
    coupling_filters: np.ndarray  # shape (n_cells, max_coupled_cells, n_bins_filter)
    coupling_indices: np.ndarray  # shape (n_cells, max_coupled_cells)
    bias: np.ndarray  # shape (n_cells, )


@dataclass
class FeedbackOnlyPackedGLMTensors:
    spatial_filters: np.ndarray  # shape (n_cells, height, width)
    timecourse_filters: np.ndarray  # shape (n_cells, n_bins_filter)
    feedback_filters: np.ndarray  # shape (n_cells, n_bins_filter)
    bias: np.ndarray  # shape (n_cells, )


def make_binomial_bin_count_tensor(bin_count_hyperparams: Dict[str, int],
                                   ordered_cells: OrderedMatchedCellsStruct) -> np.ndarray:
    cell_type_order = ordered_cells.get_cell_types()
    n_cells_of_type = ordered_cells.get_n_cells_by_type()

    empty_bin_count = np.zeros((sum(n_cells_of_type.values()),), dtype=np.float32)
    low = 0
    for ct in cell_type_order:
        n_cells = n_cells_of_type[ct]
        empty_bin_count[low:low + n_cells] = bin_count_hyperparams[ct]
        low += n_cells

    return empty_bin_count


def _compute_raw_coupling_indices(cell_ordering: OrderedMatchedCellsStruct,
                                  coupling_cell_ids: np.ndarray) \
        -> np.ndarray:
    '''

    :param cell_ordering:
    :param coupling_params:
    :return:
    '''

    coupling_indices = np.zeros_like(coupling_cell_ids, dtype=np.int64)
    for ix in range(coupling_cell_ids.shape[0]):
        coupling_indices[ix] = cell_ordering.get_concat_idx_for_cell_id(coupling_cell_ids[ix])
    return coupling_indices


def _build_padded_coupling_sel_and_filters(
        raw_coupling_filters: np.ndarray,
        raw_coupling_indices: np.ndarray,
        max_coupled_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Builds the coupling selector matrix and the coupling filter matrix

    Note that because all of the filters that correspond to unused slots
        are set to zero, we don't care about what the selection indices are
        for unused slots, since we multiply those (randomly-selected) spikes by
        zero anyway.

    :param raw_coupling_filters:
    :param raw_coupling_indices:
    :param max_coupled_cells:
    :return:
    '''

    n_coupling_filters, n_bins = raw_coupling_filters.shape

    padded_coupling_sel = np.zeros((max_coupled_cells,), dtype=np.int64)
    padded_coupling_sel[:n_coupling_filters] = raw_coupling_indices

    padded_coupling_filters = np.zeros((max_coupled_cells, n_bins), dtype=np.float32)
    padded_coupling_filters[:n_coupling_filters, :] = raw_coupling_filters

    return padded_coupling_sel, padded_coupling_filters


def convert_glm_family_to_np(glm_family: FittedGLMFamily,
                             spat_shape: Optional[Tuple[int, int]] = None) -> FittedGLMFamily:
    '''
    Helper function; converts glm_family components to np.ndarray from torch.Tensor on GPU
        if necessary (this is necessary due to a bug in the full res fitting code, and because
        refitting GLMs is a multi-day affair)

    Rebuilds the object no matter what, even if no conversions are necessary
    :param glm_family:
    :return:
    '''

    converted_params_dict = {}
    for cell_id, fg in glm_family.fitted_models.items():
        spat_w = fg.spatial_weights.detach().cpu().numpy() if isinstance(fg.spatial_weights,
                                                                         torch.Tensor) else fg.spatial_weights
        if spat_shape is not None:
            spat_w = spat_w.reshape(spat_shape)

        spat_b = fg.spatial_bias.detach().cpu().numpy() if isinstance(fg.spatial_bias,
                                                                      torch.Tensor) else fg.spatial_bias

        time_w = fg.timecourse_weights.detach().cpu().numpy() if isinstance(fg.timecourse_weights,
                                                                            torch.Tensor) else fg.timecourse_weights

        feedback_w = fg.feedback_weights.detach().cpu().numpy() if isinstance(fg.feedback_weights,
                                                                              torch.Tensor) else fg.feedback_weights

        coupling_w0 = fg.coupling_cells_weights[0].detach().cpu().numpy() if isinstance(fg.coupling_cells_weights[0],
                                                                                        torch.Tensor) else \
            fg.coupling_cells_weights[0]
        coupling_w1 = fg.coupling_cells_weights[1].detach().cpu().numpy() if isinstance(fg.coupling_cells_weights[1],
                                                                                        torch.Tensor) else \
            fg.coupling_cells_weights[1]

        converted_params_dict[cell_id] = FittedGLM(cell_id,
                                                   spat_w, spat_b, time_w, feedback_w, (coupling_w0, coupling_w1),
                                                   fg.fitting_params, fg.train_loss, fg.test_loss)

    # spatial_basis may be an np.ndarray, None, or torch.Tensor
    # convert to np.ndarray if torch.Tensor, otherwise pass through
    spatial_basis = glm_family.spatial_basis.detach().cpu().numpy() if isinstance(glm_family.spatial_basis,
                                                                                  torch.Tensor) else glm_family.spatial_basis
    timecourse_basis = glm_family.timecourse_basis.detach().cpu().numpy() if isinstance(glm_family.timecourse_basis,
                                                                                        torch.Tensor) else glm_family.timecourse_basis
    feedback_basis = glm_family.feedback_basis.detach().cpu().numpy() if isinstance(glm_family.feedback_basis,
                                                                                    torch.Tensor) else glm_family.feedback_basis
    coupling_basis = glm_family.coupling_basis.detach().cpu().numpy() if isinstance(glm_family.coupling_basis,
                                                                                    torch.Tensor) else glm_family.coupling_basis

    del glm_family

    return FittedGLMFamily(
        converted_params_dict,
        spatial_basis,
        timecourse_basis,
        feedback_basis,
        coupling_basis
    )


def reinflate_cropped_fb_only_glm_model(
        deflated_models: Dict[str, FittedFBOnlyGLMFamily],
        bounding_box_dict: Dict[str, List[CroppedSTABoundingBox]],
        cell_ordering: OrderedMatchedCellsStruct,
        raw_height: int,
        raw_width: int,
        downsample_factor: int = 1,
        crop_width_low: int = 0,
        crop_height_low: int = 0,
        crop_width_high: int = 0,
        crop_height_high: int = 0) -> FeedbackOnlyPackedGLMTensors:
    cell_type_order = cell_ordering.get_cell_types()

    bias_list = []  # type: List[np.ndarray]
    timecourse_filters_list = []  # type: List[np.ndarray]
    feedback_filters_list = []  # type: List[np.ndarray]
    spatial_filters_list = []  # type: List[np.ndarray]

    for cell_type in cell_type_order:

        glm_family = deflated_models[cell_type]

        for idx, cell_id in enumerate(cell_ordering.get_reference_cell_order(cell_type)):
            compact_model = _extract_cropped_glm_params(glm_family,
                                                        cell_id,
                                                        bounding_box_dict[cell_type][idx],
                                                        raw_height,
                                                        raw_width,
                                                        crop_height_low=crop_height_low,
                                                        crop_height_high=crop_height_high,
                                                        crop_width_low=crop_width_low,
                                                        crop_width_high=crop_width_high,
                                                        downsample_factor=downsample_factor)

            timecourse_filters_list.append(compact_model.timecourse_filter)
            feedback_filters_list.append(compact_model.feedback_filter)
            spatial_filters_list.append(compact_model.spatial_filter)
            bias_list.append(compact_model.bias)

    # now stack all of the arrays together
    spatial_filters_stacked = np.stack(spatial_filters_list, axis=0)
    timecourse_filters_stacked = np.stack(timecourse_filters_list, axis=0)
    feedback_filters_stacked = np.stack(feedback_filters_list, axis=0)
    bias_stacked = np.stack(bias_list, axis=0)

    packed_glm_tensors = FeedbackOnlyPackedGLMTensors(spatial_filters_stacked,
                                                      timecourse_filters_stacked,
                                                      feedback_filters_stacked,
                                                      bias_stacked)

    return packed_glm_tensors


def reinflate_cropped_glm_model(
        deflated_models: Dict[str, FittedGLMFamily],
        bounding_box_dict: Dict[str, List[CroppedSTABoundingBox]],
        cell_ordering: OrderedMatchedCellsStruct,
        raw_height: int,
        raw_width: int,
        downsample_factor: int = 1,
        crop_width_low: int = 0,
        crop_height_low: int = 0,
        crop_width_high: int = 0,
        crop_height_high: int = 0) -> PackedGLMTensors:
    '''

    :param deflated_models:
    :param bounding_box_dict:
    :param cell_ordering:
    :param raw_height:
    :param raw_width:
    :param downsample_factor:
    :param crop_width_low:
    :param crop_height_low:
    :param crop_width_high:
    :param crop_height_high:
    :return:
    '''
    max_coupled_cells = _count_max_coupled_cells(deflated_models)
    cell_type_order = cell_ordering.get_cell_types()

    idx_sel_list = []  # type: List[np.ndarray]
    coupling_filters_list = []  # type: List[np.ndarray]
    bias_list = []  # type: List[np.ndarray]
    timecourse_filters_list = []  # type: List[np.ndarray]
    feedback_filters_list = []  # type: List[np.ndarray]
    spatial_filters_list = []  # type: List[np.ndarray]

    for cell_type in cell_type_order:

        glm_family = deflated_models[cell_type]

        for idx, cell_id in enumerate(cell_ordering.get_reference_cell_order(cell_type)):
            compact_model = _extract_cropped_glm_params(glm_family,
                                                        cell_id,
                                                        bounding_box_dict[cell_type][idx],
                                                        raw_height,
                                                        raw_width,
                                                        crop_height_low=crop_height_low,
                                                        crop_height_high=crop_height_high,
                                                        crop_width_low=crop_width_low,
                                                        crop_width_high=crop_width_high,
                                                        downsample_factor=downsample_factor)

            raw_coupling_indices = _compute_raw_coupling_indices(cell_ordering,
                                                                 compact_model.coupling_params[0])

            coupling_idx_padded, coupling_filters_padded = _build_padded_coupling_sel_and_filters(
                compact_model.coupling_params[1],
                raw_coupling_indices,
                max_coupled_cells
            )

            idx_sel_list.append(coupling_idx_padded)
            coupling_filters_list.append(coupling_filters_padded)
            timecourse_filters_list.append(compact_model.timecourse_filter)
            feedback_filters_list.append(compact_model.feedback_filter)
            spatial_filters_list.append(compact_model.spatial_filter)
            bias_list.append(compact_model.bias)

    # now stack all of the arrays together
    spatial_filters_stacked = np.stack(spatial_filters_list, axis=0)
    idx_sel_stacked = np.stack(idx_sel_list, axis=0)
    coupling_filters_stacked = np.stack(coupling_filters_list, axis=0)
    timecourse_filters_stacked = np.stack(timecourse_filters_list, axis=0)
    feedback_filters_stacked = np.stack(feedback_filters_list, axis=0)
    bias_stacked = np.stack(bias_list, axis=0)

    packed_glm_tensors = PackedGLMTensors(spatial_filters_stacked,
                                          timecourse_filters_stacked,
                                          feedback_filters_stacked,
                                          coupling_filters_stacked,
                                          idx_sel_stacked,
                                          bias_stacked)

    return packed_glm_tensors


def translate_cropped_to_fullres_fittedglmfamily(
        cropped_models: Dict[str, FittedGLMFamily],
        bounding_boxes: Dict[str, List[CroppedSTABoundingBox]],
        cell_ordering: OrderedMatchedCellsStruct,
        fullres_height: int,
        fullres_width: int,
        downsample_factor: int = 1,
        crop_width_low: int = 0,
        crop_height_low: int = 0,
        crop_width_high: int = 0,
        crop_height_high: int = 0) -> Dict[str, FittedGLMFamily]:
    '''
    Converts cropped FittedGLMFamily into a
        full-resolution FittedGLMFamily

    Does so for every cell type, since we need to preserve the
        cell ordering anyway

    This is so that we can do spatial filter manipulations
        on the full-resolution filters, where we don't have
        to worry as much about the
    :param cropped_models:
    :param bounding_boxes:
    :param raw_height:
    :param raw_width:
    :param downsample_factor:
    :param crop_width_low:
    :param crop_height_low:
    :param crop_width_high:
    :param crop_height_high:
    :return:
    '''

    cell_type_order = cell_ordering.get_cell_types()
    full_res_glm_family_dict = {}
    for cell_type in cell_type_order:

        converted_model_dict = {}  # type: Dict[int, FittedGLM]

        glm_family = cropped_models[cell_type]
        for idx, cell_id in enumerate(cell_ordering.get_reference_cell_order(cell_type)):
            fitted_model = glm_family.fitted_models[cell_id]
            bounding_box = bounding_boxes[cell_type][idx]

            full_spatial_filter = np.zeros((fullres_height, fullres_width), dtype=np.float32)

            putback_slice_obj_h, putback_slice_obj_w = bounding_box.make_precropped_sliceobj(
                crop_hlow=crop_height_low,
                crop_hhigh=crop_height_high,
                crop_wlow=crop_width_low,
                crop_whigh=crop_width_high,
                downsample_factor=downsample_factor)

            full_spatial_filter[putback_slice_obj_h, putback_slice_obj_w] = fitted_model.spatial_weights

            fullres_fitted_glm = FittedGLM(
                cell_id,
                full_spatial_filter,
                fitted_model.spatial_bias,
                fitted_model.timecourse_weights,
                fitted_model.feedback_weights,
                fitted_model.coupling_cells_weights,
                fitted_model.fitting_params,
                fitted_model.train_loss,
                fitted_model.test_loss
            )

            converted_model_dict[cell_id] = fullres_fitted_glm

        fullres_glm_family = FittedGLMFamily(
            converted_model_dict,
            glm_family.spatial_basis,
            glm_family.timecourse_basis,
            glm_family.feedback_basis,
            glm_family.coupling_basis
        )

        full_res_glm_family_dict[cell_type] = fullres_glm_family

    return full_res_glm_family_dict


def make_full_res_packed_glm_tensors(ordered_cells: OrderedMatchedCellsStruct,
                                     fit_glms: Dict[str, FittedGLMFamily]) \
        -> PackedGLMTensors:
    max_coupled_cells = _count_max_coupled_cells(fit_glms)

    cell_type_ordering = ordered_cells.get_cell_types()

    idx_sel_list = []  # type: List[np.ndarray]
    coupling_filters_list = []  # type: List[np.ndarray]
    timecourse_filters_list = []  # type: List[np.ndarray]
    feedback_filters_list = []  # type: List[np.ndarray]
    spatial_filters_list = []  # type: List[np.ndarray]
    bias_list = []  # type: List[np.ndarray]

    for cell_type in cell_type_ordering:

        glm_family = fit_glms[cell_type]
        glm_family = convert_glm_family_to_np(glm_family)

        for idx, cell_id in enumerate(ordered_cells.get_reference_cell_order(cell_type)):
            compact_model = _extract_full_res_glm_params(glm_family, cell_id)

            bias = compact_model.bias

            raw_coupling_indices = _compute_raw_coupling_indices(ordered_cells,
                                                                 compact_model.coupling_params[0])

            coupling_idx_padded, coupling_filters_padded = _build_padded_coupling_sel_and_filters(
                compact_model.coupling_params[1],
                raw_coupling_indices,
                max_coupled_cells
            )

            idx_sel_list.append(coupling_idx_padded)
            coupling_filters_list.append(coupling_filters_padded)
            timecourse_filters_list.append(compact_model.timecourse_filter)
            feedback_filters_list.append(compact_model.feedback_filter)
            spatial_filters_list.append(compact_model.spatial_filter)
            bias_list.append(bias)

    # now stack all of the arrays together
    spatial_filters_stacked = np.stack(spatial_filters_list, axis=0)
    idx_sel_stacked = np.stack(idx_sel_list, axis=0)
    coupling_filters_stacked = np.stack(coupling_filters_list, axis=0)
    timecourse_filters_stacked = np.stack(timecourse_filters_list, axis=0)
    feedback_filters_stacked = np.stack(feedback_filters_list, axis=0)
    bias_stacked = np.stack(bias_list, axis=0)

    packed_glm_tensors = PackedGLMTensors(spatial_filters_stacked,
                                          timecourse_filters_stacked,
                                          feedback_filters_stacked,
                                          coupling_filters_stacked,
                                          idx_sel_stacked,
                                          bias_stacked)

    return packed_glm_tensors


def make_feedback_coupling_tensors(ordered_cells: OrderedMatchedCellsStruct,
                                   fit_glms: Dict[str, FittedGLMFamily],
                                   cropped_bbox_dict: Dict[str, List[CroppedSTABoundingBox]],
                                   downsample_factor: int = 1,
                                   crop_width_low: int = 0,
                                   crop_height_low: int = 0,
                                   crop_width_high: int = 0,
                                   crop_height_high: int = 0) \
        -> PackedGLMTensors:
    '''

    Implementation assumptions:
        Every cell in ordered_cells has a GLM fit to it, and that fit is stored in fit_glms

    :param ordered_cells: OrderedMatchedCellsStruct, contains cell ordering and typing information
    :param fit_glms: Previously fit GLMs for every cell type, str key is cell type.
        Must have already previously fit a GLM for every cell included in ordered_cells
    :return: PackedGLMTensors, containing the feedback, coupling, and timecourse filters necessary to run the
        GLM model in parallel over all of the cells

    '''

    max_coupled_cells = _count_max_coupled_cells(fit_glms)

    cell_type_ordering = ordered_cells.get_cell_types()

    idx_sel_list = []  # type: List[np.ndarray]
    coupling_filters_list = []  # type: List[np.ndarray]
    timecourse_filters_list = []  # type: List[np.ndarray]
    feedback_filters_list = []  # type: List[np.ndarray]
    spatial_filters_list = []  # type: List[np.ndarray]
    bias_list = []  # type: List[np.ndarray]

    for cell_type in cell_type_ordering:

        bounding_box_list = cropped_bbox_dict[cell_type]
        bbox_rectangle_dim = get_max_bounding_box_dimension(bounding_box_list,
                                                            downsample_factor=downsample_factor,
                                                            crop_width_low=crop_width_low,
                                                            crop_width_high=crop_width_high,
                                                            crop_height_low=crop_height_low,
                                                            crop_height_high=crop_height_high)

        glm_family = fit_glms[cell_type]

        for idx, cell_id in enumerate(ordered_cells.get_reference_cell_order(cell_type)):
            crop_bbox = bounding_box_list[idx]

            compact_model = _extract_glm_params(glm_family,
                                                cell_id,
                                                bbox_rectangle_dim,
                                                crop_bbox)

            spatial_filter = reinflate_spatial_filter(compact_model.spatial_filter,
                                                      crop_bbox,
                                                      downsample_factor=downsample_factor,
                                                      crop_width_low=crop_width_low,
                                                      crop_width_high=crop_width_high,
                                                      crop_height_low=crop_height_low,
                                                      crop_height_high=crop_height_high)

            bias = compact_model.bias

            raw_coupling_indices = _compute_raw_coupling_indices(ordered_cells,
                                                                 compact_model.coupling_params[0])

            coupling_idx_padded, coupling_filters_padded = _build_padded_coupling_sel_and_filters(
                compact_model.coupling_params[1],
                raw_coupling_indices,
                max_coupled_cells
            )

            idx_sel_list.append(coupling_idx_padded)
            coupling_filters_list.append(coupling_filters_padded)
            timecourse_filters_list.append(compact_model.timecourse_filter)
            feedback_filters_list.append(compact_model.feedback_filter)
            spatial_filters_list.append(spatial_filter)
            bias_list.append(bias)

    # now stack all of the arrays together
    spatial_filters_stacked = np.stack(spatial_filters_list, axis=0)
    idx_sel_stacked = np.stack(idx_sel_list, axis=0)
    coupling_filters_stacked = np.stack(coupling_filters_list, axis=0)
    timecourse_filters_stacked = np.stack(timecourse_filters_list, axis=0)
    feedback_filters_stacked = np.stack(feedback_filters_list, axis=0)
    bias_stacked = np.stack(bias_list, axis=0)

    packed_glm_tensors = PackedGLMTensors(spatial_filters_stacked,
                                          timecourse_filters_stacked,
                                          feedback_filters_stacked,
                                          coupling_filters_stacked,
                                          idx_sel_stacked,
                                          bias_stacked)

    return packed_glm_tensors


@torch.jit.script
def noreduce_bernoulli_neg_LL(
        generator_signal: torch.Tensor,
        observed_spikes: torch.Tensor) -> torch.Tensor:
    prod = generator_signal * observed_spikes
    log_sum_exp_term = torch.log1p(torch.exp(generator_signal))
    return log_sum_exp_term - prod

def noscript_noreduce_bernoulli_neg_LL(
        generator_signal: torch.Tensor,
        observed_spikes: torch.Tensor) -> torch.Tensor:
    prod = generator_signal * observed_spikes
    log_sum_exp_term = torch.log1p(torch.exp(generator_signal))
    return log_sum_exp_term - prod


def batch_bernoulli_spike_generation(gensig: torch.Tensor) -> torch.Tensor:
    '''
    Simulates Bernoulli spiking process for a single bin with sigmoid
        nonlinearity

    (can only do a single bin because GLM has feedback filter, so future
     probabilities depend on what happens in the present)

    :param spike_rate: shape (batch, n_repeats)
    :return: shape (batch, n_repeats), 0 or 1-valued tensor
    '''

    return torch.bernoulli(torch.sigmoid(gensig))


class FlashedModelRequiresPrecomputation(metaclass=ABCMeta):
    '''
    Common interface for all flashed reconstruction optimization
        problems that require precomputing quantities before
        running the optimization
    '''

    @abstractmethod
    def precompute_gensig_components(self,
                                     all_spikes: Any) -> None:
        raise NotImplementedError


class KnownSeparableTrialGLMLoss_Precompute(nn.Module,
                                            FlashedModelRequiresPrecomputation):

    def __init__(self,
                 stacked_spatial_filters: np.ndarray,
                 stacked_timecourse_filters: np.ndarray,
                 stacked_feedback_filters: np.ndarray,
                 stacked_coupling_filters: np.ndarray,
                 coupling_idx_sel: np.ndarray,
                 stacked_bias: np.ndarray,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        '''

        :param stacked_spatial_filters: shape (n_cells, height, width); spatial filters, one per cell/GLM
        :param stacked_timecourse_filters: shape (n_cells, n_bins_filter); timecourse filters, one per cell/GLM
        :param stacked_feedback_filters: shape (n_cells, n_bins_filter); feedback filters, one per cell/GLM
        :param stacked_coupling_filters: shape (n_cells, max_coupled_cells, n_bins_filter); coupling_filters,
        :param coupling_idx_sel: shape (n_cells, max_coupled_cells);
        :param stacked_bias: shape (n_cells, 1)
        :param stimulus_time_component: shape (n_bins, )
        '''
        super().__init__()

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.max_coupled_cells = stacked_coupling_filters.shape[1]
        self.n_pixels = self.height * self.width
        self.n_bins_total = stimulus_time_component.shape[0]
        self.n_bins_reconstruction = self.n_bins_total - self.n_bins_filter

        self.dtype = dtype
        self.spiking_loss_fn = spiking_loss_fn

        # fixed temporal component of the stimulus
        self.register_buffer('stim_time_component', torch.tensor(stimulus_time_component, dtype=dtype))

        ##### GLM parameters as torch buffers #############################################
        # shape (n_cells, n_pixels)
        stacked_flat_spat_filters = stacked_spatial_filters.reshape(self.n_cells, -1)
        self.register_buffer('stacked_flat_spat_filters', torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_timecourse_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_timecourse_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_timecourse_filters', torch.tensor(stacked_timecourse_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_feedback_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_feedback_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_feedback_filters', torch.tensor(stacked_feedback_filters, dtype=dtype))

        # shape (n_cells, max_coupled_cells, n_bins_filter)
        assert stacked_coupling_filters.shape == (self.n_cells, self.max_coupled_cells, self.n_bins_filter), \
            f'stacked_coupling_filters must have shape {(self.n_cells, self.max_coupled_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_coupling_filters', torch.tensor(stacked_coupling_filters, dtype=dtype))

        # shape (n_cells, 1)
        assert stacked_bias.shape == (self.n_cells, 1), f'stacked_bias must have shape {(self.n_cells, 1)}'
        self.register_buffer('stacked_bias', torch.tensor(stacked_bias, dtype=dtype))

        # shape (n_cells, max_coupled_cells), integer LongTensor
        assert coupling_idx_sel.shape == (self.n_cells, self.max_coupled_cells), \
            f'coupling_idx_sel must have shape {(self.n_cells, self.max_coupled_cells)}'
        self.register_buffer('coupled_sel', torch.tensor(coupling_idx_sel, dtype=torch.long))

        # Create buffers for pre-computed quantities (stuff that doesn't for a fixed spike train)
        self.register_buffer('precomputed_feedback_coupling_gensig',
                             torch.zeros((self.n_cells, self.n_bins_total - self.n_bins_filter + 1),
                                         dtype=torch.float32))

        self.register_buffer('precomputed_timecourse_contrib',
                             torch.zeros((self.n_cells, self.n_bins_total - self.n_bins_filter + 1)))

    def precompute_gensig_components(self,
                                     all_observed_spikes: torch.Tensor) -> None:
        fc_pc = self._precompute_feedback_coupling_gensig_components(all_observed_spikes)
        t_pc = self._precompute_timecourse_component(self.stim_time_component)

        self.precomputed_feedback_coupling_gensig.data[:] = fc_pc.data[:]
        self.precomputed_timecourse_contrib.data[:] = t_pc.data[:]

    def compute_feedback_exp_arg(self,
                                 all_observed_spikes: torch.Tensor) -> torch.Tensor:
        '''
        Computes the feedback component of the generator signal from the real data for every cell

        Implementation: 1D conv with groups
        :param all_observed_spikes:  shape (n_cells, n_bins_observed), observed spike trains for
            all of the cells, for just the image being reconstructed
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        # the feedback filters have shape
        # shape (n_cells, n_bins_filter), one for every cell

        # the observed spikes have shape (n_cells, n_bins_observed)

        # we want an output with shape (n_cells, n_bins_observed - n_bins_filter + 1)

        # (1, n_cells, n_bins_observed) \ast (n_cells, 1, n_bins_filter)
        # -> (1, n_cells, n_bins_observed - n_bins_filter + 1)
        # -> (n_cells, n_bins_observed - n_bins_filter + 1)
        conv_padded = F.conv1d(all_observed_spikes[None, :, :],
                               self.stacked_feedback_filters[:, None, :],
                               groups=self.n_cells).squeeze(0)

        return conv_padded

    def compute_coupling_exp_arg(self,
                                 all_observed_spikes: torch.Tensor) -> torch.Tensor:
        '''
        Computes the coupling component of the generator signal from the real data for every cell

        Implementation: Gather using the specified indices, then a 1D conv

        :param all_observed_spikes: shape (n_cells, n_bins_observed); observed spike trains for
            all of the cells, for just the image being reconstructed
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        _, n_bins_observed = all_observed_spikes.shape

        # we want an output set of spike trains with shape
        # (n_cells, max_coupled_cells, n_bins_observed)

        # we need to pick our data out of all_observed_spikes, which has shape
        # (n_cells, n_bins_observed)
        # using indices contained in self.coupled_sel, which has shape
        # (n_cells, max_coupled_cells), which contains indices from 0 to (n_cells - 1)

        # in order to use gather, the number of dimensions of each need to match
        # (we need 3 total dimensions)

        # shape (n_cells, max_coupled_cells, n_bins_observed), index dimension is dim1 max_coupled_cells
        indices_repeated = self.coupled_sel[:, :, None].expand(-1, -1, n_bins_observed)

        # shape (n_cells, n_cells, n_bins_observed)
        observed_spikes_repeated = all_observed_spikes[None, :, :].expand(self.n_cells, -1, -1)

        # shape (n_cells, max_coupled_cells, n_bins_observed)
        selected_spike_trains = torch.gather(observed_spikes_repeated, 1, indices_repeated)

        # now we have to do a 1D convolution with the coupling filters
        # the intended output has shape
        # (n_cells, n_bins_observed - n_bins_filter + 1)

        # the input is in selected_spike_trains and has shape
        # (n_cells, max_coupled_cells, n_bins_observed)

        # the coupling filters are in self.stacked_coupling_filters and have shape
        # (n_cells, n_coupled_cells, n_bins_filter)

        # this looks like it needs to be a grouped 1D convolution with some reshaping,
        # since we convolve along time, need to sum over the coupled cells, but have
        # an extra batch dimension

        # we do a 1D convolution, with n_cells different groups

        # shape (1, n_cells * max_coupled_cells, n_bins_observed)
        selected_spike_trains_reshape = selected_spike_trains.reshape(1, -1, n_bins_observed)

        # (1, n_cells * max_coupled_cells, n_bins_observed) \ast (n_cells, n_coupled_cells, n_bins_filter)
        # -> (1, n_cells, n_bins_filter) -> (n_cells, n_bins_observed - n_bins_filter + 1)
        coupling_conv = F.conv1d(selected_spike_trains_reshape,
                                 self.stacked_coupling_filters,
                                 groups=self.n_cells).squeeze(0)

        return coupling_conv

    def _precompute_feedback_coupling_gensig_components(self,
                                                        all_cells_spiketrain: torch.Tensor) -> torch.Tensor:
        '''
        Precompute the feedback and coupling components of the generator signal,
            since given a fixed observedspike train, these components do not depend
            on the stimulus at all. When doing reconstruction, the only thing that
            changes is the spatial component of the stimulus.
        :param all_cells_spiketrain:
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1); sum of the coupling and feedback
            components to the generator signal
        '''

        with torch.no_grad():
            coupling_component = self.compute_coupling_exp_arg(all_cells_spiketrain)
            feedback_component = self.compute_feedback_exp_arg(all_cells_spiketrain)
            return coupling_component + feedback_component

    def _precompute_timecourse_component(self,
                                         stim_time: torch.Tensor) -> torch.Tensor:
        '''
        Convolves the timecourse of each cell with the time component of the stimulus

        :param stim_time: shape (n_bins_observed, )
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1);
            convolution of the timecourse of each cell with the time component of the stimulus
        '''

        with torch.no_grad():
            # shape (1, 1, n_bins_observed) \ast (n_cells, 1, n_bins_filter)
            # -> (1, n_cells, n_bins_observed - n_bins_filter + 1)
            conv_extra_dims = F.conv1d(stim_time[None, None, :],
                                       self.stacked_timecourse_filters[:, None, :])
            return conv_extra_dims.squeeze(0)

    def compute_stimulus_spat_component(self,
                                        spat_stim_flat: torch.Tensor) -> torch.Tensor:
        '''
        Applies the spatial filters and biases to the stimuli

        :param spat_stim_flat: shape (n_pixels, )
        :return: shape (n_cells, ) ;
            result of applying the spatial filter for each cell onto each stimulus image
        '''

        # shape (1, n_pixels) @ (n_pixels, n_cells)
        # -> (1, n_cells) -> (n_cells, )
        spat_filt_applied = (spat_stim_flat[None, :] @ self.stacked_flat_spat_filters.T).squeeze(0)
        return spat_filt_applied

    def gen_sig(self, image_flattened: torch.Tensor) -> torch.Tensor:
        '''
        The generator signal no longer explicitly depends on the observed spike train,
            since for a fixed spike train we can precompute that component

        :param image_flattened: shape (n_pixels, )
        :param observed_spikes: shape (n_cells, n_bins_observed)
        :return: shape (n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        # shape (n_cells, )
        gensig_spat_component = self.compute_stimulus_spat_component(image_flattened)

        # shape (n_cells, 1) * (n_cells, n_bins_observed - n_bins_filter + 1)
        # -> (n_cells, n_bins_observed - n_bins_filter + 1)
        gensig_spat_time = gensig_spat_component[:, None] * self.precomputed_timecourse_contrib + self.stacked_bias

        # shape (n_cells, n_bins_observed - n_bins_filter + 1)
        total_gensig = gensig_spat_time + self.precomputed_feedback_coupling_gensig

        return total_gensig

    def image_loss(self, image_flattened: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_flattened: shape (n_pixels, )
        :param observed_spikes_null_padded:  torch.Tensor, shape (n_cells, n_bins_observed), first row is the
            NULL cell with no spikes observed
        '''

        # shape (n_cells, n_bins_observed - n_bins_filter + 1)
        generator_signal = self.gen_sig(image_flattened)

        loss_per_timestep = self.spiking_loss_fn(generator_signal[:, :-1],
                                                 observed_spikes[:, self.n_bins_filter:])
        return loss_per_timestep

    def calculate_loss(self, image_imshape: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_imshape: shape (height, width)
        :param observed_spikes:  shape (n_cells, n_bins_observed)
        :return:
        '''

        # shape (n_pixels, )
        image_flat = image_imshape.reshape(self.n_pixels)

        loss_per_bin = self.image_loss(image_flat, observed_spikes)

        return torch.sum(loss_per_bin, dim=0)

    def image_gradient(self, image_imshape: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_imshape: shape (height, width)
        :param observed_spikes: shape (n_cells, n_bins_observed)
        :return: shape (height, width)
        '''
        image_flattened = image_imshape.reshape(self.height * self.width, )
        image_flattened.requires_grad_(True)

        # shape (n_bins_reconstructed, )
        loss_per_bin = self.image_loss(image_flattened, observed_spikes)

        total_loss = torch.sum(loss_per_bin, dim=0)

        gradient_image_flat, = autograd.grad(total_loss, image_flattened)

        gradient_imshape = gradient_image_flat.reshape(self.height, self.width)

        return -gradient_imshape


class KnownSeparableTrialGLM_Precompute_HQS_XProb(SingleUnconstrainedProblem,
                                                  HQS_X_Problem,
                                                  FlashedModelRequiresPrecomputation):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 stacked_spatial_filters: np.ndarray,
                 stacked_timecourse_filters: np.ndarray,
                 stacked_feedback_filters: np.ndarray,
                 stacked_coupling_filters: np.ndarray,
                 coupling_idx_sel: np.ndarray,
                 stacked_bias: np.ndarray,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 hqs_rho: float,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.rho = hqs_rho
        self.valid_prox_arg = False

        self.glm_encoding_loss = KnownSeparableTrialGLMLoss_Precompute(
            stacked_spatial_filters,
            stacked_timecourse_filters,
            stacked_feedback_filters,
            stacked_coupling_filters,
            coupling_idx_sel,
            stacked_bias,
            stimulus_time_component,
            spiking_loss_fn,
            dtype=dtype
        )

        self.n_cells, self.height, self.width = stacked_spatial_filters.shape

        ### HQS constants #####################################################
        self.register_buffer('z_const_tensor', torch.zeros((self.height, self.width), dtype=dtype))

        ### OPTIMIZATION VARIABLES ############################################
        self.x_image = nn.Parameter(torch.empty((self.height, self.width), dtype=dtype),
                                    requires_grad=True)
        nn.init.normal_(self.x_image, mean=0.0, std=0.25)  # FIXME we may want a different noise initialization strategy

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def precompute_gensig_components(self,
                                     all_observed_spikes: torch.Tensor) -> None:
        return self.glm_encoding_loss.precompute_gensig_components(all_observed_spikes)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        # shape (height, width)
        image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape (n_cells, )
        observed_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape ()
        encoding_loss = self.glm_encoding_loss.calculate_loss(image_imshape, observed_spikes)

        prox_diff = image_imshape - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff)

        return encoding_loss + prox_loss

    def reinitialize_variables(self,
                               initialized_z_const: Optional[torch.Tensor] = None) -> None:
        # nn.init.normal_(self.z_const_tensor, mean=0.0, std=1.0)
        if initialized_z_const is None:
            self.z_const_tensor.data[:] = 0.0
        else:
            self.z_const_tensor.data[:] = initialized_z_const.data[:]

        nn.init.normal_(self.x_image, mean=0.0, std=0.5)

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def get_reconstructed_image(self) -> np.ndarray:
        return self.x_image.detach().cpu().numpy()


class BatchKnownSeparableTrialGLMLoss_Precompute(nn.Module,
                                                 FlashedModelRequiresPrecomputation):

    def __init__(self,
                 batch: int,
                 packed_model_tensors: PackedGLMTensors,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        stacked_spatial_filters = packed_model_tensors.spatial_filters
        stacked_timecourse_filters = packed_model_tensors.timecourse_filters
        stacked_feedback_filters = packed_model_tensors.feedback_filters
        stacked_coupling_filters = packed_model_tensors.coupling_filters
        coupling_idx_sel = packed_model_tensors.coupling_indices
        stacked_bias = packed_model_tensors.bias

        self.batch = batch
        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.max_coupled_cells = stacked_coupling_filters.shape[1]
        self.n_pixels = self.height * self.width
        self.n_bins_total = stimulus_time_component.shape[0]
        self.n_bins_reconstruction = self.n_bins_total - self.n_bins_filter

        self.dtype = dtype
        self.spiking_loss_fn = spiking_loss_fn

        # fixed temporal component of the stimulus
        self.register_buffer('stim_time_component', torch.tensor(stimulus_time_component, dtype=dtype))

        ##### GLM parameters as torch buffers #############################################
        # shape (n_cells, n_pixels)
        stacked_flat_spat_filters = stacked_spatial_filters.reshape(self.n_cells, -1)
        self.register_buffer('stacked_flat_spat_filters', torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_timecourse_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_timecourse_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_timecourse_filters', torch.tensor(stacked_timecourse_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_feedback_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_feedback_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_feedback_filters', torch.tensor(stacked_feedback_filters, dtype=dtype))

        # shape (n_cells, max_coupled_cells, n_bins_filter)
        assert stacked_coupling_filters.shape == (self.n_cells, self.max_coupled_cells, self.n_bins_filter), \
            f'stacked_coupling_filters must have shape {(self.n_cells, self.max_coupled_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_coupling_filters', torch.tensor(stacked_coupling_filters, dtype=dtype))

        # shape (n_cells, 1)
        assert stacked_bias.shape == (self.n_cells, 1), f'stacked_bias must have shape {(self.n_cells, 1)}'
        self.register_buffer('stacked_bias', torch.tensor(stacked_bias, dtype=dtype))

        # shape (n_cells, max_coupled_cells), integer LongTensor
        assert coupling_idx_sel.shape == (self.n_cells, self.max_coupled_cells), \
            f'coupling_idx_sel must have shape {(self.n_cells, self.max_coupled_cells)}'
        self.register_buffer('coupled_sel', torch.tensor(coupling_idx_sel, dtype=torch.long))

        # Create buffers for pre-computed quantities (stuff that doesn't for a fixed spike train)
        self.register_buffer('precomputed_feedback_coupling_gensig',
                             torch.zeros((self.batch, self.n_cells, self.n_bins_total - self.n_bins_filter + 1),
                                         dtype=torch.float32))

        self.register_buffer('precomputed_timecourse_contrib',
                             torch.zeros((self.n_cells, self.n_bins_total - self.n_bins_filter + 1)))

    def precompute_gensig_components(self,
                                     all_observed_spikes: torch.Tensor) -> None:
        with torch.no_grad():
            feedback_component = flashed_recons_compute_feedback_exp_arg(
                self.stacked_feedback_filters, all_observed_spikes)
            coupling_component = flashed_recons_compute_coupling_exp_arg(
                self.stacked_coupling_filters, self.coupled_sel, all_observed_spikes)
            fc_pc = feedback_component + coupling_component

            t_pc = flashed_recons_compute_timecourse_component(
                self.stacked_timecourse_filters, self.stim_time_component)

        self.precomputed_feedback_coupling_gensig.data[:] = fc_pc.data[:]
        self.precomputed_timecourse_contrib.data[:] = t_pc.data[:]

    def compute_stimulus_spat_component(self,
                                        spat_stim_flat: torch.Tensor) -> torch.Tensor:
        '''
        Applies the spatial filters and biases to the stimuli

        :param spat_stim_flat: shape (batch, n_pixels)
        :return: shape (n_cells, ) ;
            result of applying the spatial filter for each cell onto each stimulus image
        '''

        # shape (batch, n_pixels) @ (n_pixels, n_cells)
        # -> (batch, n_cells) -> (n_cells, )
        spat_filt_applied = (spat_stim_flat @ self.stacked_flat_spat_filters.T)
        return spat_filt_applied

    def gen_sig(self, image_flattened: torch.Tensor) -> torch.Tensor:
        '''
        The generator signal no longer explicitly depends on the observed spike train,
            since for a fixed spike train we can precompute that component

        :param image_flattened: shape (batch, n_pixels)
        :return: shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        # shape (batch, n_cells)
        gensig_spat_component = self.compute_stimulus_spat_component(image_flattened)

        # shape (batch, n_cells, 1) * (1, n_cells, n_bins_observed - n_bins_filter + 1)
        # -> (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        gensig_spat_time = gensig_spat_component[:, :, None] * self.precomputed_timecourse_contrib[None, :, :] \
                           + self.stacked_bias[None, :, :]

        # shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        total_gensig = gensig_spat_time + self.precomputed_feedback_coupling_gensig

        return total_gensig

    def image_loss(self, image_flattened: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_flattened: shape (batch, n_pixels)
        :param observed_spikes_null_padded:  torch.Tensor, shape (batch, n_cells, n_bins_observed), first row is the
            NULL cell with no spikes observed
        '''

        # shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        generator_signal = self.gen_sig(image_flattened)

        # shape (batch, n_bins_observed - n_bins_filter + 1)
        loss_per_timestep = self.spiking_loss_fn(generator_signal[:, :, :-1],
                                                 observed_spikes[:, :, self.n_bins_filter:])
        return loss_per_timestep

    def forward(self, image_imshape, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_imshape: shape (batch, height, width)
        :param observed_spikes: shape (batch, n_cells, n_bins_observed)
        :return:
        '''

        image_flat = image_imshape.reshape(self.batch, self.n_pixels)

        # shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        generator_signal = self.gen_sig(image_flat)

        loss_per_timestep = self.spiking_loss_fn(generator_signal[:, :, :-1],
                                                 observed_spikes[:, :, self.n_bins_filter:])
        return torch.sum(loss_per_timestep, dim=(2, 1))


class BatchKnownSeparableFBOnlyTrialGLMLoss_Precompute(nn.Module,
                                                       FlashedModelRequiresPrecomputation):

    def __init__(self,
                 batch: int,
                 model_params: FeedbackOnlyPackedGLMTensors,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        stacked_spatial_filters = model_params.spatial_filters
        stacked_timecourse_filters = model_params.timecourse_filters
        stacked_feedback_filters = model_params.feedback_filters
        stacked_bias = model_params.bias

        self.batch = batch
        self.n_cells, self.height, self.width = stacked_spatial_filters.shape
        self.n_bins_filter = stacked_timecourse_filters.shape[1]
        self.n_pixels = self.height * self.width
        self.n_bins_total = stimulus_time_component.shape[0]
        self.n_bins_reconstruction = self.n_bins_total - self.n_bins_filter

        self.dtype = dtype
        self.spiking_loss_fn = spiking_loss_fn

        # fixed temporal component of the stimulus
        self.register_buffer('stim_time_component', torch.tensor(stimulus_time_component, dtype=dtype))

        ##### GLM parameters as torch buffers #############################################
        # shape (n_cells, n_pixels)
        stacked_flat_spat_filters = stacked_spatial_filters.reshape(self.n_cells, -1)
        self.register_buffer('stacked_flat_spat_filters', torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_timecourse_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_timecourse_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_timecourse_filters', torch.tensor(stacked_timecourse_filters, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        assert stacked_feedback_filters.shape == (self.n_cells, self.n_bins_filter), \
            f'stacked_feedback_filters must have shape {(self.n_cells, self.n_bins_filter)}'
        self.register_buffer('stacked_feedback_filters', torch.tensor(stacked_feedback_filters, dtype=dtype))

        # shape (n_cells, 1)
        assert stacked_bias.shape == (self.n_cells, 1), f'stacked_bias must have shape {(self.n_cells, 1)}'
        self.register_buffer('stacked_bias', torch.tensor(stacked_bias, dtype=dtype))

        # Create buffers for pre-computed quantities (stuff that doesn't for a fixed spike train)
        self.register_buffer('precomputed_feedback_gensig',
                             torch.zeros((self.batch, self.n_cells, self.n_bins_total - self.n_bins_filter + 1),
                                         dtype=torch.float32))

        self.register_buffer('precomputed_timecourse_contrib',
                             torch.zeros((self.n_cells, self.n_bins_total - self.n_bins_filter + 1)))

    def precompute_gensig_components(self,
                                     all_observed_spikes: torch.Tensor) -> None:
        with torch.no_grad():
            fc_pc = flashed_recons_compute_feedback_exp_arg(
                self.stacked_feedback_filters, all_observed_spikes)
            t_pc = flashed_recons_compute_timecourse_component(
                self.stacked_timecourse_filters, self.stim_time_component)

        self.precomputed_feedback_gensig.data[:] = fc_pc.data[:]
        self.precomputed_timecourse_contrib.data[:] = t_pc.data[:]

    def compute_stimulus_spat_component(self,
                                        spat_stim_flat: torch.Tensor) -> torch.Tensor:
        '''
        Applies the spatial filters and biases to the stimuli

        :param spat_stim_flat: shape (batch, n_pixels)
        :return: shape (n_cells, ) ;
            result of applying the spatial filter for each cell onto each stimulus image
        '''

        # shape (batch, n_pixels) @ (n_pixels, n_cells)
        # -> (batch, n_cells) -> (n_cells, )
        spat_filt_applied = (spat_stim_flat @ self.stacked_flat_spat_filters.T)
        return spat_filt_applied

    def gen_sig(self, image_flattened: torch.Tensor) -> torch.Tensor:
        '''
        The generator signal no longer explicitly depends on the observed spike train,
            since for a fixed spike train we can precompute that component

        :param image_flattened: shape (batch, n_pixels)
        :param observed_spikes: shape (batch, n_cells, n_bins_observed)
        :return: shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        '''

        # shape (batch, n_cells)
        gensig_spat_component = self.compute_stimulus_spat_component(image_flattened)

        # shape (batch, n_cells, 1) * (1, n_cells, n_bins_observed - n_bins_filter + 1)
        # -> (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        gensig_spat_time = gensig_spat_component[:, :, None] * self.precomputed_timecourse_contrib[None, :, :] \
                           + self.stacked_bias[None, :, :]

        # shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        total_gensig = gensig_spat_time + self.precomputed_feedback_gensig

        return total_gensig

    def image_loss(self, image_flattened: torch.Tensor, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_flattened: shape (batch, n_pixels)
        :param observed_spikes_null_padded:  torch.Tensor, shape (batch, n_cells, n_bins_observed), first row is the
            NULL cell with no spikes observed
        '''

        # shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
        generator_signal = self.gen_sig(image_flattened)

        # shape (batch, n_bins_observed - n_bins_filter + 1)
        loss_per_timestep = self.spiking_loss_fn(generator_signal[:, :, :-1],
                                                 observed_spikes[:, :, self.n_bins_filter:])
        return loss_per_timestep

    def forward(self, image_imshape, observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param image_imshape: shape (batch, height, width)
        :param observed_spikes: shape (batch, n_cells, n_bins_observed)
        :return:
        '''

        image_flat = image_imshape.reshape(self.batch, self.n_pixels)

        # shape (batch, n_Cells, n_bins_observed - n_bins_filter + 1)
        generator_signal = self.gen_sig(image_flat)

        loss_per_timestep = self.spiking_loss_fn(generator_signal[:, :, :-1],
                                                 observed_spikes[:, :, self.n_bins_filter:])
        return torch.sum(loss_per_timestep, dim=(2, 1))


class BatchKnownSeparable_TrialGLM_ProxProblem(BatchParallelUnconstrainedProblem,
                                               BatchParallel_HQS_X_Problem,
                                               FlashedModelRequiresPrecomputation):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 batch: int,
                 packed_model_tensors: Union[PackedGLMTensors, FeedbackOnlyPackedGLMTensors],
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 rho: float = 1.0,  # placeholder, allows object to be created before we know rho
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.batch_size = batch
        self.rho = rho

        if isinstance(packed_model_tensors, PackedGLMTensors):
            self.glm_loss_calc = BatchKnownSeparableTrialGLMLoss_Precompute(
                batch,
                packed_model_tensors,
                stimulus_time_component,
                spiking_loss_fn,
                dtype=dtype
            )  # type: BatchKnownSeparableTrialGLMLoss_Precompute
        else:
            self.glm_loss_calc = BatchKnownSeparableFBOnlyTrialGLMLoss_Precompute(
                batch,
                packed_model_tensors,
                stimulus_time_component,
                spiking_loss_fn,
                dtype=dtype
            )  # type: BatchKnownSeparableFBOnlyTrialGLMLoss_Precompute

        self.height, self.width = self.glm_loss_calc.height, self.glm_loss_calc.width

        #### CONSTANTS ###############################################
        self.register_buffer('z_const_tensor', torch.empty((self.batch_size, self.height, self.width), dtype=dtype))

        #### OPTIM VARIABLES #########################################
        # OPT VARIABLE 0: image, shape (batch, height, width)
        self.image = nn.Parameter(torch.empty((self.batch_size, self.height, self.width),
                                              dtype=dtype))
        nn.init.uniform_(self.image, a=-1e-2, b=1e-2)

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def reinitialize_variables(self,
                               initialized_z_const: Optional[torch.Tensor] = None) -> None:
        # nn.init.normal_(self.z_const_tensor, mean=0.0, std=1.0)
        if initialized_z_const is None:
            self.z_const_tensor.data[:] = 0.0
        else:
            self.z_const_tensor.data[:] = initialized_z_const.data[:]

        nn.init.normal_(self.image, mean=0.0, std=0.5)

    def get_reconstructed_image(self) -> np.ndarray:
        return self.image.detach().cpu().numpy()

    @property
    def n_problems(self) -> int:
        return self.batch_size

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        '''

        :param observed_spikes: shape (batch, n_cells, n_bins_observed)
        :return:
        '''
        self.glm_loss_calc.precompute_gensig_components(observed_spikes)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return:
        '''

        # shape (batch, height, width)
        batched_image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape (batch, n_cells, n_bins)
        batched_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (batch, )
        encoding_loss = self.glm_loss_calc(batched_image_imshape,
                                           batched_spikes)

        # shape (batch, )
        prox_diff = batched_image_imshape - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

        return encoding_loss + prox_loss

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]


class MixNMatch_BatchKnownSeparableTrialGLMLoss_Precompute(nn.Module,
                                                           FlashedModelRequiresPrecomputation):

    def __init__(self,
                 batch: int,
                 typed_packed_glm_tensors: Dict[str, PackedGLMTensors],
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.loss_module_dict = nn.ModuleDict({
            key: BatchKnownSeparableTrialGLMLoss_Precompute(batch,
                                                            val,
                                                            stimulus_time_component,
                                                            spiking_loss_fn,
                                                            dtype=dtype)
            for key, val in typed_packed_glm_tensors.items()
        })

        _firstkey = list(typed_packed_glm_tensors.keys())[0]
        self.height = self.loss_module_dict[_firstkey].height
        self.width = self.loss_module_dict[_firstkey].width

    def precompute_gensig_components(self,
                                     typed_perturbed_spikes: Dict[str, torch.Tensor]) -> None:
        for key, module in self.loss_module_dict.items():
            module.precompute_gensig_components(typed_perturbed_spikes[key])

    def forward(self,
                image_imshape: torch.Tensor,
                typed_perturbed_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
        '''

        :param image_imshape: shape (batch, height, width)
        :param observed_spikes: shape (batch, n_cells, n_bins_observed)
        :return:
        '''

        acc_loss_by_type = []
        for key, module in self.loss_module_dict.items():
            loss_per_timestep = module(image_imshape, typed_perturbed_spikes[key])
            acc_loss_by_type.append(loss_per_timestep)

        return torch.sum(torch.stack(acc_loss_by_type, dim=1), dim=1)


class MixNMatch_BatchKnownSeparable_Trial_GLM_ProxProblem(BatchParallelUnconstrainedProblem,
                                                          BatchParallel_HQS_X_Problem,
                                                          FlashedModelRequiresPrecomputation):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 batch: int,
                 typed_packed_model_tensors: Dict[str, PackedGLMTensors],
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 rho: float = 1.0,
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.batch_size = batch
        self.rho = rho

        self.glm_loss_calc = MixNMatch_BatchKnownSeparableTrialGLMLoss_Precompute(
            batch,
            typed_packed_model_tensors,
            stimulus_time_component,
            spiking_loss_fn,
            dtype=dtype
        )

        self.height, self.width = self.glm_loss_calc.height, self.glm_loss_calc.width

        #### CONSTANTS ###############################################
        self.register_buffer('z_const_tensor', torch.empty((self.batch_size, self.height, self.width), dtype=dtype))

        #### OPTIM VARIABLES #########################################
        # OPT VARIABLE 0: image, shape (batch, height, width)
        self.image = nn.Parameter(torch.empty((self.batch_size, self.height, self.width),
                                              dtype=dtype))
        nn.init.uniform_(self.image, a=-1e-2, b=1e-2)

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def reinitialize_variables(self,
                               initialized_z_const: Optional[torch.Tensor] = None) -> None:
        # nn.init.normal_(self.z_const_tensor, mean=0.0, std=1.0)
        if initialized_z_const is None:
            self.z_const_tensor.data[:] = 0.0
        else:
            self.z_const_tensor.data[:] = initialized_z_const.data[:]

        nn.init.normal_(self.image, mean=0.0, std=0.5)

    def get_reconstructed_image(self) -> np.ndarray:
        return self.image.detach().cpu().numpy()

    @property
    def n_problems(self) -> int:
        return self.batch_size

    def precompute_gensig_components(self, typed_perturbed_spikes: Dict[str, torch.Tensor]) -> None:
        '''

        :param observed_spikes: shape (batch, n_cells, n_bins_observed)
        :return:
        '''
        self.glm_loss_calc.precompute_gensig_components(typed_perturbed_spikes)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return:
        '''

        # shape (batch, height, width)
        batched_image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape Dict of (batch, n_cells, n_bins)
        batched_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (batch, )
        encoding_loss = self.glm_loss_calc(batched_image_imshape,
                                           batched_spikes)

        # shape (batch, )
        prox_diff = batched_image_imshape - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

        return encoding_loss + prox_loss

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]


class BatchParallelPatchGaussian1FPriorGLMReconstruction(BatchParallelUnconstrainedProblem,
                                                         FlashedModelRequiresPrecomputation):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 batch_size: int,
                 patch_zca_matrix: np.ndarray,
                 gaussian_prior_lambda: float,
                 packed_glm_tensors: PackedGLMTensors,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32,
                 patch_stride: int = 1):
        super().__init__()

        self.batch = batch_size

        self.gaussian_prior_lambda = gaussian_prior_lambda
        self.batch_prior_callable = ConvPatch1FGaussianPrior(patch_zca_matrix,
                                                             patch_stride=patch_stride,
                                                             dtype=dtype)

        self.encoding_loss_callable = BatchKnownSeparableTrialGLMLoss_Precompute(
            batch_size,
            packed_glm_tensors,
            stimulus_time_component,
            spiking_loss_fn,
            dtype=dtype)

        self.height, self.width = self.encoding_loss_callable.height, self.encoding_loss_callable.width

        self.reconstructed_image = nn.Parameter(torch.empty((batch_size, self.height, self.width), dtype=dtype),
                                                requires_grad=True)
        nn.init.uniform_(self.reconstructed_image, a=-0.1, b=0.1)

    def get_reconstructed_image(self) -> np.ndarray:
        return self.reconstructed_image.detach().cpu().numpy()

    @property
    def n_problems(self) -> int:
        return self.batch

    def precompute_gensig_components(self, observed_spikes: torch.Tensor) -> None:
        self.encoding_loss_callable.precompute_gensig_components(observed_spikes)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        # shape (batch, height, width)
        batched_image_imshape = args[self.IMAGE_IDX_ARGS]

        # shape (batch, n_cells)
        batched_spikes = kwargs[self.OBSERVED_SPIKES_KWARGS]

        # shape (batch, )
        encoding_loss = self.encoding_loss_callable(batched_image_imshape,
                                                    batched_spikes)

        # shape (batch, )
        gaussian_prior_penalty = 0.5 * self.gaussian_prior_lambda * self.batch_prior_callable(batched_image_imshape)

        # shape (batch, )
        return encoding_loss + gaussian_prior_penalty
