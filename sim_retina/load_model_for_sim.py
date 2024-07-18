from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np

from denoise_inverse_alg.glm_inverse_alg import _compute_raw_coupling_indices
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from optimization_encoder.trial_glm import FittedGLMFamily, FittedFBOnlyGLMFamily


@dataclass
class SingleCellGLMForSim:

    cell_id: int

    cropped_spatial_filter: np.ndarray
    stimulus_timecourse: np.ndarray

    bias: np.ndarray

    feedback_filter: np.ndarray

    coupling_params: Tuple[np.ndarray, np.ndarray]


def reinflate_full_glm_model_for_sim(full_glm_families: Dict[str, FittedGLMFamily],
                                     cells_ordered: OrderedMatchedCellsStruct,
                                     target_cell_iden: Tuple[str, int],
                                     compute_indices: bool = True) \
        -> SingleCellGLMForSim:
    cell_type, cell_id = target_cell_iden

    glm_family = full_glm_families[cell_type]

    coupling_basis = glm_family.coupling_basis
    feedback_basis = glm_family.feedback_basis
    timecourse_basis = glm_family.timecourse_basis

    fitted_glm = glm_family.fitted_models[cell_id]

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

    # coupling_weights shape (n_coupled_cells, n_coupling_basis)
    # coupling_ids shape (n_coupled_cells, ); these are reference dataset cell ids
    coupling_weights, coupling_ids = fitted_glm.coupling_cells_weights

    # (n_coupled_cells, n_coupling_basis) @ (n_coupling_basis, n_bins_filter)
    # -> (n_coupled_cells, n_bins_filter)
    coupling_filters_cell = coupling_weights @ coupling_basis

    if compute_indices:
        coupling_indices = _compute_raw_coupling_indices(cells_ordered,
                                                         coupling_ids)

        return SingleCellGLMForSim(
            cell_id,
            fitted_glm.spatial_weights,
            timecourse_filter,
            fitted_glm.spatial_bias,
            feedback_filter,
            (coupling_filters_cell, coupling_indices)
        )
    else:
        return SingleCellGLMForSim(
            cell_id,
            fitted_glm.spatial_weights,
            timecourse_filter,
            fitted_glm.spatial_bias,
            feedback_filter,
            (coupling_filters_cell, coupling_ids)
        )


@dataclass
class SingleCellUncoupledGLMForSim:
    cell_id: int
    cropped_spatial_filter: np.ndarray
    stimulus_timecourse: np.ndarray

    bias: np.ndarray

    feedback_filter: np.ndarray


def reinflate_uncoupled_glm_model_for_sim(
        full_glm_families: Dict[str, FittedFBOnlyGLMFamily],
        cells_ordered: OrderedMatchedCellsStruct,
        target_cell_iden: Tuple[str, int]) \
        -> SingleCellUncoupledGLMForSim:
    cell_type, cell_id = target_cell_iden

    glm_family = full_glm_families[cell_type]

    feedback_basis = glm_family.feedback_basis
    timecourse_basis = glm_family.timecourse_basis

    fitted_glm = glm_family.fitted_models[cell_id]

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

    return SingleCellUncoupledGLMForSim(
        cell_id,
        fitted_glm.spatial_weights,
        timecourse_filter,
        fitted_glm.spatial_bias,
        feedback_filter,
    )
