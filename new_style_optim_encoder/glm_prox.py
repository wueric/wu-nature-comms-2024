from typing import Tuple

import torch


def _time_model_prox_project_variables(
        coupling_w: torch.Tensor,
        feedback_w: torch.Tensor,
        timecourse_w: torch.Tensor,
        bias: torch.Tensor,
        coupling_aux: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        # shape (n_coupled_cells, )
        coupling_norm = torch.linalg.norm(coupling_w, dim=1)

        # shape (n_coupled_cells, ), binary-valued
        less_than = (coupling_norm <= coupling_aux)
        less_than_neg = (coupling_norm <= (-coupling_aux))

        # shape (n_coupled_cells, )
        scale_mult_numerator = (coupling_norm + coupling_aux)
        scale_mult = (scale_mult_numerator / (2 * coupling_norm))
        scale_mult[less_than] = 1.0
        scale_mult[less_than_neg] = 0.0

        # shape (n_coupled_cells, n_coupling_filters) * (n_coupled_cells, 1)
        # -> (n_coupled_cells, n_coupling filters)
        coupling_w_prox_applied = coupling_w * scale_mult[:, None]

        auxvar_prox = scale_mult_numerator / 2.0
        auxvar_prox[less_than] = coupling_aux[less_than]
        auxvar_prox[less_than_neg] = 0.0

    return coupling_w_prox_applied, feedback_w, timecourse_w, bias, auxvar_prox


def _spatial_model_prox_project_variables(coupling_filter_w: torch.Tensor,
                                          feedback_filter_w: torch.Tensor,
                                          spat_filter_w: torch.Tensor,
                                          bias_w: torch.Tensor,
                                          coupling_auxvar: torch.Tensor,
                                          spatial_sparsity_l1_lambda) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    The only not-smooth part of the penalty that we need to project is the
        neighboring cell coupling filter coefficients, and the neighboring cell
        coupling norm auxiliary variables.

    Everything else we can just pass through

    :param args:
    :param kwargs:
    :return:
    '''

    with torch.no_grad():
        # shape (n_coupled_cells, )
        coupling_norm = torch.linalg.norm(coupling_filter_w, dim=1)

        # shape (n_coupled_cells, ), binary-valued
        less_than = (coupling_norm <= coupling_auxvar)
        less_than_neg = (coupling_norm <= (-coupling_auxvar))

        # shape (n_coupled_cells, )
        scale_mult_numerator = (coupling_norm + coupling_auxvar)
        scale_mult = (scale_mult_numerator / (2.0 * coupling_norm))
        scale_mult[less_than] = 1.0
        scale_mult[less_than_neg] = 0.0

        # shape (n_coupled_cells, n_coupling_filters) * (n_coupled_cells, 1)
        # -> (n_coupled_cells, n_coupling filters)
        coupling_w_prox_applied = coupling_filter_w * scale_mult[:, None]

        auxvar_prox = scale_mult_numerator / 2.0
        auxvar_prox[less_than] = coupling_auxvar[less_than]
        auxvar_prox[less_than_neg] = 0.0

        if spatial_sparsity_l1_lambda != 0.0:
            spat_filter_w = torch.clamp_min_(spat_filter_w - spatial_sparsity_l1_lambda, 0.0) \
                            - torch.clamp_min_(-spat_filter_w - spatial_sparsity_l1_lambda, 0.0)

    return coupling_w_prox_applied, feedback_filter_w, spat_filter_w, bias_w, auxvar_prox


def _feedback_only_spatial_model_prox_project_variables(feedback_filter_w: torch.Tensor,
                                                        spat_filter_w: torch.Tensor,
                                                        bias_w: torch.Tensor,
                                                        spatial_sparsity_l1_lambda: float) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    with torch.no_grad():
        if spatial_sparsity_l1_lambda != 0.0:
            spat_filter_w = torch.clamp_min_(spat_filter_w - spatial_sparsity_l1_lambda, 0.0) \
                            - torch.clamp_min_(-spat_filter_w - spatial_sparsity_l1_lambda, 0.0)

    return feedback_filter_w, spat_filter_w, bias_w


def _lnp_prox_project_variables(spat_filter_w: torch.Tensor,
                                bias_w: torch.Tensor,
                                spatial_sparsity_l1_lambda: float) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if spatial_sparsity_l1_lambda != 0.0:
            spat_filter_w = torch.clamp_min_(spat_filter_w - spatial_sparsity_l1_lambda, 0.0) \
                            - torch.clamp_min_(-spat_filter_w - spatial_sparsity_l1_lambda, 0.0)

    return spat_filter_w, bias_w

