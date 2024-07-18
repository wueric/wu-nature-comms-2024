from typing import List

import torch


def _movie_spatial_stimulus_contrib_gensig(spat_filt_w: torch.Tensor,
                                           time_filt_applied: torch.Tensor) -> torch.Tensor:
    '''

    :param spat_filt_w: shape (1, n_basis_stim_spat)
    :param multimovie_filtered_list: shape (n_basis_stim_spat, n_bins - n_bins_filter + 1)
    :return:
    '''

    # shape (1, n_basis_stim_spat) @ (n_basis_stim_spat, n_bins - n_bins_filter + 1)
    # -> (1, n_bins - n_bins_filter + 1) -> (n_bins - n_bins_filter + 1)
    spat_filt_applied = (spat_filt_w @ time_filt_applied).squeeze(0)
    return spat_filt_applied


def _multimovie_spatial_stimulus_contrib_gensig(
        spat_filt_w,
        multimovie_filtered_list: List[torch.Tensor]) \
        -> List[torch.Tensor]:
    '''

    :param spat_filt_w: shape (1, n_basis_stim_spat)
    :param multimovie_filtered_list: List of (n_basis_stim_spat, n_bins - n_bins_filter + 1)
        n_bins could be different for each item in the list
    :return:
    '''

    return [_movie_timecourse_stimulus_contrib_gensig(spat_filt_w, x) for x in multimovie_filtered_list]


def _movie_timecourse_stimulus_contrib_gensig(time_filt_w: torch.Tensor,
                                              movie_flat_binrate: torch.Tensor) -> torch.Tensor:
    '''

    :param time_filt_w: shape (1, n_basis_stim_time)
    :param movie_flat_binrate: shape (n_basis_stim_time, n_bins - n_bins_filter + 1),
    :return:
    '''
    # shape (1, n_basis_stim_time) @ (n_basis_stim_time, n_bins - n_bins_filter + 1)
    # -> (1, n_bins - n_bins_filter + 1)
    # -> (n_bins - n_bins_filter + 1, )
    time_filt_applied = (time_filt_w @ movie_flat_binrate).squeeze(0)
    return time_filt_applied


def _multimovie_timecourse_stimulus_contrib_gensig(
        time_filt_w: torch.Tensor,
        multimovie_filtered_list: List[torch.Tensor]) -> List[torch.Tensor]:
    '''

    :param time_filt_w: shape (1, n_basis_stim_time)
    :param multimovie_filtered_list: List of (n_basis_stim_time, n_bins - n_bins_filter + 1),
            where n_bins could be different for each item in the list
    :return:
    '''
    return [_movie_timecourse_stimulus_contrib_gensig(time_filt_w, x) for x in multimovie_filtered_list]


def _movie_feedback_contrib_gensig(feedback_filt_w: torch.Tensor,
                                   feedback_convolved: torch.Tensor) -> torch.Tensor:
    '''

    :param feedback_filt_w: shape (1, n_basis_feedback)
    :param feedback_convolved: shape (1, n_basis_feedback, n_bins - n_bins_filter + 1)
    :return:
    '''

    # shape (1, 1, n_basis_feedback) @ (1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
    # -> (1, 1, n_bins_total - n_bins_filter + 1)
    # -> (n_bins_total - n_bins_filter + 1, )
    feedback_filt_applied = (feedback_filt_w[:, None, :] @ feedback_convolved).squeeze(1).squeeze(0)
    return feedback_filt_applied


def _multimovie_feedback_contrib_gensig(feedback_filt_w: torch.Tensor,
                                        multi_feedback_convolved: List[torch.Tensor]) -> List[torch.Tensor]:
    return [_movie_feedback_contrib_gensig(feedback_filt_w, x) for x in multi_feedback_convolved]


def _movie_coupling_feedback_contrib_gensig(coupling_filt_w: torch.Tensor,
                                            feedback_filt_w: torch.Tensor,
                                            coupling_convolved: torch.Tensor,
                                            feedback_convolved: torch.Tensor) \
        -> torch.Tensor:
    '''

    :param coupling_filt_w: shape (n_coupled_cells, n_basis_coupling)
    :param feedback_filt_w: shape (1, n_basis_feedback)
    :param coupling_convolved: shape (n_coupled_cells, n_basis_coupling, n_bins - n_bins_filter + 1)
    :param feedback_convolved: shape (1, n_basis_feedback, n_bins - n_bins_filter + 1)
    :return:
    '''
    # shape (n_coupled_cells, 1, n_basis_coupling) @ (n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
    # -> (n_coupled_cells, 1, n_bins_total - n_bins_filter + 1)
    # -> (n_coupled_cells, n_bins_total - n_bins_filter + 1)
    coupling_filt_applied = (coupling_filt_w[:, None, :] @ coupling_convolved).squeeze(1)

    # shape (1, 1, n_basis_feedback) @ (1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
    # -> (1, 1, n_bins_total - n_bins_filter + 1)
    # -> (n_bins_total - n_bins_filter + 1, )
    feedback_filt_applied = (feedback_filt_w[:, None, :] @ feedback_convolved).squeeze(1).squeeze(0)

    # shape (n_bins_total - n_bins_filter + 1, )
    output_val = torch.sum(coupling_filt_applied, dim=0) + feedback_filt_applied

    return output_val


def _multimovie_coupling_feedback_contrib_gensig(
        coupling_filt_w: torch.Tensor,
        feedback_filt_w: torch.Tensor,
        multi_coupling_convolved: List[torch.Tensor],
        multi_feedback_convolved: List[torch.Tensor]) \
        -> List[torch.Tensor]:
    '''

    :param coupling_filt_w: shape (n_coupled_cells, n_basis_coupling)
    :param feedback_filt_w: shape (1, n_basis_feedback)
    :param multi_coupling_convolved: List of (n_coupled_cells, n_basis_coupling, n_bins - n_bins_filter + 1)
         n_bins could be different for each entry in the list
    :param multi_feedback_convolved: List of (1, n_basis_feedback, n_bins - n_bins_filter + 1)
         n_bins could be different for each entry in the list
    :return:
    '''
    return [_movie_coupling_feedback_contrib_gensig(coupling_filt_w, feedback_filt_w, x, y)
            for x, y, in zip(multi_coupling_convolved, multi_feedback_convolved)]


def _flashed_spatial_stimulus_contrib_gensig(spat_filt_w: torch.Tensor,
                                             spat_component_stimulus: torch.Tensor,
                                             time_filter_applied: torch.Tensor) -> torch.Tensor:
    '''

    :param spat_filt_w: shape (1, n_basis_stim_spat),
    :param spat_component_stimulus: shape (batch, n_basis_stim_spat), flashed stimulus
        image for each trial
    :param time_filter_applied: shape (n_bins - n_bins_filter + 1, ), time component
        of the stimulus presentation
    :return:
    '''

    # shape (batch, n_basis_stim_spat) @ (n_basis_stim_spat, 1)
    # -> (batch, 1)
    spatial_filter_applied = (spat_component_stimulus @ spat_filt_w.T)

    # shape (batch, 1) @ (1, n_bins - n_bins_filter + 1)
    # -> (batch, n_bins - n_bins_filter + 1)
    spacetime_filter_applied = spatial_filter_applied @ time_filter_applied[None, :]

    # -> (batch, n_bins - n_bins_filter + 1)
    return spacetime_filter_applied


def _flashed_timecourse_stimulus_contrib_gensig(time_filter_w: torch.Tensor,
                                                ns_filt_spat: torch.Tensor,
                                                ns_filt_time_movie: torch.Tensor) -> torch.Tensor:
    '''

    :param time_filter_w: shape (1, n_basis_stim_time)
    :param ns_filt_spat: shape (batch, )
    :param ns_filt_time_movie: shape (n_basis_stim_time, n_bins - n_bins_filter + 1)
    :return:
    '''

    # shape (1, n_basis_stim_time) @ (n_basis_stim_time, n_bins - n_bins_filter + 1)
    # -> (1, n_bins - n_bins_filter + 1)  -> (n_bins - n_bins_filter + 1, )
    time_filter_applied = (time_filter_w @ ns_filt_time_movie).squeeze(0)

    # shape (batch, 1) @ (1, n_bins - n_bins_filter + 1)
    # -> (batch, n_bins - n_bins_filter + 1)
    return ns_filt_spat[:, None] @ time_filter_applied[None, :]


def _flashed_feedback_contrib_gensig(feedback_filt_w: torch.Tensor,
                                     feedback_convolved: torch.Tensor) -> torch.Tensor:
    '''

    :param feedback_filt_w: shape (1, n_basis_feedback)
    :param feedback_convolved: shape (batch, n_basis_feedback, n_bins_total - n_bins_filter + 1)
    :return:
    '''

    # shape (1, 1, 1, n_basis_feedback) @
    #   (batch, 1, n_basis_feedback, n_bins_total - n_bins_filter + 1)
    # -> (batch, 1, 1, n_bins_total - n_bins_filter + 1)
    # -> (batch, n_bins_total - n_bins_filter + 1)
    feedback_filt_applied = (feedback_filt_w[None, :, None, :] @ feedback_convolved[:, None, :, :]).squeeze(2).squeeze(1)
    return feedback_filt_applied


def _flashed_coupling_contrib_gensig(coupling_filt_w: torch.Tensor,
                                     coupling_convolved: torch.Tensor) -> torch.Tensor:
    '''

    :param coupling_filt_w: shape (n_coupled_cells, n_basis_coupling)
    :param coupling_convolved: shape (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
    :return:
    '''

    # shape (1, n_coupled_cells, 1, n_basis_coupling) @
    #   (batch, n_coupled_cells, n_basis_coupling, n_bins_total - n_bins_filter + 1)
    # -> (batch, n_coupled_cells, 1, n_bins_total - n_bins_filter + 1)
    # -> (batch, n_coupled_cells, n_bins_total - n_bins_filter + 1)
    coupling_filt_applied = (coupling_filt_w[None, :, None, :] @ coupling_convolved).squeeze(2)
    return torch.sum(coupling_filt_applied, dim=1)
