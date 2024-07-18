import torch
import torch.nn.functional as F


def single_bin_flashed_recons_compute_coupling_exp_arg(stacked_coupling_filters: torch.Tensor,
                                                       coupled_sel: torch.Tensor,
                                                       all_observed_spikes: torch.Tensor) -> torch.Tensor:
    '''
    Special case for the simulation, where we can't do the whole convolution ahead of time
        beecause the spikes need to be simulated

    In this case computes the coupling contribution to the generator signal for one time step
        using matrix multiplication, rather than for all of the timesteps using grouped convolution

    :param stacked_coupling_filters: shape (n_cells, max_coupled_cells, n_bins_filter)
    :param coupled_sel: shape (n_cells, max_coupled_cells), integer LongTensor
    :param all_observed_spikes: shape (batch, n_cells, n_bins_filter)
    :return: shape (batch, n_cells, 1)
    '''

    batch, n_cells, n_bins_filter = all_observed_spikes.shape
    n_cells_grouped = stacked_coupling_filters.shape[0]

    # we want an output set of spike trains with shape
    # (batch, n_cells, max_coupled_cells, n_bins_observed)

    # we need to pick our data out of all_observed_spikes, which has shape
    # (batch, n_cells, n_bins_observed)
    # using indices contained in self.coupled_sel, which has shape
    # (n_cells, max_coupled_cells), which contains indices from 0 to (n_cells - 1)

    # in order to use gather, the number of dimensions of each need to match
    # (we need 4 total dimensions)

    # shape (batch, n_cells, max_coupled_cells, n_bins_filter), index dimension is dim1 max_coupled_cells
    indices_repeated = coupled_sel[:, :, None].expand(batch, -1, -1, n_bins_filter)

    # shape (batch, n_cells, n_cells, n_bins_filter)
    observed_spikes_repeated = all_observed_spikes[:, None, :, :].expand(-1, n_cells_grouped, -1, -1)

    # shape (batch, n_cells, max_coupled_cells, n_bins_filter)
    selected_spike_trains = torch.gather(observed_spikes_repeated, 2, indices_repeated)

    # now we have to do a 1D convolution with the coupling filters
    # the intended output has shape
    # (n_cells, 1)

    # the input is in selected_spike_trains and has shape
    # (n_cells, max_coupled_cells, n_bins_observed)

    # the coupling filters are in self.stacked_coupling_filters and have shape
    # (n_cells, n_coupled_cells, n_bins_filter)

    # this looks like it needs to be a grouped 1D convolution with some reshaping,
    # since we convolve along time, need to sum over the coupled cells, but have
    # an extra batch dimension

    # we do a 1D convolution, with n_cells different groups

    # shape (batch, n_cells, max_coupled_cells * n_bins_filter)
    selected_spike_trains_flat = selected_spike_trains.reshape(batch, n_cells_grouped, -1)

    # shape (n_cells, max_coupled_cells * n_bins_filter)
    flattened_coupling_filters = stacked_coupling_filters.reshape(n_cells_grouped, -1)

    # shape (batch, n_cells, 1, max_coupled_cells * n_bins_filter) @
    #       (1, n_cells, max_coupled_cells * n_bins_filter, 1)
    # -> (batch, n_cells, 1, 1)
    filters_applied = selected_spike_trains_flat[:, :, None, :] @ flattened_coupling_filters[None, :, :, None]

    return filters_applied.squeeze(3)


def mixreal_single_bin_flashed_recons_compute_coupling_exp_arg(stacked_coupling_filters: torch.Tensor,
                                                               coupled_sel: torch.Tensor,
                                                               all_simulated_spikes: torch.Tensor,
                                                               all_data_spikes: torch.Tensor,
                                                               is_kaput: torch.Tensor) -> torch.Tensor:
    '''
    Even specialer-case for the simulation, where we can't do the whole convolution ahead of time because
        the spikes need to be simulated, and where some of the GLM fits are known to be bad and therefore
        we also mix in real spikes for those cells

    :param stacked_coupling_filters: stacked_coupling_filters: shape (n_cells, max_coupled_cells, n_bins_filter)
    :param coupled_sel: shape (n_cells, max_coupled_cells), integer LongTensor
    :param all_simulated_spikes: shape (batch, n_cells, n_bins_filter), simulated spike train so far
    :param all_data_spikes: shape (batch, n_cells, n_bins_filter), real spike train from data
    :param is_kaput: shape (n_cells, ), boolean-valued; True if the cell model fit has been
        deemed to be bad, and we should take the real data spikes when computing the coupling
        rather than when
    :return:
    '''

    batch, n_cells, n_bins_filter = all_simulated_spikes.shape
    n_cells_grouped = stacked_coupling_filters.shape[0]

    # we want an output set of spike trains with shape
    # (batch, n_cells, max_coupled_cells, n_bins_observed)

    # we need to pick our data out of all_observed_spikes, which has shape
    # (batch, n_cells, n_bins_observed)
    # using indices contained in self.coupled_sel, which has shape
    # (n_cells, max_coupled_cells), which contains indices from 0 to (n_cells - 1)

    # in order to use gather, the number of dimensions of each need to match
    # (we need 4 total dimensions)

    # shape (batch, n_cells, max_coupled_cells, n_bins_filter), index dimension is dim1 max_coupled_cells
    indices_repeated = coupled_sel[:, :, None].expand(batch, -1, -1, n_bins_filter)

    # shape (batch, n_cells, n_cells, n_bins_filter)
    simulated_spikes_repeated = all_simulated_spikes[:, None, :, :].expand(-1, n_cells_grouped, -1, -1)
    data_spikes_repeated = all_data_spikes[:, None, :, :].expand(-1, n_cells_grouped, -1, -1)

    # shape (batch, n_cells, max_coupled_cells, n_bins_filter)
    selected_simulated_spike_trains = torch.gather(simulated_spikes_repeated, 2, indices_repeated)
    selected_data_spike_trains = torch.gather(data_spikes_repeated, 2, indices_repeated)

    # now mix the two, based on which cells are kaput and which cells are not
    selected_spike_trains = torch.zeros_like(selected_simulated_spike_trains)
    selected_spike_trains[:, is_kaput, :] += selected_data_spike_trains[:, is_kaput, :]
    selected_spike_trains[:, ~is_kaput, :] += selected_simulated_spike_trains[:, ~is_kaput, :]

    # now we have to do a 1D convolution with the coupling filters
    # the intended output has shape
    # (n_cells, 1)

    # the input is in selected_spike_trains and has shape
    # (n_cells, max_coupled_cells, n_bins_observed)

    # the coupling filters are in self.stacked_coupling_filters and have shape
    # (n_cells, n_coupled_cells, n_bins_filter)

    # this looks like it needs to be a grouped 1D convolution with some reshaping,
    # since we convolve along time, need to sum over the coupled cells, but have
    # an extra batch dimension

    # we do a 1D convolution, with n_cells different groups

    # shape (batch, n_cells, max_coupled_cells * n_bins_filter)
    selected_spike_trains_flat = selected_spike_trains.reshape(batch, n_cells_grouped, -1)

    # shape (n_cells, max_coupled_cells * n_bins_filter)
    flattened_coupling_filters = stacked_coupling_filters.reshape(n_cells_grouped, -1)

    # shape (batch, n_cells, 1, max_coupled_cells * n_bins_filter) @
    #       (1, n_cells, max_coupled_cells * n_bins_filter, 1)
    # -> (batch, n_cells, 1, 1)
    filters_applied = selected_spike_trains_flat[:, :, None, :] @ flattened_coupling_filters[None, :, :, None]

    return filters_applied.squeeze(3)


def mixreal_single_bin_flashed_recons_compute_feedback_exp_arg(stacked_feedback_filters,
                                                               all_simulated_spikes: torch.Tensor,
                                                               all_data_spikes: torch.Tensor,
                                                               is_kaput: torch.Tensor) -> torch.Tensor:
    '''
    Even-specialer case for the simulation, where we can't do the whole convolution ahead of time because the spikes
        need to be simualted, and where some the GLM fits are known to be bad and so
        we use real spikes for those cells

    :param stacked_feedback_filters: shape (n_cells, n_bins_filter)
    :param all_simulated_spikes: (batch, n_cells, n_bins_filter), simulated spike train
    :param all_data_spikes: (batch, n_cells, n_bins_filter), real spike train
    :param is_kaput: shape (n_cells, ), boolean-valued; True if the cell model fit has been
        deemed to be bad, and we should take the real data spikes when computing the coupling
        rather than when
    :return:
    '''

    # compute the feedback contribution to the generator signal
    # shape (batch, n_cells, 1, n_bins_filter) @ (1, n_cells, n_bins_filter, 1)
    # -> (batch, n_cells, 1, 1) -> (batch, n_cells)
    sim_feedback_val = all_simulated_spikes[:, :, None, :] @ stacked_feedback_filters[None, :, :, None]
    sim_feedback_val = sim_feedback_val.squeeze(3).squeeze(2)

    data_feedback_val = all_data_spikes[:, :, None, :] @ stacked_feedback_filters[None, :, :, None]
    data_feedback_val = data_feedback_val.squeeze(3).squeeze(2)

    feedback_val = torch.zeros_like(sim_feedback_val)
    feedback_val[:, is_kaput] += data_feedback_val[:, is_kaput]
    feedback_val[:, ~is_kaput] += sim_feedback_val[:, ~is_kaput]

    return feedback_val


def flashed_recons_compute_feedback_exp_arg(stacked_feedback_filters: torch.Tensor,
                                            all_observed_spikes: torch.Tensor) -> torch.Tensor:
    '''

    :param stacked_feedback_filters: shape (n_cells, n_bins_filter)
    :param all_observed_spikes: (batch, n_cells, n_bins_observed),
            one entry for every batch, every cell
    :return:
    '''

    n_cells = stacked_feedback_filters.shape[0]

    conv_padded = F.conv1d(all_observed_spikes,
                           stacked_feedback_filters[:, None, :],
                           groups=n_cells)
    return conv_padded


def flashed_recons_compute_coupling_exp_arg(stacked_coupling_filters: torch.Tensor,
                                            coupled_sel: torch.Tensor,
                                            all_observed_spikes: torch.Tensor) -> torch.Tensor:
    '''

    :param stacked_coupling_filters: shape (n_cells_grouped, max_coupled_cells, n_bins_filter)
    :param coupled_sel: shape (n_cells_grouped, max_coupled_cells), integer LongTensor
    :param all_observed_spikes: shape (batch, n_cells, n_bins_observed)
    :return: shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
    '''

    batch, n_cells, n_bins_observed = all_observed_spikes.shape
    n_cells_grouped = stacked_coupling_filters.shape[0]

    # we want an output set of spike trains with shape
    # (batch, n_cells, max_coupled_cells, n_bins_observed)

    # we need to pick our data out of all_observed_spikes, which has shape
    # (batch, n_cells, n_bins_observed)
    # using indices contained in self.coupled_sel, which has shape
    # (n_cells_grouped, max_coupled_cells), which contains indices from 0 to (n_cells - 1)

    # in order to use gather, the number of dimensions of each need to match
    # (we need 4 total dimensions)

    # shape (batch, n_cells_grouped, max_coupled_cells, n_bins_observed), index dimension is dim1 max_coupled_cells
    indices_repeated = coupled_sel[:, :, None].expand(batch, -1, -1, n_bins_observed)

    # shape (batch, n_cells_grouped, n_cells, n_bins_observed)
    observed_spikes_repeated = all_observed_spikes[:, None, :, :].expand(-1, n_cells_grouped, -1, -1)

    # shape (batch, n_cells_grouped, max_coupled_cells, n_bins_observed)
    selected_spike_trains = torch.gather(observed_spikes_repeated, 2, indices_repeated)

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

    # shape (batch, n_cells * max_coupled_cells, n_bins_observed)
    selected_spike_trains_reshape = selected_spike_trains.reshape(batch, -1, n_bins_observed)

    # (batch, n_cells * max_coupled_cells, n_bins_observed) \ast (n_cells, n_coupled_cells, n_bins_filter)
    # -> (batch, n_cells, n_bins_observed - n_bins_filter + 1)
    coupling_conv = F.conv1d(selected_spike_trains_reshape,
                             stacked_coupling_filters,
                             groups=n_cells_grouped)

    return coupling_conv


def flashed_recons_compute_timecourse_component(stacked_timecourse_filters,
                                                stim_time: torch.Tensor) -> torch.Tensor:
    '''

    :param stacked_timecourse_filters: shape (n_cells, n_bins_filter)
    :param stim_time: shape (n_bins, )
    :return:
    '''

    conv_extra_dims = F.conv1d(stim_time[None, None, :],
                               stacked_timecourse_filters[:, None, :])
    return conv_extra_dims.squeeze(0)
