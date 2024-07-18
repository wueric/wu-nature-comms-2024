import torch
import torch.nn.functional as F


def precompute_feedback_exp_args(stacked_feedback_filters: torch.Tensor,
                                 all_observed_spikes: torch.Tensor) -> torch.Tensor:
    '''

    :param stacked_feedback_filters: shaep (n_cells, n_bins_filter)
    :param all_observed_spikes: shape (batch, n_cells, n_bins_observed)
    :return: shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
    '''
    n_cells = all_observed_spikes.shape[1]
    conv_padded = F.conv1d(all_observed_spikes,
                           stacked_feedback_filters[:, None, :],
                           groups=n_cells)
    return conv_padded


def precompute_coupling_exp_args(
        stacked_coupling_filters: torch.Tensor,
        coupled_sel: torch.Tensor,
        all_observed_spikes: torch.Tensor) -> torch.Tensor:
    '''

    :param stacked_coupling_filters: shape (n_cells, max_coupled_cells, n_bins_filter)
    :param coupled_sel: shape (n_cells, max_coupled_cells), integer index valued
        LongTensor
    :param all_observed_spikes: shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
    :return: shape (batch, n_cells, n_bins_observed - n_bins_filter + 1)
    '''
    batch, n_cells, n_bins_observed = all_observed_spikes.shape

    # we want an output set of spike trains with shape
    # (batch, n_cells, max_coupled_cells, n_bins_observed)

    # we need to pick our data out of all_observed_spikes, which has shape
    # (batch, n_cells, n_bins_observed)
    # using indices contained in self.coupled_sel, which has shape
    # (n_cells, max_coupled_cells), which contains indices from 0 to (n_cells - 1)

    # in order to use gather, the number of dimensions of each need to match
    # (we need 4 total dimensions)

    # shape (batch, n_cells, max_coupled_cells, n_bins_observed), index dimension is dim1 max_coupled_cells
    indices_repeated = coupled_sel[:, :, None].expand(batch, -1, -1, n_bins_observed)

    # shape (batch, n_cells, n_cells, n_bins_observed)
    observed_spikes_repeated = all_observed_spikes[:, None, :, :].expand(-1, n_cells, -1, -1)

    # shape (batch, n_cells, max_coupled_cells, n_bins_observed)
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
                             groups=n_cells)

    return coupling_conv
