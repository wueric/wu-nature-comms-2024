import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union, Callable, List


class MM_PreappliedSpatialSingleCellLNPLoss(nn.Module):

    def __init__(self,
                 timecourse_filter: Union[torch.Tensor, np.ndarray],
                 bias: Union[torch.Tensor, np.ndarray],
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.loss_callable = loss_callable

        # shape (n_bins_filter, )
        if isinstance(timecourse_filter, np.ndarray):
            self.register_buffer('time_filt', torch.tensor(timecourse_filter, dtype=dtype))
        else:
            self.register_buffer('time_filt', timecourse_filter)

        # shape (1, )
        if isinstance(bias, np.ndarray):
            self.register_buffer('bias', torch.tensor(bias, dtype=dtype))
        else:
            self.register_buffer('bias', bias)

    def forward(self,
                multimovie_filt_preapplied_stimuli: List[torch.Tensor],
                multimovie_cell_spikes: List[torch.Tensor]) -> torch.Tensor:

        loss_acc = []
        total_N = 0
        for spat_inner_prod, center_spikes in \
                zip(multimovie_filt_preapplied_stimuli, multimovie_cell_spikes):
            # stimulus movie: shape (1, n_bins)
            # center_spikes: shape (1, n_bins)
            # coupled_spikes: shape (n_coupled_cells, n_bins)

            # shape (n_bins - n_bins_filter + 1)
            stimulus_gensig_contrib = F.conv1d(spat_inner_prod[None, :, :],
                                               self.time_filt[None, None, :]).squeeze(1).squeeze(0)

            total_gensig = stimulus_gensig_contrib + self.bias

            center_spikes_squeeze = center_spikes.squeeze(0)
            n_count = total_gensig.shape[0]
            loss = self.loss_callable(total_gensig, center_spikes_squeeze) * n_count
            total_N += n_count
            loss_acc.append(loss)

        loss_total = torch.sum(torch.stack(loss_acc, dim=0)) / total_N
        return loss_total

