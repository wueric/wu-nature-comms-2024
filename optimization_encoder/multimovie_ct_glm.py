import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import List, Callable, Union

class MM_FBOnly_PreappliedSpatialSingleCellGLMLoss(nn.Module):

    def __init__(self,
                 timecourse_filter: Union[torch.Tensor, np.ndarray],
                 feedback_filter: Union[torch.Tensor, np.ndarray],
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

        # shape (n_bins_filter, )
        if isinstance(feedback_filter, np.ndarray):
            self.register_buffer('feedback_filt', torch.tensor(feedback_filter, dtype=dtype))
        else:
            self.register_buffer('feedback_filt', feedback_filter)

        # shape (1, )
        if isinstance(bias, np.ndarray):
            self.register_buffer('bias', torch.tensor(bias, dtype=dtype))
        else:
            self.register_buffer('bias', bias)

    def forward(self,
                multimovie_filt_preapplied_stimuli: List[torch.Tensor],
                multimovie_cell_spikes: List[torch.Tensor]) -> torch.Tensor:

        n_bins_filter = self.time_filt.shape[0]

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

            # shape (n_bins - n_bins_filter + 1)
            feedback_applied = F.conv1d(center_spikes[None, :, :],
                                        self.feedback_filt[None, None, :]).squeeze(1).squeeze(0)

            total_gensig = stimulus_gensig_contrib + feedback_applied + self.bias

            center_spikes_squeeze = center_spikes.squeeze(0)
            n_count = total_gensig.shape[0] - 1
            loss = self.loss_callable(total_gensig[:-1], center_spikes_squeeze[n_bins_filter:]) * n_count
            total_N += n_count
            loss_acc.append(loss)

        loss_total = torch.sum(torch.stack(loss_acc, dim=0)) / total_N
        return loss_total


class MM_PreappliedSpatialSingleCellEncodingLoss(nn.Module):

    def __init__(self,
                 timecourse_filter: Union[torch.Tensor, np.ndarray],
                 feedback_filter: Union[torch.Tensor, np.ndarray],
                 coupling_filters: Union[torch.Tensor, np.ndarray],
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

        # shape (n_bins_filter, )
        if isinstance(feedback_filter, np.ndarray):
            self.register_buffer('feedback_filt', torch.tensor(feedback_filter, dtype=dtype))
        else:
            self.register_buffer('feedback_filt', feedback_filter)

        # shape (n_coupled_cells, n_bins_filter)
        if isinstance(coupling_filters, np.ndarray):
            self.register_buffer('couple_filt', torch.tensor(coupling_filters, dtype=dtype))
        else:
            self.register_buffer('couple_filt', coupling_filters)

        # shape (1, )
        if isinstance(bias, np.ndarray):
            self.register_buffer('bias', torch.tensor(bias, dtype=dtype))
        else:
            self.register_buffer('bias', bias)

    def forward(self,
                multimovie_filt_preapplied_stimuli: List[torch.Tensor],
                multimovie_cell_spikes: List[torch.Tensor],
                multimovie_coupled_spikes: List[torch.Tensor]) -> torch.Tensor:

        n_bins_filter = self.time_filt.shape[0]

        loss_acc = []
        total_N = 0
        for spat_inner_prod, center_spikes, coupled_spikes in \
                zip(multimovie_filt_preapplied_stimuli, multimovie_cell_spikes, multimovie_coupled_spikes):
            # stimulus movie: shape (1, n_bins)
            # center_spikes: shape (1, n_bins)
            # coupled_spikes: shape (n_coupled_cells, n_bins)

            # shape (n_bins - n_bins_filter + 1)
            stimulus_gensig_contrib = F.conv1d(spat_inner_prod[None, :, :],
                                               self.time_filt[None, None, :]).squeeze(1).squeeze(0)

            # shape (n_bins - n_bins_filter + 1)
            feedback_applied = F.conv1d(center_spikes[None, :, :],
                                        self.feedback_filt[None, None, :]).squeeze(1).squeeze(0)

            # shape (n_bins - n_bins_filter + 1)
            coupling_applied = F.conv1d(coupled_spikes[None, :, :],
                                        self.couple_filt[None, :, :]).squeeze(1).squeeze(0)

            total_gensig = stimulus_gensig_contrib + feedback_applied + coupling_applied + self.bias

            center_spikes_squeeze = center_spikes.squeeze(0)
            n_count = total_gensig.shape[0] - 1
            loss = self.loss_callable(total_gensig[:-1], center_spikes_squeeze[n_bins_filter:]) * n_count
            total_N += n_count
            loss_acc.append(loss)

        loss_total = torch.sum(torch.stack(loss_acc, dim=0)) / total_N
        return loss_total


class MM_SingleCellEncodingLoss(nn.Module):

    def __init__(self,
                 spatial_filter: Union[torch.Tensor, np.ndarray],
                 timecourse_filter: Union[torch.Tensor, np.ndarray],
                 feedback_filter: Union[torch.Tensor, np.ndarray],
                 coupling_filters: Union[torch.Tensor, np.ndarray],
                 bias: Union[torch.Tensor, np.ndarray],
                 loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 dtype: torch.dtype = torch.float32):
        '''

        :param spatial_filter:
        :param timecourse_filter:
        :param feedback_filter:
        :param coupling_filters:
        :param bias:
        :param loss_callable:
        '''

        super().__init__()

        self.loss_callable = loss_callable

        # shape (n_pixels, )
        if isinstance(spatial_filter, np.ndarray):
            self.register_buffer('spat_filt', torch.tensor(spatial_filter, dtype=dtype))
        else:
            self.register_buffer('spat_filt', spatial_filter)

        # shape (n_bins_filter, )
        if isinstance(timecourse_filter, np.ndarray):
            self.register_buffer('time_filt', torch.tensor(timecourse_filter, dtype=dtype))
        else:
            self.register_buffer('time_filt', timecourse_filter)

        # shape (n_bins_filter, )
        if isinstance(feedback_filter, np.ndarray):
            self.register_buffer('feedback_filt', torch.tensor(feedback_filter, dtype=dtype))
        else:
            self.register_buffer('feedback_filt', feedback_filter)

        # shape (n_coupled_cells, n_bins_filter)
        if isinstance(coupling_filters, np.ndarray):
            self.register_buffer('couple_filt', torch.tensor(coupling_filters, dtype=dtype))
        else:
            self.register_buffer('couple_filt', coupling_filters)

        # shape (1, )
        if isinstance(bias, np.ndarray):
            self.register_buffer('bias', torch.tensor(bias, dtype=dtype))
        else:
            self.register_buffer('bias', bias)

    def forward(self,
                multimovie_stimuli: List[torch.Tensor],
                multimovie_cell_spikes: List[torch.Tensor],
                multimovie_coupled_spikes: List[torch.Tensor]) -> torch.Tensor:

        n_bins_filter = self.time_filt.shape[0]

        loss_acc = []
        total_N = 0
        for stimulus_movie, center_spikes, coupled_spikes in \
                zip(multimovie_stimuli, multimovie_cell_spikes, multimovie_coupled_spikes):
            # stimulus movie: shape (n_pixel, n_bins)
            # center_spikes: shape (1, n_bins)
            # coupled_spikes: shape (n_coupled_cells, n_bins)

            # shape (1, n_pixels) @ (n_pixels, n_bins)
            # -> (1, n_bins)
            spat_inner_prod = self.spat_filt[None, :] @ stimulus_movie

            # shape (n_bins - n_bins_filter + 1)
            stimulus_gensig_contrib = F.conv1d(spat_inner_prod[None, :, :],
                                               self.time_filt[None, None, :]).squeeze(1).squeeze(0)

            # shape (n_bins - n_bins_filter + 1)
            feedback_applied = F.conv1d(center_spikes[None, :, :],
                                        self.feedback_filt[None, None, :]).squeeze(1).squeeze(0)

            # shape (n_bins - n_bins_filter + 1)
            coupling_applied = F.conv1d(coupled_spikes[None, :, :],
                                        self.couple_filt[None, :, :]).squeeze(1).squeeze(0)

            total_gensig = stimulus_gensig_contrib + feedback_applied + coupling_applied + self.bias


            center_spikes_squeeze = center_spikes.squeeze(0)
            n_count = total_gensig.shape[0] - 1
            loss = self.loss_callable(total_gensig[:-1], center_spikes_squeeze[n_bins_filter:]) * n_count
            total_N += n_count
            loss_acc.append(loss)

        loss_total = torch.sum(torch.stack(loss_acc, dim=0)) / total_N
        return loss_total
