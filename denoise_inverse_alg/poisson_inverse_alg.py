import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from convex_optim_base.optim_base import BatchParallelUnconstrainedProblem
from denoise_inverse_alg.glm_inverse_alg import FlashedModelRequiresPrecomputation
from denoise_inverse_alg.hqs_alg import BatchParallel_HQS_X_Problem

from typing import Optional, Dict, List, Callable
from dataclasses import dataclass

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import CroppedSTABoundingBox
from optimization_encoder.trial_glm import FittedLNPFamily


@dataclass
class PackedLNPTensors:
    spatial_filters: np.ndarray  # shape (n_cells, height, width)
    timecourse_filters: np.ndarray  # shape (n_cells, n_bins_filter)
    bias: np.ndarray  # shape (n_cells, )


@dataclass
class FullResolutionCompactLNP:
    '''
    Used as an intermediate representation of the paramters of a single
        cell GLM for spot-checking the model

    Note that the model here is fitted on the full resolution stimulus
        with no spatial basis
    '''

    spatial_filter: np.ndarray  # shape (height, width), same shape as the stimulus
    timecourse_filter: np.ndarray  # shape (n_bins, )
    bias: np.ndarray


def _extract_cropped_lnp_params(lnp_family: FittedLNPFamily,
                                cell_id: int,
                                bounding_box: CroppedSTABoundingBox,
                                full_height: int,
                                full_width: int,
                                downsample_factor: int = 1,
                                crop_width_low: int = 0,
                                crop_width_high: int = 0,
                                crop_height_low: int = 0,
                                crop_height_high: int = 0) -> FullResolutionCompactLNP:
    fitted_glm = lnp_family.fitted_models[cell_id]

    # shape (n_timecourse_basis, n_bins_filter)
    timecourse_basis = lnp_family.timecourse_basis

    # shape (1, n_basis_stim_time)
    timecourse_weights = fitted_glm.timecourse_weights

    # shape (1, n_basis_stim_time) @ (n_timecourse_basis, n_bins_filter)
    # -> (1, n_bins_filter) -> (n_bins_filter, )
    timecourse_filter = (timecourse_weights @ timecourse_basis).squeeze(0)

    full_spatial_filter = np.zeros((full_height, full_width), dtype=np.float32)

    putback_slice_obj_h, putback_slice_obj_w = bounding_box.make_precropped_sliceobj(
        crop_hlow=crop_height_low,
        crop_hhigh=crop_height_high,
        crop_wlow=crop_width_low,
        crop_whigh=crop_width_high,
        downsample_factor=downsample_factor)

    full_spatial_filter[putback_slice_obj_h, putback_slice_obj_w] = fitted_glm.spatial_weights

    bias = fitted_glm.spatial_bias

    return FullResolutionCompactLNP(
        full_spatial_filter,
        timecourse_filter,
        bias
    )


def reinflate_cropped_lnp_model(
        deflated_models: Dict[str, FittedLNPFamily],
        bounding_box_dict: Dict[str, List[CroppedSTABoundingBox]],
        cell_ordering: OrderedMatchedCellsStruct,
        raw_height: int,
        raw_width: int,
        downsample_factor: int = 1,
        crop_width_low: int = 0,
        crop_height_low: int = 0,
        crop_width_high: int = 0,
        crop_height_high: int = 0) -> PackedLNPTensors:
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
    cell_type_order = cell_ordering.get_cell_types()

    bias_list = []  # type: List[np.ndarray]
    timecourse_filters_list = []  # type: List[np.ndarray]
    spatial_filters_list = []  # type: List[np.ndarray]

    for cell_type in cell_type_order:

        glm_family = deflated_models[cell_type]

        for idx, cell_id in enumerate(cell_ordering.get_reference_cell_order(cell_type)):
            compact_model = _extract_cropped_lnp_params(glm_family,
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
            spatial_filters_list.append(compact_model.spatial_filter)
            bias_list.append(compact_model.bias)

    # now stack all of the arrays together
    spatial_filters_stacked = np.stack(spatial_filters_list, axis=0)
    timecourse_filters_stacked = np.stack(timecourse_filters_list, axis=0)
    bias_stacked = np.stack(bias_list, axis=0)

    packed_lnp_tensors = PackedLNPTensors(spatial_filters_stacked,
                                          timecourse_filters_stacked,
                                          bias_stacked)

    return packed_lnp_tensors


@torch.jit.script
def poisson_loss_noreduce(generator_sig: torch.Tensor,
                                   observed_spikes: torch.Tensor) -> torch.Tensor:
    exp_gen_sig = torch.exp(generator_sig)
    spike_prod = generator_sig * observed_spikes
    loss_per_bin_per_cell = exp_gen_sig - spike_prod
    return loss_per_bin_per_cell


class BatchFlashedFrameRatePoissonEncodingLoss_Precompute(nn.Module):

    def __init__(self,
                 lnp_filters: np.ndarray,
                 lnp_timecourse: np.ndarray,
                 lnp_biases: np.ndarray,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn_noreduce: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 n_problems: int = 1,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.n_problems = n_problems
        self.n_cells, self.height, self.width = lnp_filters.shape
        self.n_pixels = self.height * self.width
        self.n_bins_filter = lnp_timecourse.shape[1]
        self.n_bins_loss = stimulus_time_component.shape[0] - self.n_bins_filter + 1

        lnp_filters_flat = lnp_filters.reshape(self.n_cells, -1)

        # shape (n_cells, height, width)
        self.register_buffer('lnp_filters',
                             torch.tensor(lnp_filters_flat, dtype=dtype))

        # shape (n_cells, n_bins_filter)
        self.register_buffer('lnp_timecourse',
                             torch.tensor(lnp_timecourse, dtype=dtype))

        # shape (n_cells, )
        self.register_buffer('lnp_biases',
                             torch.tensor(lnp_biases, dtype=dtype))

        # shape (n_bins, )
        self.register_buffer('stimulus_time_component',
                             torch.tensor(stimulus_time_component, dtype=dtype))

        # shape (n_cells, n_bins_loss = n_bins_total - n_bins_filter + 1)
        self.register_buffer('precomputed_time_component',
                             torch.zeros((self.n_cells, self.n_bins_loss), dtype=dtype))

        self.spiking_loss_fn_noreduce = spiking_loss_fn_noreduce

    def precompute_time_convolutions(self):
        with torch.no_grad():
            conv_extra_dims = F.conv1d(self.stimulus_time_component[None, None, :],
                                       self.lnp_timecourse[:, None, :]).squeeze(0)
            self.precomputed_time_component.data[:] = conv_extra_dims.data[:]

    def forward(self,
                batched_images: torch.Tensor,
                batched_observed_spikes: torch.Tensor) -> torch.Tensor:
        '''

        :param batched_images_flat: (batch, height, width)
        :param batched_observed_spikes: (batch, n_cells, n_bins_loss)
        :return:
        '''

        # shape (batch, n_pixels)
        batched_images_flat = batched_images.reshape(batched_images.shape[0], -1)

        # shape (n_cells, n_pixels)
        lnp_filters_flat = self.lnp_filters.reshape(self.n_cells, -1)

        # shape (batch, n_pixels) @ (n_pixels, n_cells) -> (batch, n_cells)
        spat_filt_applied = batched_images_flat @ lnp_filters_flat.T

        # shape (batch, n_cells, 1) * (1, n_cells, n_bins_loss)
        # -> (batch, n_cells, n_bins_loss)
        spat_filt_with_time = spat_filt_applied[:, :, None] * self.precomputed_time_component[None, :, :]

        # shape (batch, n_cells, n_bins_loss)
        gensig = spat_filt_with_time + self.lnp_biases[None, :, None]

        # shape (batch, n_cells, n_bins_loss)
        loss_per_bin = self.spiking_loss_fn_noreduce(gensig[:, :, :-1],
                                                     batched_observed_spikes[:, :, self.n_bins_filter:])

        # shape (batch, )
        return torch.sum(loss_per_bin, dim=(1, 2))


class BatchFlashedFrameRatePoissonProxProblem(BatchParallelUnconstrainedProblem,
                                              BatchParallel_HQS_X_Problem,
                                              FlashedModelRequiresPrecomputation):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 batch: int,
                 lnp_filters: np.ndarray,
                 lnp_timecourse: np.ndarray,
                 lnp_biases: np.ndarray,
                 stimulus_time_component: np.ndarray,
                 spiking_loss_fn_noreduce: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 rho: float = 1.0,  # placeholder value, allows object to be created even when we don't yet know rho
                 dtype: torch.dtype = torch.float32):
        '''

        :param batch:
        :param lnp_filters:
        :param lnp_timecourse:
        :param lnp_biases:
        :param stimulus_time_component:
        :param spiking_loss_fn_noreduce:
        :param rho:
        :param dtype:
        '''

        super().__init__()

        self.batch = batch
        self.rho = rho

        self.n_cells, self.height, self.width = lnp_filters.shape

        self.lnp_loss_calc = BatchFlashedFrameRatePoissonEncodingLoss_Precompute(
            lnp_filters,
            lnp_timecourse,
            lnp_biases,
            stimulus_time_component,
            spiking_loss_fn_noreduce,
            n_problems=batch,
            dtype=dtype
        )

        #### CONSTANTS ###############################################
        self.register_buffer('z_const_tensor', torch.empty((self.batch, self.height, self.width), dtype=dtype))

        #### OPTIM VARIABLES #########################################
        # OPT VARIABLE 0: image, shape (batch, height, width)
        self.image = nn.Parameter(torch.empty((self.batch, self.height, self.width),
                                              dtype=dtype))
        nn.init.uniform_(self.image, a=-1e-2, b=1e-2)

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

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

        # shape ()
        encoding_loss = self.lnp_loss_calc(batched_image_imshape, batched_spikes)

        # shape ()
        prox_diff = batched_image_imshape - self.z_const_tensor
        prox_loss = 0.5 * self.rho * torch.sum(prox_diff * prox_diff, dim=(1, 2))

        return encoding_loss + prox_loss

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

    def assign_proxto(self, prox_to: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = prox_to.data[:]

    @property
    def n_problems(self) -> int:
        return self.batch

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def get_output_image(self) -> torch.Tensor:
        return self.reconstructed_images.detach().clone()

    def precompute_gensig_components(self,
                                     all_spikes: torch.Tensor) -> None:
        return self.lnp_loss_calc.precompute_time_convolutions()
