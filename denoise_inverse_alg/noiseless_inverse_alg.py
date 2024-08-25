import torch
import torch.nn as nn

import numpy as np

from typing import Optional, Tuple, Union

from convex_optim_base.optim_base import BatchParallelDirectSolveProblem
from denoise_inverse_alg.hqs_alg import BatchParallel_HQS_X_Problem


class BatchNoiselessLinear_HQS_XProb(BatchParallelDirectSolveProblem,
                                     BatchParallel_HQS_X_Problem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_PROJECTIONS_KWARGS = 'projections'

    def __init__(self,
                 batch: int,
                 linear_filter_tensors: Union[torch.Tensor, np.ndarray],
                 dtype: torch.dtype = torch.float32):
        '''
        linear_filter_tensors: torch.Tensor, shape (n_cells, height, width)
        '''

        super().__init__()
        self.batch_size = batch

        self.n_cells, self.height, self.width = linear_filter_tensors.shape

        #### Linear filters ##############################
        stacked_flat_spat_filters = linear_filter_tensors.reshape(self.n_cells, -1)
        self.register_buffer('stacked_linear_filters',
                             torch.tensor(stacked_flat_spat_filters, dtype=dtype))

        #### CONSTANTS ############################
        self.register_buffer('z_const_tensor', torch.empty((self.batch_size, self.height, self.width), dtype=dtype))

        #### OPTIM VARIABLES ######################
        self.image = nn.Parameter(torch.empty((self.batch_size, self.height, self.width),
                                              dtype=dtype),
                                  requires_grad=True)
        nn.init.normal_(self.image, mean=0.0, std=0.1)

    def assign_z(self, z: torch.Tensor) -> None:
        self.z_const_tensor.data[:] = z.data

    def set_rho(self, new_rho: float) -> None:
        # not needed for the noiseless case
        pass

    def reinitialize_variables(self,
                               initialized_z_const: Optional[torch.Tensor] = None) -> None:
        # nn.init.normal_(self.z_const_tensor, mean=0.0, std=1.0)
        if initialized_z_const is None:
            self.z_const_tensor.data[:] = 0.0
        else:
            self.z_const_tensor.data[:] = initialized_z_const.data[:]

        nn.init.normal_(self.image, mean=0.0, std=0.1)

    def get_reconstructed_image(self) -> np.ndarray:
        return self.image.detach().cpu().numpy()

    @property
    def n_problems(self) -> int:
        return self.batch_size

    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def direct_solve(self, **kwargs) -> Tuple[torch.Tensor, ...]:

        with torch.no_grad():
            # shape (n_cells, n_cells), should be full rank
            mt_m = self.stacked_linear_filters @ self.stacked_linear_filters.T

            flat_z = self.z_const_tensor.reshape(self.batch_size, -1)

            # shape (1, n_cells, n_pixels) @ (batch_size, n_pixels, 1)
            # -> (batch_size, n_cells, 1) -> (batch_size, n_cells)
            proj_image = (self.stacked_linear_filters[None, :, :] @ flat_z[:, :, None]).squeeze(-1)

            # shape (batch_size, n_cells)
            diff_error = proj_image - kwargs[BatchNoiselessLinear_HQS_XProb.OBSERVED_PROJECTIONS_KWARGS]

            # shape (batch_size, n_cells)
            solved = torch.linalg.solve(mt_m[None, :, :], diff_error[:, :, None]).squeeze(-1)

            optim_soln_flat = flat_z - ((self.stacked_linear_filters.T)[None, :, :] @ solved[:, :, None]).squeeze(-1)

            optim_soln = optim_soln_flat.reshape(self.batch_size, self.height, self.width)

            self.image.data[:] = optim_soln.data[:]

            return (optim_soln,)

    def eval_loss(self, *args, **kwargs) -> torch.Tensor:
        return 0.0  # we don't care about the loss, since we have closed form solution
