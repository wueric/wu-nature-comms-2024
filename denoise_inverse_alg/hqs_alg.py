import torch
import torch.optim as optim

import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, Tuple, Iterator, Optional, List, Any
from dataclasses import dataclass

from torch import nn

from convex_optim_base.direct_optim import single_direct_solve, batch_parallel_direct_solve
from convex_optim_base.optim_base import SingleProblem, BatchParallelProblem, SingleUnconstrainedProblem, \
    BatchParallelUnconstrainedProblem, BatchParallelDirectSolveProblem, SingleDirectSolveProblem
from convex_optim_base.unconstrained_optim import single_unconstrained_solve, \
    FistaSolverParams, batch_parallel_unconstrained_solve
from gaussian_denoiser import denoiser_wrappers


@dataclass
class ScheduleVal:
    rho_start: float
    rho_end: float

    lambda_prior: float

    n_iter: int


def make_logspaced_rho_schedule(sched: ScheduleVal) -> np.ndarray:
    return np.logspace(np.log10(sched.rho_start), np.log10(sched.rho_end), sched.n_iter)


def make_schedules(sched_list: List[ScheduleVal]) -> Tuple[np.ndarray, np.ndarray]:
    to_cat_rho = []
    to_cat_lambda = []
    for sched in sched_list:
        to_cat_rho.append(np.logspace(np.log10(sched.rho_start), np.log10(sched.rho_end), sched.n_iter))
        to_cat_lambda.append(np.logspace(np.log10(sched.lambda_prior), np.log10(sched.lambda_prior), sched.n_iter))

    return np.concatenate(to_cat_rho), np.concatenate(to_cat_lambda)


HQS_ParameterizedSolveFn = Callable[[Union[SingleProblem, BatchParallelProblem], Optional[bool], Any],
                                    Union[float, torch.Tensor]]


class HQS_X_Problem(metaclass=ABCMeta):

    @abstractmethod
    def assign_z(self, z: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def set_rho(self, new_rho: float) -> None:
        raise NotImplementedError


class BatchParallel_HQS_X_Problem(metaclass=ABCMeta):

    @abstractmethod
    def assign_z(self, z: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_A_x(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def set_rho(self, new_rho: float) -> None:
        raise NotImplementedError


class HQS_Z_Problem(metaclass=ABCMeta):

    @abstractmethod
    def assign_A_x(self, Ax: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_z(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def set_rho(self, new_rho: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_prior_lambda(self, lambda_val: float) -> None:
        raise NotImplementedError


class BatchParallel_HQS_Z_Problem(metaclass=ABCMeta):

    @abstractmethod
    def assign_A_x(self, Ax: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_z(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def set_rho(self, new_rho: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_prior_lambda(self, lambda_val: float) -> None:
        raise NotImplementedError


def scheduled_rho_hqs_solve(x_problem: Union[SingleProblem, HQS_X_Problem,
                                             BatchParallelProblem, BatchParallel_HQS_X_Problem],
                            x_solver_it: Iterator[HQS_ParameterizedSolveFn],
                            z_problem: Union[SingleProblem, HQS_Z_Problem,
                                             BatchParallelProblem, BatchParallel_HQS_Z_Problem],
                            z_solver_it: Iterator[HQS_ParameterizedSolveFn],
                            rho_schedule: Iterator[float],
                            prior_lambda: float,
                            max_iter_hqs: int,
                            verbose: bool = False,
                            save_intermediates: bool = False,
                            z_initialization_point: Optional[torch.Tensor] = None,
                            **kwargs) \
        -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    intermediates_list = None
    if save_intermediates:
        intermediates_list = []

    if z_initialization_point is not None:
        x_problem.assign_z(z_initialization_point)

    for it, x_solve_fn, z_solve_fn, rho_val in \
            zip(range(max_iter_hqs), x_solver_it, z_solver_it, rho_schedule):

        x_problem.set_rho(rho_val)
        z_problem.set_rho(rho_val)
        z_problem.set_prior_lambda(prior_lambda)

        loss_x_prob = x_solve_fn(x_problem, verbose=verbose, **kwargs)

        with torch.no_grad():
            next_ax = x_problem.compute_A_x(*x_problem.parameters(recurse=False), **kwargs)
            z_problem.assign_A_x(next_ax)

        loss_z_prob = z_solve_fn(z_problem, verbose=verbose, **kwargs)

        with torch.no_grad():
            next_z = z_problem.get_z(*z_problem.parameters(recurse=False), **kwargs)
            x_problem.assign_z(next_z)

        if save_intermediates:
            intermediates_list.append((x_problem.get_reconstructed_image(), next_z.detach().cpu().numpy()))

        if verbose:
            print(f"HQS iter {it}, xloss {loss_x_prob}, zloss {loss_z_prob}")

    return intermediates_list


def iter_rho_fixed_prior_hqs_solve(x_problem: Union[HQS_X_Problem, BatchParallel_HQS_X_Problem],
                                   x_solver_it: Iterator[HQS_ParameterizedSolveFn],
                                   z_problem: Union[HQS_Z_Problem, BatchParallel_HQS_Z_Problem],
                                   z_solver_it: Iterator[HQS_ParameterizedSolveFn],
                                   rho_schedule: Iterator[float],
                                   fixed_prior_weight: float,
                                   verbose: bool = False,
                                   save_intermediates: bool = False,
                                   z_initialization_point: Optional[torch.Tensor] = None,
                                   **kwargs) \
        -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    intermediates_list = None
    if save_intermediates:
        intermediates_list = []

    if z_initialization_point is not None:
        x_problem.assign_z(z_initialization_point)

    for it, (x_solve_fn, z_solve_fn, rho_val) in enumerate(zip(x_solver_it, z_solver_it, rho_schedule)):

        x_problem.set_rho(rho_val)
        z_problem.set_rho(rho_val)
        z_problem.set_prior_lambda(fixed_prior_weight)

        loss_x_prob = x_solve_fn(x_problem, verbose=verbose, **kwargs)

        with torch.no_grad():
            next_ax = x_problem.compute_A_x(*x_problem.parameters(recurse=False), **kwargs).detach().clone()
            z_problem.assign_A_x(next_ax)

        loss_z_prob = z_solve_fn(z_problem, verbose=verbose, **kwargs)

        with torch.no_grad():
            next_z = z_problem.get_z(*z_problem.parameters(recurse=False), **kwargs).detach().clone()
            x_problem.assign_z(next_z)

        if save_intermediates:
            intermediates_list.append((x_problem.get_reconstructed_image(), next_z.detach().cpu().numpy()))

        if verbose:
            print(f"HQS iter {it}, xloss {loss_x_prob}, zloss {loss_z_prob}")

    return intermediates_list


def scheduled_rho_single_hqs_solve(x_problem: Union[SingleProblem, HQS_X_Problem],
                                   x_solver_it: Iterator[HQS_ParameterizedSolveFn],
                                   z_problem: Union[SingleProblem, HQS_Z_Problem],
                                   z_solver_it: Iterator[HQS_ParameterizedSolveFn],
                                   rho_schedule: Iterator[float],
                                   max_iter_hqs: int,
                                   verbose: bool = False,
                                   save_intermediates: bool = False,
                                   **kwargs) \
        -> Optional[List[np.ndarray]]:
    intermediates_list = None
    if save_intermediates:
        intermediates_list = []
    for it, x_solve_fn, z_solve_fn, rho_val in zip(range(max_iter_hqs), x_solver_it, z_solver_it, rho_schedule):

        x_problem.set_rho(rho_val)
        z_problem.set_rho(rho_val)

        loss_x_prob = x_solve_fn(x_problem, verbose=verbose, **kwargs)

        with torch.no_grad():
            next_ax = x_problem.compute_A_x(*x_problem.parameters(recurse=False), **kwargs)
            z_problem.assign_A_x(next_ax)

        loss_z_prob = z_solve_fn(z_problem, verbose=verbose, **kwargs)

        with torch.no_grad():
            next_z = z_problem.get_z(*z_problem.parameters(recurse=False), **kwargs)
            x_problem.assign_z(next_z)

        if save_intermediates:
            intermediates_list.append(x_problem.get_reconstructed_image())

        if verbose:
            print(f"HQS iter {it}, xloss {loss_x_prob}, zloss {loss_z_prob}")

    return intermediates_list


def single_hqs_solve(x_problem: Union[SingleProblem, HQS_X_Problem],
                     x_solver_it: Iterator[HQS_ParameterizedSolveFn],
                     z_problem: Union[SingleProblem, HQS_Z_Problem],
                     z_solver_it: Iterator[HQS_ParameterizedSolveFn],
                     max_iter_hqs: int,
                     rho_value: float,
                     converge_epsilon: float = 1e-5,
                     verbose: bool = False,
                     save_intermediates: bool = False,
                     **kwargs) \
        -> Optional[List[np.ndarray]]:
    x_problem.set_rho(rho_value)
    z_problem.set_rho(rho_value)

    intermediates_list = None
    if save_intermediates:
        intermediates_list = []
    for it, x_solve_fn, z_solve_fn in zip(range(max_iter_hqs), x_solver_it, z_solver_it):

        loss_x_prob = x_solve_fn(x_problem, verbose=verbose, **kwargs)

        with torch.no_grad():
            next_ax = x_problem.compute_A_x(*x_problem.parameters(recurse=False), **kwargs)
            z_problem.assign_A_x(next_ax)

        loss_z_prob = z_solve_fn(z_problem, verbose=verbose, **kwargs)

        with torch.no_grad():
            next_z = z_problem.get_z(*z_problem.parameters(recurse=False), **kwargs)
            x_problem.assign_z(next_z)

        if save_intermediates:
            intermediates_list.append(x_problem.get_reconstructed_image())

        if verbose:
            print(f"HQS iter {it}, xloss {loss_x_prob}, zloss {loss_z_prob}")

    return intermediates_list


class HQS_XGenerator:
    def __init__(self,
                 first_niter: int = 1000,
                 subsequent_niter: int = 500):

        self.iter_count = 0
        self.first_n_iter = first_niter
        self.subsequent_niter = subsequent_niter

    def __iter__(self):
        return self

    def __next__(self) -> HQS_ParameterizedSolveFn:
        if self.iter_count == 0:
            self.iter_count += 1

            def applied_fista_solve(prob: SingleUnconstrainedProblem,
                                    verbose: bool = False,
                                    **kwargs) -> float:
                return single_unconstrained_solve(prob, FistaSolverParams(max_iter=self.first_n_iter),
                                                  verbose=verbose, **kwargs)

            return applied_fista_solve
        else:
            self.iter_count += 1

            def applied_fista_solve(prob: SingleUnconstrainedProblem,
                                    verbose: bool = False,
                                    **kwargs) -> float:
                return single_unconstrained_solve(prob, FistaSolverParams(max_iter=self.subsequent_niter),
                                                  verbose=verbose, **kwargs)

            return applied_fista_solve


@dataclass
class AdamOptimParams:
    n_iters: int
    learn_rate: float


class Adam_HQS_XGenerator:
    def __init__(self,
                 init_sched: List[AdamOptimParams],
                 default_params: AdamOptimParams = AdamOptimParams(10, 5e-2)):

        self.iter_count = 0
        self.init_sched = init_sched
        self.default_params = default_params

    def __iter__(self):
        return self

    def __next__(self) -> HQS_ParameterizedSolveFn:
        if self.iter_count < len(self.init_sched):

            temp = self.iter_count

            def adam_solve(prob: SingleUnconstrainedProblem,
                           verbose: bool = False,
                           **kwargs) -> torch.Tensor:
                optimizer = optim.Adam(prob.parameters(), lr=self.init_sched[temp].learn_rate)
                for i in range(self.init_sched[temp].n_iters):
                    optimizer.zero_grad(set_to_none=True)
                    loss = prob(**kwargs)
                    loss.backward()
                    optimizer.step()
                    if verbose:
                        print(f'iter {i}, mean loss {loss.item()}\r', end='')

                return loss.detach().clone()

            self.iter_count += 1
            return adam_solve

        else:
            def adam_solve(prob: SingleUnconstrainedProblem,
                           verbose: bool = False,
                           **kwargs) -> torch.Tensor:
                optimizer = optim.Adam(prob.parameters(), lr=self.default_params.learn_rate)
                for i in range(self.default_params.n_iters):
                    optimizer.zero_grad(set_to_none=True)
                    loss = prob(**kwargs)
                    loss.backward()
                    optimizer.step()
                    if verbose:
                        print(f'iter {i}, mean loss {loss.item()}\r', end='')

                return loss.detach().clone()

            self.iter_count += 1
            return adam_solve


class BatchParallel_Adam_HQS_XGenerator:
    def __init__(self,
                 init_sched: List[AdamOptimParams],
                 default_params: AdamOptimParams = AdamOptimParams(10, 5e-2)):

        self.iter_count = 0
        self.init_sched = init_sched
        self.default_params = default_params

    def __iter__(self):
        return self

    def __next__(self) -> HQS_ParameterizedSolveFn:
        if self.iter_count < len(self.init_sched):

            temp = self.iter_count

            def adam_solve(prob: BatchParallelUnconstrainedProblem,
                           verbose: bool = False,
                           **kwargs) -> torch.Tensor:
                optimizer = optim.Adam(prob.parameters(), lr=self.init_sched[temp].learn_rate)

                for i in range(self.init_sched[temp].n_iters):
                    optimizer.zero_grad(set_to_none=True)
                    loss_eval = prob(**kwargs)
                    loss = torch.sum(loss_eval)
                    loss.backward()
                    optimizer.step()
                    if verbose:
                        print(f'iter {i}, mean loss {loss.item()}\r', end='')

                return loss_eval.detach().clone()

            self.iter_count += 1
            return adam_solve

        else:
            def adam_solve(prob: BatchParallelUnconstrainedProblem,
                           verbose: bool = False,
                           **kwargs) -> torch.Tensor:
                optimizer = optim.Adam(prob.parameters(), lr=self.default_params.learn_rate)

                for i in range(self.default_params.n_iters):
                    optimizer.zero_grad(set_to_none=True)
                    loss_eval = prob(**kwargs)
                    loss = torch.sum(loss_eval)
                    loss.backward()
                    optimizer.step()
                    if verbose:
                        print(f'iter {i}, mean loss {loss.item()}\r', end='')

                return loss_eval.detach().clone()

            self.iter_count += 1
            return adam_solve


class BatchParallel_HQS_XGenerator:
    def __init__(self,
                 first_niter: int = 1000,
                 subsequent_niter: int = 500):

        self.iter_count = 0
        self.first_n_iter = first_niter
        self.subsequent_niter = subsequent_niter

    def __iter__(self):
        return self

    def __next__(self) -> HQS_ParameterizedSolveFn:
        if self.iter_count == 0:
            self.iter_count += 1

            def applied_fista_solve(prob: BatchParallelUnconstrainedProblem,
                                    verbose: bool = False,
                                    **kwargs) -> torch.Tensor:
                return batch_parallel_unconstrained_solve(prob, FistaSolverParams(max_iter=self.first_n_iter),
                                                          verbose=verbose, **kwargs)

            return applied_fista_solve
        else:
            self.iter_count += 1

            def applied_fista_solve(prob: BatchParallelUnconstrainedProblem,
                                    verbose: bool = False,
                                    **kwargs) -> torch.Tensor:
                return batch_parallel_unconstrained_solve(prob, FistaSolverParams(max_iter=self.subsequent_niter),
                                                          verbose=verbose, **kwargs)

            return applied_fista_solve


class BatchParallel_DirectSolve_HQS_XGenerator:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self) -> HQS_ParameterizedSolveFn:
        return batch_parallel_direct_solve



class DirectSolve_HQS_ZGenerator:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self) -> HQS_ParameterizedSolveFn:
        return single_direct_solve


class BatchParallel_DirectSolve_HQS_ZGenerator:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self) -> HQS_ParameterizedSolveFn:
        return batch_parallel_direct_solve


class BatchParallel_UnblindDenoiserPrior_HQS_ZProb(BatchParallelDirectSolveProblem,
                                                   BatchParallel_HQS_Z_Problem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 batch_size: int,
                 unblind_denoiser_module: Callable[[torch.Tensor, Union[float, torch.Tensor]], torch.Tensor],
                 image_shape: Tuple[int, int],
                 hqs_rho: float = 1.0, # default value
                 prior_lambda: float = 1.0,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.batch_size = batch_size
        self.rho = hqs_rho
        self.prior_lambda = prior_lambda

        self.unblind_denoiser_callable = unblind_denoiser_module
        self.height, self.width = image_shape

        self.dtype = dtype

        #### HQS constants #####################################################
        self.register_buffer('ax_const_tensor', torch.zeros((self.batch_size, self.height, self.width), dtype=dtype))

        #### OPTIMIZATION VARIABLES ############################################
        self.z_image = nn.Parameter(torch.empty((self.batch_size, self.height, self.width), dtype=dtype),
                                    requires_grad=False)
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def set_prior_lambda(self, lambda_val: float) -> None:
        self.prior_lambda = lambda_val

    def assign_A_x(self, Ax: torch.Tensor) -> None:
        self.ax_const_tensor.data[:] = Ax

    def direct_solve(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            noise_sigma2 = self.prior_lambda / self.rho
            image_denoiser_applied = self.unblind_denoiser_callable(self.ax_const_tensor,
                                                                    noise_sigma2)

            self.z_image.data[:] = image_denoiser_applied.data[:]

            return (image_denoiser_applied,)

    def get_z(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def eval_loss(self, *args, **kwargs) -> torch.Tensor:
        return 0.0

    def reinitialize_variables(self) -> None:
        self.ax_const_tensor.data[:] = 0.0
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)

    def get_reconstructed_image_torch(self) -> torch.Tensor:
        return self.z_image.detach().clone()

    def get_reconstructed_image(self) -> np.ndarray:
        return self.z_image.detach().cpu().numpy()


class BatchParallel_MaskedUnblindDenoiserPrior_HQS_ZProb(BatchParallelDirectSolveProblem,
                                                         BatchParallel_HQS_Z_Problem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 batch_size: int,
                 masked_unblind_denoiser_mod: Callable[
                     [torch.Tensor, Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
                 image_shape: Tuple[int, int],
                 mask: Union[torch.Tensor, np.ndarray],
                 hqs_rho: float = 1.0, # default value
                 prior_lambda: float = 1.0,
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.batch_size = batch_size
        self.rho = hqs_rho
        self.prior_lambda = prior_lambda

        self.masked_denoiser_callable = masked_unblind_denoiser_mod
        self.height, self.width = image_shape

        self.dtype = dtype

        ### Constants #####################################
        if isinstance(mask, np.ndarray):
            self.register_buffer('mask', torch.tensor(mask, dtype=self.dtype))
        else:
            self.register_buffer('mask', mask)

        #### HQS constants #####################################################
        self.register_buffer('ax_const_tensor', torch.zeros((self.batch_size, self.height, self.width), dtype=dtype))

        #### OPTIMIZATION VARIABLES ############################################
        self.z_image = nn.Parameter(torch.empty((self.batch_size, self.height, self.width), dtype=dtype),
                                    requires_grad=False)
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def set_prior_lambda(self, lambda_val: float) -> None:
        self.prior_lambda = lambda_val

    def assign_A_x(self, Ax: torch.Tensor) -> None:
        self.ax_const_tensor.data[:] = Ax.data[:]

    def direct_solve(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            noise_sigma2 = self.prior_lambda / self.rho
            image_denoiser_applied = self.masked_denoiser_callable(self.ax_const_tensor,
                                                                   noise_sigma2,
                                                                   self.mask)

            self.z_image.data[:] = image_denoiser_applied.data[:]

            return (image_denoiser_applied,)

    def get_z(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def eval_loss(self, *args, **kwargs) -> torch.Tensor:
        return 0.0

    def reinitialize_variables(self) -> None:
        self.ax_const_tensor.data[:] = 0.0
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)

    def get_reconstructed_image_torch(self) -> torch.Tensor:
        return self.z_image.detach().clone()

    def get_reconstructed_image(self) -> np.ndarray:
        return self.z_image.detach().cpu().numpy()


def construct_create_batched_denoiser_HQS_Z_problem_fn(
        image_shape: Tuple[int, int],
        image_range_tuple: Tuple[float, float],
        device: torch.device,
        valid_region_mask: Optional[np.ndarray] = None) \
        -> Callable[[int], Union[BatchParallel_UnblindDenoiserPrior_HQS_ZProb,
                                 BatchParallel_MaskedUnblindDenoiserPrior_HQS_ZProb]]:

    use_masked_denoiser = valid_region_mask is not None
    if use_masked_denoiser:
        unblind_denoiser_model = denoiser_wrappers.load_masked_drunet_unblind_denoiser(device)
        unblind_denoiser_callable = denoiser_wrappers.make_unblind_apply_dpir_denoiser_with_mask(
            unblind_denoiser_model,
            image_range_tuple, (0.0, 255))

    else:
        unblind_denoiser_model = denoiser_wrappers.load_zhang_drunet_unblind_denoiser(device)
        unblind_denoiser_callable = denoiser_wrappers.make_unblind_apply_zhang_dpir_denoiser(
            unblind_denoiser_model,
            image_range_tuple, (0.0, 255))

    def create_batched_denoiser_HQS_Z_problem(batch_size: int) \
            -> Union[BatchParallel_UnblindDenoiserPrior_HQS_ZProb,
                     BatchParallel_MaskedUnblindDenoiserPrior_HQS_ZProb]:

        if use_masked_denoiser:
            unblind_denoise_hqs_z_prob = BatchParallel_MaskedUnblindDenoiserPrior_HQS_ZProb(
                batch_size,
                unblind_denoiser_callable,
                image_shape,
                valid_region_mask,
            ).to(device)
        else:
            unblind_denoise_hqs_z_prob = BatchParallel_UnblindDenoiserPrior_HQS_ZProb(
                batch_size,
                unblind_denoiser_callable,
                image_shape,
            ).to(device)

        return unblind_denoise_hqs_z_prob

    return create_batched_denoiser_HQS_Z_problem


class MaskedUnblindDenoiserPrior_HQS_ZProb(SingleDirectSolveProblem,
                                           HQS_Z_Problem):
    IMAGE_IDX_ARGS = 0

    def __init__(self,
                 masked_unblind_denoiser_mod: Callable[
                     [torch.Tensor, Union[float, torch.Tensor], torch.Tensor], torch.Tensor],
                 image_shape: Tuple[int, int],
                 mask: Union[torch.Tensor, np.ndarray],
                 hqs_rho: float,
                 prior_lambda: float = 1.0,
                 dtype: torch.dtype = torch.float32):

        super().__init__()

        self.rho = hqs_rho
        self.prior_lambda = prior_lambda

        self.masked_denoiser_callable = masked_unblind_denoiser_mod
        self.height, self.width = image_shape

        self.dtype = dtype

        ### Constants #####################################
        if isinstance(mask, np.ndarray):
            self.register_buffer('mask', torch.tensor(mask, dtype=self.dtype))
        else:
            self.register_buffer('mask', mask)

        #### HQS constants #####################################################
        self.register_buffer('ax_const_tensor', torch.zeros((self.height, self.width), dtype=dtype))

        #### OPTIMIZATION VARIABLES ############################################
        self.z_image = nn.Parameter(torch.empty((self.height, self.width), dtype=dtype),
                                    requires_grad=False)
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def set_prior_lambda(self, lambda_val: float) -> None:
        self.prior_lambda = lambda_val

    def assign_A_x(self, Ax: torch.Tensor) -> None:
        self.ax_const_tensor.data[:] = Ax.data[:]

    def direct_solve(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            noise_sigma2 = self.prior_lambda / self.rho
            image_denoiser_applied = self.masked_denoiser_callable(self.ax_const_tensor[None, :, :],
                                                                   noise_sigma2,
                                                                   self.mask).squeeze(0)

            self.z_image.data[:] = image_denoiser_applied.data[:]

            return (image_denoiser_applied,)

    def get_z(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def eval_loss(self, *args, **kwargs) -> torch.Tensor:
        return 0.0

    def reinitialize_variables(self) -> None:
        self.ax_const_tensor.data[:] = 0.0
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)

    def get_reconstructed_image_torch(self) -> torch.Tensor:
        return self.z_image.detach().clone()

    def get_reconstructed_image(self) -> np.ndarray:
        return self.z_image.detach().cpu().numpy()


class UnblindDenoiserPrior_HQS_ZProb(SingleDirectSolveProblem,
                                     HQS_Z_Problem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 unblind_denoiser_module: Callable[[torch.Tensor, Union[float, torch.Tensor]], torch.Tensor],
                 image_shape: Tuple[int, int],
                 hqs_rho: float,
                 prior_lambda: float = 1.0,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.rho = hqs_rho
        self.prior_lambda = prior_lambda

        self.unblind_denoiser_callable = unblind_denoiser_module
        self.height, self.width = image_shape

        self.dtype = dtype

        #### HQS constants #####################################################
        self.register_buffer('ax_const_tensor', torch.zeros((self.height, self.width), dtype=dtype))

        #### OPTIMIZATION VARIABLES ############################################
        self.z_image = nn.Parameter(torch.empty((self.height, self.width), dtype=dtype),
                                    requires_grad=False)
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)

    def assign_A_x(self, Ax: torch.Tensor) -> None:
        self.ax_const_tensor.data[:] = Ax

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def set_prior_lambda(self, lambda_val: float) -> None:
        self.prior_lambda = lambda_val

    def direct_solve(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            noise_sigma2 = self.prior_lambda / self.rho
            # noise_sigma2 = 1.0 / (self.rho * self.prior_lambda)
            image_denoiser_applied = self.unblind_denoiser_callable(self.ax_const_tensor[None, :, :],
                                                                    noise_sigma2)

            temp_image = image_denoiser_applied.squeeze(0)

            self.z_image.data[:] = temp_image.data[:]

            return (temp_image,)

    def get_z(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def eval_loss(self, *args, **kwargs) -> torch.Tensor:
        return 0.0

    def reinitialize_variables(self) -> None:
        self.ax_const_tensor.data[:] = 0.0
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)

    def get_reconstructed_image(self) -> np.ndarray:
        return self.z_image.detach().cpu().numpy()


class BlindDenoiserPrior_HQS_ZProb(SingleDirectSolveProblem,
                                   HQS_Z_Problem):
    IMAGE_IDX_ARGS = 0
    OBSERVED_SPIKES_KWARGS = 'observed_spikes'

    def __init__(self,
                 denoiser_module: Callable[[torch.Tensor], torch.Tensor],
                 image_shape: Tuple[int, int],
                 hqs_rho: float,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.rho = hqs_rho

        self.denoiser_callable = denoiser_module
        self.height, self.width = image_shape

        self.dtype = dtype

        #### HQS constants #####################################################
        self.register_buffer('ax_const_tensor', torch.zeros((self.height, self.width), dtype=dtype))

        #### OPTIMIZATION VARIABLES ############################################
        self.z_image = nn.Parameter(torch.empty((self.height, self.width), dtype=dtype),
                                    requires_grad=False)
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)

    def assign_A_x(self, Ax: torch.Tensor) -> None:
        self.ax_const_tensor.data[:] = Ax

    def set_rho(self, new_rho: float) -> None:
        self.rho = new_rho

    def set_prior_lambda(self, lambda_val: float) -> None:
        self.prior_lambda = lambda_val

    def direct_solve(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            image_denoiser_applied = self.denoiser_callable(self.ax_const_tensor[None, :, :])
            return (image_denoiser_applied.squeeze(0),)

    def get_z(self, *args, **kwargs) -> torch.Tensor:
        return args[self.IMAGE_IDX_ARGS]

    def eval_loss(self, *args, **kwargs) -> torch.Tensor:
        return 0.0

    def reinitialize_variables(self) -> None:
        self.ax_const_tensor.data[:] = 0.0
        nn.init.uniform_(self.z_image, a=-0.1, b=0.1)
