import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Tuple, List
from functools import reduce

from torch import autocast
from torch.cuda.amp import GradScaler

from convex_optim_base.unconstrained_optim import FistaSolverParams


def _single_backtrack_search(eval_at_point: Callable[[torch.Tensor], float],
                             s_vars_vectorized: torch.Tensor,
                             flat_gradient: torch.Tensor,
                             s_loss: float,
                             step_size: float,
                             backtracking_beta: float) \
        -> Tuple[torch.Tensor, float, float]:
    '''
    :param eval_at_point:
    :param s_vars_vectorized:
    :param flat_gradient:
    :param s_loss:
    :param step_size:
    :param backtracking_beta:
    :return:
    '''

    def _eval_quadratic_approx(next_vars: torch.Tensor,
                               approx_center: torch.Tensor,
                               step: float) -> float:
        div_mul = 1.0 / (2.0 * step)
        diff = next_vars - approx_center
        return (s_loss + torch.sum(flat_gradient * diff) + div_mul * torch.sum(diff * diff)).item()

    # we don't care for computing gradients here
    with torch.no_grad():
        stepped_vars_vectorized = s_vars_vectorized - flat_gradient * step_size
        quad_approx_val = _eval_quadratic_approx(stepped_vars_vectorized, s_vars_vectorized, step_size)

        # assign variables, evaluate loss, unset variables
        stepped_loss = eval_at_point(stepped_vars_vectorized)

        while quad_approx_val < stepped_loss:
            step_size *= backtracking_beta

            stepped_vars_vectorized = s_vars_vectorized - flat_gradient * step_size
            quad_approx_val = _eval_quadratic_approx(stepped_vars_vectorized, s_vars_vectorized, step_size)

            # assign variables, evaluate loss, unset variables
            stepped_loss = eval_at_point(stepped_vars_vectorized)

        return stepped_vars_vectorized, step_size, stepped_loss


class NewStyleSingleFISTAOptim(optim.Optimizer):
    '''
    We do not support parameter groups, only 1 parameter group allowed
    '''

    def __init__(self,
                 params,
                 lr: float = 1.0,
                 backtrack_beta: float = 0.5,
                 eval_loss_callable_fn: Callable[[torch.Tensor, ...], float] = lambda x: 0,
                 prox_callable_fn: Callable[[torch.Tensor, ...], Tuple[torch.Tensor, ...]] = lambda x: x,
                 verbose: bool = False):

        defaults = dict(
            lr=lr,
            backtrack_beta=backtrack_beta,
            eval_loss_callable_fn=eval_loss_callable_fn,
            prox_callable_fn=prox_callable_fn
        )

        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Prox doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._problem_numvar_cache = None
        self.verbose = verbose

    def _problem_numvar(self) -> int:
        if self._problem_numvar_cache is None:
            self._problem_numvar_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._problem_numvar_cache

    def _gather_flat_grad(self) -> torch.Tensor:
        '''
        Gathers and flattens gradients from the .grad attribute of all of the variables;
            assumes that the gradient has already been computed using .backward()

        Returns a new Tensor, does not modify the gradients
        :return:
        '''
        buffer = []
        for p in self._params:
            if p.grad is None:
                p_grad = p.new_zeros().view(-1)
            else:
                p_grad = p.grad.view(-1)
            buffer.append(p_grad)
        return torch.cat(buffer, dim=0)

    def _flatten_vars(self, var_list: List[torch.Tensor]) -> torch.Tensor:
        buffer = []
        for p in var_list:
            buffer.append(p.view(-1))
        return torch.cat(buffer, dim=0)

    def _gather_flat_vars(self) -> torch.Tensor:
        '''
        Gathers and flattens variables

        Returns a new Tensor, does not modify the gradient
        :return:
        '''
        return self._flatten_vars(self._params)

    def _unflatten_vars(self, flat_vars: torch.Tensor) -> List[torch.Tensor]:
        buffer, offset = [], 0
        for p in self._params:
            numel = p.numel()
            buffer.append(flat_vars[offset:offset + numel].detach().clone().view_as(p))
            offset += numel
        assert offset == self._problem_numvar()
        return buffer

    def _clone_variables(self) -> List[torch.Tensor]:
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_variables(self, variables: List[torch.Tensor]):
        for p, p_data in zip(self._params, variables):
            p.data[:] = p_data[:]

    def _flat_eval_loss_nograd(self, flat_vars: torch.Tensor) -> float:

        group = self.param_groups[0]
        nograd_loss_eval = group['eval_loss_callable_fn']

        unflat_vars = self._unflatten_vars(flat_vars)
        loss_eval = nograd_loss_eval(*unflat_vars)
        return loss_eval

    def step(self, closure=None):
        '''
        This implements a single step of the FISTA optim, without proximal projection

        There are a few subtleties here that make this implementation weird

            1. We use the descent version of FISTA, where we only accept
                new values for the variables if they decrease the loss function.
                This means that we have to keep track of the best variables and
                loss that we've seen so far as a state variable. This also means
                that we need a special last step of FISTA to take the best set
                of variables
            2. We use a backtracking line search within each FISTA step. Doing this
                backtracking line search requires re-evaluation of the loss function.
                We hide an additional closure that doesn't touch autograd to do this
            3. FISTA uses the gradients at s_iter, which isn't actually the best guess
                at any iteration. The first value of s_iter is the same as the initialization
                point, and subsequent values of s_iter need to be calculated from
                values from the previous iteration (momentum)

                Since mixed-precision training doesn't allow usage of the closure
                argument, the only way we can compute the gradient around s_iter is to
                put the value of s_iter that the next iteration needs into the variables,
                and then have a special .finish() last step that puts in the real solution
                after the last iteration

        :param closure: can't use
        :return:
        '''

        group = self.param_groups[0]

        # NOTE: we keep global state inside the first param
        global_state = self.state[self._params[0]]

        global_state.setdefault('n_iter', 0)
        global_state.setdefault('current_lr', group['lr'])
        global_state.setdefault('t_iter_min1', 1.0)
        global_state.setdefault('t_iter', (1.0 + np.sqrt(1 + 4)) / 2.0)

        # gradient is guaranteed to be for this iteration
        # since the caller of .step() should have called .backward()
        # on the correct value of s_iter
        gradient_flattened = self._gather_flat_grad()
        s_iter = self._gather_flat_vars()
        s_loss_val = self._flat_eval_loss_nograd(s_iter)

        global_state.setdefault('best_loss', s_loss_val)
        global_state.setdefault('best_vars', s_iter)

        vars_iter, current_lr, candidate_loss = _single_backtrack_search(
            lambda x: self._flat_eval_loss_nograd(x),
            s_iter,
            gradient_flattened,
            s_loss_val,
            global_state['current_lr'],
            group['backtrack_beta']
        )

        # first check if we should accept the solution
        best_loss = global_state.get('best_loss')
        if candidate_loss < best_loss:
            global_state['best_loss'] = candidate_loss
            global_state['best_vars'] = vars_iter

        if self.verbose:
            print(
                f"iter {global_state['n_iter']} loss={global_state['best_loss']}\r", end="")

        # now we need to update state variables for the next iteration
        t_iter_min1 = global_state['t_iter_min1']
        t_iter = global_state['t_iter']
        alpha_iter = t_iter_min1 / t_iter

        global_state['t_iter_min1'] = t_iter
        global_state['t_iter'] = (1.0 + np.sqrt(1 + 4 * t_iter ** 2)) / 2.0

        prev_vars_iter = global_state.get('prev_vars_iter')
        if prev_vars_iter is None:
            prev_vars_iter = s_iter

        s_iter = vars_iter + alpha_iter * (vars_iter - prev_vars_iter)
        self._set_variables(self._unflatten_vars(s_iter))

        global_state['prev_vars_iter'] = vars_iter
        global_state['current_lr'] = current_lr
        global_state['n_iter'] += 1

    def finish(self):
        '''
        Call this after the last iteration of FISTA to set the parameters
            to the correct optimal values
        :return:
        '''
        # NOTE: FISTA will keep global state in the first param
        global_state = self.state[self._params[0]]
        if 'best_vars' in global_state:
            self._set_variables(self._unflatten_vars(global_state['best_vars']))


def _optim_unconstrained_FISTA(model: nn.Module,
                               fista_params: FistaSolverParams,
                               *data_args) -> float:
    no_autocast_fn = model.make_loss_eval_callable(*data_args)

    def autocast_nograd_loss_eval(*args) -> float:
        with autocast('cuda'):
            return no_autocast_fn(*args)

    optimizer = NewStyleSingleFISTAOptim(
        model.parameters(),
        lr=fista_params.initial_learning_rate,
        backtrack_beta=fista_params.backtracking_beta,
        eval_loss_callable_fn=autocast_nograd_loss_eval,
        verbose=False
    )

    scaler = GradScaler()

    for iter in range(fista_params.max_iter):
        optimizer.zero_grad()

        with autocast('cuda'):
            loss = model(*data_args)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    optimizer.finish()

    with torch.no_grad(), autocast('cuda'):
        return model(*data_args).item()


def _batch_backtrack_search(batched_eval_at_point: Callable[[torch.Tensor], torch.Tensor],
                            batched_s_vars_flat: torch.Tensor,
                            batched_flat_gradient: torch.Tensor,
                            batched_s_loss: torch.Tensor,
                            step_size: torch.Tensor,
                            backtracking_beta: float) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''

    :param batched_eval_at_point:
    :param batched_s_vars_flat:
    :param batched_flat_gradient:
    :param batched_s_loss:
    :param step_size:
    :param backtracking_beta:
    :return:
    '''

    def _eval_quadratic_approx(batched_next_vars: torch.Tensor,
                               batched_approx_center: torch.Tensor,
                               batched_step: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # shape (batch, )
            div_mul = 1.0 / (2.0 * batched_step)

            # shape (batch, ?)
            diff = batched_next_vars - batched_approx_center

            # shape (batch, )
            gradient_term = torch.sum(batched_flat_gradient * diff, dim=1)
            diff_term = torch.sum(diff * diff, dim=1)

            # shape (batch, )
            return batched_s_loss + gradient_term + div_mul * diff_term

    with torch.no_grad():
        # shape (n_problems, ?)
        stepped_vars_vectorized = batched_s_vars_flat - batched_flat_gradient * step_size[:, None]

        # shape (n_problems, )
        quad_approx_val = _eval_quadratic_approx(stepped_vars_vectorized, batched_s_vars_flat, step_size)

        # shape (n_problems, )
        stepped_loss = batched_eval_at_point(stepped_vars_vectorized)

        approx_smaller_loss = quad_approx_val < stepped_loss

        while torch.any(approx_smaller_loss):
            # shape (n_problems, )
            update_multiplier = approx_smaller_loss.float() * backtracking_beta \
                                + (~approx_smaller_loss).float() * 1.0
            # shape (n_problems, )
            step_size = step_size * update_multiplier

            # shape (n_problems, ?)
            stepped_vars_vectorized = batched_s_vars_flat - batched_flat_gradient * step_size[:, None]

            # shape (n_problems, )
            quad_approx_val = _eval_quadratic_approx(stepped_vars_vectorized, batched_s_vars_flat, step_size)

            # shape (n_problems, )
            stepped_loss = batched_eval_at_point(stepped_vars_vectorized)

            approx_smaller_loss = quad_approx_val < stepped_loss

        return stepped_vars_vectorized, step_size, stepped_loss


class NewStyleBatchFISTAOptim(optim.Optimizer):
    '''
    We do not support parameter groups
    '''

    def __init__(self,
                 params,
                 batch_size: int = 1,
                 lr: float = 1.0,
                 backtrack_beta: float = 0.5,
                 batch_eval_loss_callable_fn: Callable[[torch.Tensor, ...], torch.Tensor] = lambda x: 0,
                 verbose: bool = False):

        defaults = dict(
            batch_size=batch_size,
            lr=lr,
            backtrack_beta=backtrack_beta,
            batch_eval_loss_callable_fn=batch_eval_loss_callable_fn,
        )

        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Prox doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._problem_numvar_cache = None
        self.verbose = verbose
        self.batch_size = batch_size

    def _problem_numvar(self) -> int:
        if self._problem_numvar_cache is None:
            total_numvar = reduce(lambda total, p: total + p.numel(), self._params, 0)
            self._problem_numvar_cache = total_numvar // self.batch_size
        return self._problem_numvar_cache

    def _gather_batch_flat_grad(self) -> torch.Tensor:
        '''
        Gathers and flattens gradients from the .grad attribute of all of the variables;
            assumes that the gradient has already been computed using .backward()

        Returns a new Tensor, does not modify the parameters or the gradients
        :return:
        '''
        buffer = []
        for p in self._params:
            if p.grad is None:
                p_grad = p.new_zeros().view(self.batch_size, -1)
            else:
                p_grad = p.grad.view(self.batch_size, -1)
            buffer.append(p_grad)
        return torch.cat(buffer, dim=1)

    def _flatten_batch_vars(self, var_list: List[torch.Tensor]) -> torch.Tensor:
        buffer = []
        for p in var_list:
            buffer.append(p.view(self.batch_size, -1))
        return torch.cat(buffer, dim=1)

    def _gather_batch_flat_vars(self) -> torch.Tensor:
        return self._flatten_batch_vars(self._params)

    def _unflatten_batch_vars(self, flat_batch_vars: torch.Tensor) -> List[torch.Tensor]:
        buffer, offset = [], 0
        for p in self._params:
            numel = p.numel()
            buffer.append(flat_batch_vars[offset + offset + numel].detach().clone().view_as(p))
            offset += numel
        assert offset == self._problem_numvar()
        return buffer

    def _clone_variables(self) -> List[torch.Tensor]:
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_variables(self, variables: List[torch.Tensor]):
        for p, p_data in zip(self._params, variables):
            p.data[:] = p_data[:]

    def _flat_eval_batch_loss_nograd(self, flat_batch_vars: torch.Tensor) -> torch.Tensor:
        group = self.param_groups[0]
        nograd_loss_eval = group['batch_eval_loss_callable_fn']

        unflat_vars = self._unflatten_batch_vars(flat_batch_vars)
        loss_eval = nograd_loss_eval(*unflat_vars)
        return loss_eval

    def step(self, closure=None):
        '''
        This implements a single step of FISTA optimization without proximal projection,
            for a batched problem (parallel multiproblem) optimization

        There are a few subtleties here that make this implementation weird

            1. We use the descent version of FISTA, where we only accept
                new values for the variables if they decrease the loss function.
                This means that we have to keep track of the best variables and
                loss that we've seen so far as a state variable. This also means
                that we need a special last step of FISTA to take the best set
                of variables
            2. We use a backtracking line search within each FISTA step. Doing this
                backtracking line search requires re-evaluation of the loss function.
                We hide an additional closure that doesn't touch autograd to do this
            3. FISTA uses the gradients at s_iter, which isn't actually the best guess
                at any iteration. The first value of s_iter is the same as the initialization
                point, and subsequent values of s_iter need to be calculated from
                values from the previous iteration (momentum)

                Since mixed-precision training doesn't allow usage of the closure
                argument, the only way we can compute the gradient around s_iter is to
                put the value of s_iter that the next iteration needs into the variables,
                and then have a special .finish() last step that puts in the real solution
                after the last iteration

        :param closure:
        :return:
        '''

        group = self.param_groups[0]

        # NOTE: prox FISTA will keep global state in the first param
        global_state = self.state[self._params[0]]

        global_state.setdefault('n_iter', 0)
        global_state.setdefault('t_iter_min1', 1.0)
        global_state.setdefault('t_iter', (1.0 + np.sqrt(1 + 4)) / 2.0)

        # gradient is guaranteed to be for this iteration
        # since the caller of .step() should have called .backward()
        # on the correct value of s_iter
        batch_gradient_flattened = self._gather_batch_flat_grad()
        batched_s_iter = self._gather_batch_flat_vars()
        batched_s_loss_val = self._flat_eval_batch_loss_nograd(batched_s_iter)

        global_state.setdefault('current_lr', group['lr'] * torch.ones_like(batched_s_loss_val))
        global_state.setdefault('best_loss', batched_s_loss_val)
        global_state.setdefault('best_vars', batched_s_iter)

        batched_vars_iter, batched_current_lr, batched_candidate_loss = _batch_backtrack_search(
            lambda x: self._flat_eval_batch_loss_nograd(x),
            batched_s_iter,
            batch_gradient_flattened,
            batched_s_loss_val,
            global_state['current_lr'],
            group['backtrack_beta']
        )

        # check if we should accept some (if any) solutions
        is_improvement = (batched_candidate_loss < global_state['best_loss'])
        old_best_loss, old_best_vars = global_state['best_loss'], global_state['best_vars']
        old_best_loss[is_improvement] = batched_candidate_loss[is_improvement]
        old_best_vars.data[is_improvement, :] = batched_vars_iter[is_improvement, :]
        global_state['best_loss'] = old_best_loss
        global_state['best_vars'] = old_best_vars

        if self.verbose:
            print(f"iter {global_state['n_iter']}, mean loss={torch.mean(old_best_loss).item()}\r", end='')

        # now we need to update state variables for the next iteration
        t_iter_min1 = global_state['t_iter_min1']
        t_iter = global_state['t_iter']
        alpha_iter = t_iter_min1 / t_iter

        global_state['t_iter_min1'] = t_iter
        global_state['t_iter'] = (1.0 + np.sqrt(1 + 4 * t_iter ** 2)) / 2.0

        prev_vars_iter = global_state.get('prev_vars_iter')
        if prev_vars_iter is None:
            prev_vars_iter = batched_s_iter

        batched_s_iter = batched_vars_iter + alpha_iter * (batched_vars_iter - prev_vars_iter)
        self._set_variables(self._unflatten_batch_vars(batched_s_iter))

        global_state['prev_vars_iter'] = batched_vars_iter
        global_state['current_lr'] = batched_current_lr
        global_state['n_iter'] += 1

    def finish(self):
        '''
        Call this after the last iteration of FISTA to set the parameters
            to the correct optimal values
        :return:
        '''
        # NOTE: prox FISTA will keep global state in the first param
        global_state = self.state[self._params[0]]
        self._set_variables(self._unflatten_batch_vars(global_state['best_vars']))


def _batch_optim_unconstrained_FISTA(model: nn.Module,
                                     fista_params: FistaSolverParams,
                                     *data_args):

    no_autocast_fn = model.make_loss_eval_callable(*data_args)

    def autocast_nograd_loss_eval(*args) -> torch.Tensor:
        with autocast('cuda'):
            return no_autocast_fn(*args)

    optimizer = NewStyleBatchFISTAOptim(
        model.parameters(),
        lr=fista_params.initial_learning_rate,
        backtrack_beta=fista_params.backtracking_beta,
        batch_eval_loss_callable_fn=autocast_nograd_loss_eval,
        verbose=True
    )

    scaler = GradScaler()

    for iter in range(fista_params.max_iter):
        optimizer.zero_grad()

        with autocast('cuda'):
            loss = torch.sum(model(*data_args))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    optimizer.finish()

    with torch.no_grad(), autocast('cuda'):
        return model(*data_args).item()
