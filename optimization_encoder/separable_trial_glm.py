from convex_optim_base.unconstrained_optim import UnconstrainedSolverParams, FistaSolverParams
from convex_optim_base.prox_optim import ProxFISTASolverParams


class UnconstrainedSolverParameterGenerator:
    def __init__(self,
                 first_niter: int = 500,
                 subsequent_niter: int = 1):

        self.iter_count = 0
        self.first_n_iter = first_niter
        self.subsequent_niter = subsequent_niter

    def __iter__(self):
        return self

    def __next__(self) -> UnconstrainedSolverParams:
        if self.iter_count == 0:
            self.iter_count += 1
            return FistaSolverParams(
                initial_learning_rate=1.0,
                max_iter=self.first_n_iter,
                converge_epsilon=1e-6,
                backtracking_beta=0.5)
        else:
            self.iter_count += 1
            return FistaSolverParams(
                initial_learning_rate=1.0,
                max_iter=self.subsequent_niter,
                converge_epsilon=1e-6,
                backtracking_beta=0.5)


class UnconstrainedFISTASolverParameterGenerator:
    def __init__(self,
                 first_niter: int = 500,
                 subsequent_niter: int = 1):

        self.iter_count = 0
        self.first_n_iter = first_niter
        self.subsequent_niter = subsequent_niter

    def __iter__(self):
        return self

    def __next__(self) -> FistaSolverParams:
        if self.iter_count == 0:
            self.iter_count += 1
            return FistaSolverParams(
                initial_learning_rate=1.0,
                max_iter=self.first_n_iter,
                converge_epsilon=1e-6,
                backtracking_beta=0.5)
        else:
            self.iter_count += 1
            return FistaSolverParams(
                initial_learning_rate=1.0,
                max_iter=self.subsequent_niter,
                converge_epsilon=1e-6,
                backtracking_beta=0.5)


class ProxSolverParameterGenerator:
    def __init__(self,
                 first_niter: int = 500,
                 subsequent_niter: int = 1):

        self.iter_count = 0
        self.first_n_iter = first_niter
        self.subsequent_niter = subsequent_niter

    def __iter__(self):
        return self

    def __next__(self) -> ProxFISTASolverParams:
        if self.iter_count == 0:
            self.iter_count += 1
            return ProxFISTASolverParams(
                initial_learning_rate=1.0,
                max_iter=self.first_n_iter,
                converge_epsilon=1e-6,
                backtracking_beta=0.5)
        else:
            self.iter_count += 1
            return ProxFISTASolverParams(
                initial_learning_rate=1.0,
                max_iter=self.subsequent_niter,
                converge_epsilon=1e-6,
                backtracking_beta=0.5)
