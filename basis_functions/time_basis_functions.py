import numpy as np

def make_cosine_bump_family(a: float,
                            c: float,
                            n_basis: int,
                            t_timesteps: np.ndarray) -> np.ndarray:
    log_term = a * np.log(t_timesteps + c)  # shape (n_timesteps, )
    phases = np.r_[0.0:n_basis * np.pi / 2:np.pi / 2]  # shape (n_basis, )
    log_term_with_phases = -phases[:, None] + log_term[None, :]  # shape (n_basis, n_timesteps)

    should_keep = np.logical_and(log_term_with_phases >= -np.pi,
                                 log_term_with_phases <= np.pi)

    cosine_all = 0.5 * np.cos(log_term_with_phases) + 0.5
    cosine_all[~should_keep] = 0.0

    return cosine_all


def backshift_n_samples(bump_basis: np.ndarray, n_samples: int) -> np.ndarray:
    '''
    To avoid self-feedback (i.e. peeking at the observed spikes
        from the current time bin), we need to shift the bump
        functions back by 1 time bin to guarantee that the weights for
        the current time bin (the first entry) in the bump basis vectors
        is exactly zero
    :param bump_basis:
    :return:
    '''

    bump_basis_shifted = np.zeros_like(bump_basis)
    bump_basis_shifted[:, n_samples:] = bump_basis[:, :-n_samples]
    return bump_basis_shifted


def trim_basis_by_cond_index(basis_set: np.ndarray,
                             max_condition_number: float) -> int:

    n_basis, basis_samples = basis_set.shape
    i = 0
    while i < n_basis:

        subset_basis = basis_set[i:, :]
        inv_mat = subset_basis @ subset_basis.T

        cond_number = np.linalg.cond(inv_mat)

        if cond_number < max_condition_number:
            break

        i += 1

    return i


def basis_shift_forward(basis_set: np.ndarray):

    n_basis, basis_samples = basis_set.shape

    # now we need to shift forward
    forward_shift_amount = np.nonzero(basis_set[0, :])[0][0]

    basis_set_shifted = np.zeros_like(basis_set)
    basis_set_shifted[:, :basis_samples - forward_shift_amount] = basis_set[:, forward_shift_amount:]

    return basis_set_shifted


def trim_basis_reduce_condition_number(basis_set: np.ndarray,
                                       max_condition_number: float) -> np.ndarray:
    '''

    :param basis_set: cosine bump basis, in the original order returned by make_cosine_bump_family
    :param max_condition_number: maximum allowed condition number of the matrix A A^T,
        where A is basis_set
    :return:
    '''
    n_basis, basis_samples = basis_set.shape
    i = 0
    while i < n_basis:

        subset_basis = basis_set[i:, :]
        inv_mat = subset_basis @ subset_basis.T

        cond_number = np.linalg.cond(inv_mat)

        if cond_number < max_condition_number:
            break

        i += 1

    reduced_basis_set = basis_set[i:, :]

    # now we need to shift forward
    forward_shift_amount = np.nonzero(reduced_basis_set[0, :])[0][0]

    reduced_basis_set_shifted = np.zeros_like(reduced_basis_set)
    reduced_basis_set_shifted[:, :basis_samples - forward_shift_amount] = reduced_basis_set[:, forward_shift_amount:]

    return reduced_basis_set_shifted
