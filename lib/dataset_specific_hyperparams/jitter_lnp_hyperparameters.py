import numpy as np

from typing import Optional, Dict, Union
from dataclasses import dataclass

from basis_functions.time_basis_functions import make_cosine_bump_family, backshift_n_samples, \
    trim_basis_reduce_condition_number, trim_basis_by_cond_index, basis_shift_forward

@dataclass
class LNPModelHyperparameters:
    spatial_basis: Union[None, np.ndarray]  # shape (n_pixels, n_basis_spat_stim) if specified
    timecourse_basis: np.ndarray  # shape (n_basis_timecourse, n_bins_filter)
    l1_spat_sparse_reg_const: float = 0.0


# FIXME fix the time component
def LNP_make_joint_wn_jitter_cropped_basis_2018_08_07_5_hyperparams() \
        -> Dict[str, LNPModelHyperparameters]:
    A__timecourse = 4.2
    C__timecourse = 1.0

    N_BASIS__timecourse = 8
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:30]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                         N_BASIS__timecourse,
                                                         TIMESTEPS__timecourse)
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_full,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_full[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 1)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    # These hyperparameters have been optimized already with a fairly rigorous
    # grid search, so no need to further tune them
    ret_dict = {
        'ON parasol': LNPModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              l1_spat_sparse_reg_const=1e-7),
        'OFF parasol': LNPModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               l1_spat_sparse_reg_const=1e-7),
        'ON midget': LNPModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             l1_spat_sparse_reg_const=1e-7),
        'OFF midget': LNPModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              l1_spat_sparse_reg_const=3.16e-6),
    }

    return ret_dict


# FIXME fix the time component
def LNP_make_joint_wn_jitter_cropped_basis_2017_11_29_0_hyperparams() \
        -> Dict[str, LNPModelHyperparameters]:
    A__timecourse = 4.2
    C__timecourse = 1.0

    N_BASIS__timecourse = 8
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:30]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                         N_BASIS__timecourse,
                                                         TIMESTEPS__timecourse)
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_full,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_full[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 1)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    # These hyperparameters have been optimized already with a fairly rigorous
    # grid search, so no need to further tune them
    ret_dict = {
        'ON parasol': LNPModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              l1_spat_sparse_reg_const=1e-7),
        'OFF parasol': LNPModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               l1_spat_sparse_reg_const=1e-7),
        'ON midget': LNPModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             l1_spat_sparse_reg_const=1e-7),
        'OFF midget': LNPModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              l1_spat_sparse_reg_const=3.16e-6),
    }

    return ret_dict


# FIXME fix the time component
def LNP_make_joint_wn_jitter_cropped_basis_2019_11_07_0_hyperparams() \
        -> Dict[str, LNPModelHyperparameters]:
    A__timecourse = 4.2
    C__timecourse = 1.0

    N_BASIS__timecourse = 8
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:30]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                         N_BASIS__timecourse,
                                                         TIMESTEPS__timecourse)
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_full,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_full[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 1)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    # These hyperparameters have been optimized already with a fairly rigorous
    # grid search, so no need to further tune them
    ret_dict = {
        'ON parasol': LNPModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              l1_spat_sparse_reg_const=1e-7),
        'OFF parasol': LNPModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               l1_spat_sparse_reg_const=1e-7),
        'ON midget': LNPModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             l1_spat_sparse_reg_const=1e-7),
        'OFF midget': LNPModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              l1_spat_sparse_reg_const=3.16e-6),
    }

    return ret_dict


# FIXME fix the time component
def LNP_make_joint_wn_jitter_cropped_basis_2018_11_12_5_hyperparams() \
        -> Dict[str, LNPModelHyperparameters]:
    A__timecourse = 4.2
    C__timecourse = 1.0

    N_BASIS__timecourse = 8
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:30]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                         N_BASIS__timecourse,
                                                         TIMESTEPS__timecourse)
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_full,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_full[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 1)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    # These hyperparameters have been optimized already with a fairly rigorous
    # grid search, so no need to further tune them
    ret_dict = {
        'ON parasol': LNPModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              l1_spat_sparse_reg_const=1e-7),
        'OFF parasol': LNPModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               l1_spat_sparse_reg_const=1e-7),
        'ON midget': LNPModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             l1_spat_sparse_reg_const=1e-7),
        'OFF midget': LNPModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              l1_spat_sparse_reg_const=3.16e-6),
    }

    return ret_dict


# FIXME fix the time component
def LNP_make_joint_wn_jitter_cropped_basis_2018_03_01_0_hyperparams() \
        -> Dict[str, LNPModelHyperparameters]:
    A__timecourse = 4.2
    C__timecourse = 1.0

    N_BASIS__timecourse = 8
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:30]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                         N_BASIS__timecourse,
                                                         TIMESTEPS__timecourse)
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_full,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_full[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 1)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    # These hyperparameters have been optimized already with a fairly rigorous
    # grid search, so no need to further tune them
    ret_dict = {
        'ON parasol': LNPModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              l1_spat_sparse_reg_const=1e-7),
        'OFF parasol': LNPModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               l1_spat_sparse_reg_const=1e-7),
        'ON midget': LNPModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             l1_spat_sparse_reg_const=1e-7),
        'OFF midget': LNPModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              l1_spat_sparse_reg_const=3.16e-6),
    }

    return ret_dict


CROPPED_JOINT_WN_LNP_HYPERPARAMETERS_FN_BY_PIECE2 = {
    ('2018-08-07-5', 'data011') : LNP_make_joint_wn_jitter_cropped_basis_2018_08_07_5_hyperparams,
    ('2019-11-07-0', 'data003') : LNP_make_joint_wn_jitter_cropped_basis_2019_11_07_0_hyperparams,
    ('2018-11-12-5', 'data002'): LNP_make_joint_wn_jitter_cropped_basis_2018_11_12_5_hyperparams,
}


CROPPED_FLASHED_WN_LNP_HYPERPARAMETERS_FN_BY_PIECE = {
    ('2018-08-07-5', 'data000'): LNP_make_joint_wn_jitter_cropped_basis_2018_08_07_5_hyperparams,
    ('2018-11-12-5', 'data008'): LNP_make_joint_wn_jitter_cropped_basis_2018_11_12_5_hyperparams,
    ('2017-11-29-0', 'data001'): LNP_make_joint_wn_jitter_cropped_basis_2018_11_12_5_hyperparams,
    ('2018-03-01-0', 'data010'): LNP_make_joint_wn_jitter_cropped_basis_2018_03_01_0_hyperparams,
}

