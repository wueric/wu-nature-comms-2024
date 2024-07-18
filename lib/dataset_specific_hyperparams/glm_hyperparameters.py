from dataclasses import dataclass

import numpy as np

from typing import Optional, Dict, Union

from basis_functions.time_basis_functions import make_cosine_bump_family, backshift_n_samples, \
    trim_basis_reduce_condition_number, trim_basis_by_cond_index, basis_shift_forward
from basis_functions.spatial_basis_functions import build_spline_matrix


@dataclass
class GLMModelHyperparameters:
    spatial_basis: Union[None, np.ndarray]  # shape (n_pixels, n_basis_spat_stim) if specified
    timecourse_basis: np.ndarray  # shape (n_basis_timecourse, n_bins_filter)
    feedback_basis: np.ndarray  # shape (n_basis_feedback, n_bins_filter)
    coupling_basis: np.ndarray  # shape (n_basis_coupling, n_bins_filter)

    neighboring_cell_dist: Dict[str, float]

    l21_reg_const: float = 0.0
    l1_spat_sparse_reg_const: float = 0.0
    l2_prior_reg_const: float = 0.0
    l2_prior_filt_scale: float = 1.0
    wn_model_weight: float = 0.2
    wn_relative_downsample_factor: int = 2
    n_iter_inner: int = 750
    n_iter_outer: int = 2
    n_bins_binom: Optional[int] = None


def make_joint_wn_flashed_cropped_basis_2018_03_01_0_hyperparams() \
        -> Dict[str, GLMModelHyperparameters]:

    A__timecourse = 4.5
    C__timecourse = 1.0

    N_BASIS__timecourse = 15
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:250]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                     N_BASIS__timecourse,
                                                     TIMESTEPS__timecourse)
    bump_basis_timecourse_temp_downsample = bump_basis_timecourse_full[:, ::5]
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_temp_downsample,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_full[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 15)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])


    ###################################################
    A__feedback = 4.5
    C__feedback = 1.0
    N_BASIS__feedback = 15
    TIMESTEPS__feedback = np.r_[0:250]

    bump_basis_feedback_ = make_cosine_bump_family(A__feedback, C__feedback,
                                                   N_BASIS__feedback, TIMESTEPS__feedback)
    bump_basis_feedback_ = trim_basis_reduce_condition_number(bump_basis_feedback_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_feedback_rev = np.ascontiguousarray(bump_basis_feedback_[:, ::-1])

    ####################################################
    A__coupling = 4.5 # was 3.0
    C__coupling = 1.0
    N_BASIS__coupling = 15 # was 8
    TIMESTEPS__coupling = np.r_[0:250]

    bump_basis_coupling_ = make_cosine_bump_family(A__coupling, C__coupling,
                                                   N_BASIS__coupling, TIMESTEPS__coupling)
    bump_basis_coupling_ = trim_basis_reduce_condition_number(bump_basis_coupling_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_coupling__rev = np.ascontiguousarray(bump_basis_coupling_[:, ::-1])

    # parasols first, since we have done the hyperparam
    # tuning for these already
    parasol_neighboring_cell_distance = {'ON parasol': 2.5,
                                        'OFF parasol': 2.5,
                                        'ON midget': 1.75,
                                        'OFF midget': 1.75}

    midget_neighboring_cell_distance = {'ON parasol': 2.5,
                                        'OFF parasol': 2.5,
                                        'ON midget': 1.75,
                                        'OFF midget': 1.75}

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    # These hyperparameters have been optimized already with a fairly rigorous
    # grid search, so no need to further tune them
    ret_dict = {
        'ON parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              parasol_neighboring_cell_distance,
                                              l21_reg_const=1e-5,
                                              l1_spat_sparse_reg_const=1e-6,
                                              wn_relative_downsample_factor=8,
                                              wn_model_weight=1e-2,
                                              n_iter_outer=2),
        'OFF parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               bump_basis_feedback_rev,
                                               bump_basis_coupling__rev,
                                               parasol_neighboring_cell_distance,
                                               l21_reg_const=5.62e-5,
                                               l1_spat_sparse_reg_const=1e-6,
                                               wn_relative_downsample_factor=8,
                                               wn_model_weight=1e-2,
                                               n_iter_outer=2),
        'ON midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             bump_basis_feedback_rev,
                                             bump_basis_coupling__rev,
                                             midget_neighboring_cell_distance,
                                             l21_reg_const=1e-5,
                                             l1_spat_sparse_reg_const=1e-6,
                                             wn_relative_downsample_factor=8,
                                             wn_model_weight=1e-2,
                                             n_iter_outer=2),
        'OFF midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              midget_neighboring_cell_distance,
                                              l21_reg_const=5.62e-5,
                                              l1_spat_sparse_reg_const=3.16e-6,
                                              wn_relative_downsample_factor=8,
                                              wn_model_weight=1e-2,
                                              n_iter_outer=2),
    }

    return ret_dict


def make_joint_wn_flashed_cropped_basis_2017_12_04_5_hyperparams() \
    -> Dict[str, GLMModelHyperparameters]:

    A__timecourse = 4.5
    C__timecourse = 1.0

    N_BASIS__timecourse = 15
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:250]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                         N_BASIS__timecourse,
                                                         TIMESTEPS__timecourse)
    bump_basis_timecourse_temp_downsample = bump_basis_timecourse_full[:, ::5]
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_temp_downsample,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_full[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 5)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])

    ###################################################
    A__feedback = 4.5
    C__feedback = 1.0
    N_BASIS__feedback = 15
    TIMESTEPS__feedback = np.r_[0:250]

    bump_basis_feedback_ = make_cosine_bump_family(A__feedback, C__feedback,
                                                   N_BASIS__feedback, TIMESTEPS__feedback)
    bump_basis_feedback_ = trim_basis_reduce_condition_number(bump_basis_feedback_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_feedback_rev = np.ascontiguousarray(bump_basis_feedback_[:, ::-1])

    ####################################################
    A__coupling = 4.5 # was 3.0
    C__coupling = 1.0
    N_BASIS__coupling = 15 # was 8
    TIMESTEPS__coupling = np.r_[0:250]

    bump_basis_coupling_ = make_cosine_bump_family(A__coupling, C__coupling,
                                                   N_BASIS__coupling, TIMESTEPS__coupling)
    bump_basis_coupling_ = trim_basis_reduce_condition_number(bump_basis_coupling_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_coupling__rev = np.ascontiguousarray(bump_basis_coupling_[:, ::-1])

    # parasols first, since we have done the hyperparam
    # tuning for these already
    parasol_neighboring_cell_distance = {'ON parasol': 6,
                                         'OFF parasol': 6,
                                         'ON midget': 4,
                                         'OFF midget': 4}

    midget_neighboring_cell_distance = {'ON parasol': 6,
                                        'OFF parasol': 6,
                                        'ON midget': 4,
                                        'OFF midget': 4}

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    ret_dict = {
        'ON parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              parasol_neighboring_cell_distance),
        'OFF parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               bump_basis_feedback_rev,
                                               bump_basis_coupling__rev,
                                               parasol_neighboring_cell_distance),
        'ON midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             bump_basis_feedback_rev,
                                             bump_basis_coupling__rev,
                                             midget_neighboring_cell_distance),
        'OFF midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              midget_neighboring_cell_distance)
    }

    return ret_dict


def make_joint_wn_flashed_cropped_basis_2019_11_07_0_hyperparams() \
    -> Dict[str, GLMModelHyperparameters]:

    A__timecourse = 4.5
    C__timecourse = 1.0

    N_BASIS__timecourse = 15
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:250]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                         N_BASIS__timecourse,
                                                         TIMESTEPS__timecourse)
    bump_basis_timecourse_temp_downsample = bump_basis_timecourse_full[:, ::5]
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_temp_downsample,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_full[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 5)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])

    ###################################################
    A__feedback = 4.5
    C__feedback = 1.0
    N_BASIS__feedback = 15
    TIMESTEPS__feedback = np.r_[0:250]

    bump_basis_feedback_ = make_cosine_bump_family(A__feedback, C__feedback,
                                                   N_BASIS__feedback, TIMESTEPS__feedback)
    bump_basis_feedback_ = trim_basis_reduce_condition_number(bump_basis_feedback_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_feedback_rev = np.ascontiguousarray(bump_basis_feedback_[:, ::-1])

    ####################################################
    A__coupling = 4.5 # was 3.0
    C__coupling = 1.0
    N_BASIS__coupling = 15 # was 8
    TIMESTEPS__coupling = np.r_[0:250]

    bump_basis_coupling_ = make_cosine_bump_family(A__coupling, C__coupling,
                                                   N_BASIS__coupling, TIMESTEPS__coupling)
    bump_basis_coupling_ = trim_basis_reduce_condition_number(bump_basis_coupling_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_coupling__rev = np.ascontiguousarray(bump_basis_coupling_[:, ::-1])

    # parasols first, since we have done the hyperparam
    # tuning for these already
    parasol_neighboring_cell_distance = {'ON parasol': 8,
                                         'OFF parasol': 8,
                                         'ON midget': 5,
                                         'OFF midget': 5}

    midget_neighboring_cell_distance = {'ON parasol': 8,
                                        'OFF parasol': 8,
                                        'ON midget': 5,
                                        'OFF midget': 5}

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    ret_dict = {
        'ON parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              parasol_neighboring_cell_distance,
                                              l21_reg_const=1e-4,
                                              l1_spat_sparse_reg_const=4e-7,
                                              wn_relative_downsample_factor=5,
                                              wn_model_weight=5e-2,
                                              n_iter_outer=2,
                                              n_bins_binom=10),
        'OFF parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               bump_basis_feedback_rev,
                                               bump_basis_coupling__rev,
                                               parasol_neighboring_cell_distance,
                                               l21_reg_const=1e-4,
                                               l1_spat_sparse_reg_const=4e-7,
                                               wn_relative_downsample_factor=5,
                                               wn_model_weight=5e-2,
                                               n_iter_outer=2,
                                               n_bins_binom=10),
        'ON midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             bump_basis_feedback_rev,
                                             bump_basis_coupling__rev,
                                             midget_neighboring_cell_distance,
                                             l21_reg_const=1e-4,
                                             l1_spat_sparse_reg_const=1e-6,
                                             wn_relative_downsample_factor=5,
                                             wn_model_weight=5e-2,
                                             n_iter_outer=2,
                                             n_bins_binom=10),
        'OFF midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              midget_neighboring_cell_distance,
                                              l21_reg_const=1e-4,
                                              l1_spat_sparse_reg_const=1e-6,
                                              wn_relative_downsample_factor=5,
                                              wn_model_weight=5e-2,
                                              n_iter_outer=2,
                                              n_bins_binom=10),
    }

    return ret_dict


def make_joint_wn_flashed_cropped_basis_2017_11_29_0_hyperparams() \
        -> Dict[str, GLMModelHyperparameters]:

    A__timecourse = 4.5
    C__timecourse = 1.0

    N_BASIS__timecourse = 15
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:250]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                     N_BASIS__timecourse,
                                                     TIMESTEPS__timecourse)
    bump_basis_timecourse_temp_downsample = bump_basis_timecourse_full[:, ::5]
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_temp_downsample,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_full[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 15)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])


    ###################################################
    A__feedback = 4.5
    C__feedback = 1.0
    N_BASIS__feedback = 15
    TIMESTEPS__feedback = np.r_[0:250]

    bump_basis_feedback_ = make_cosine_bump_family(A__feedback, C__feedback,
                                                   N_BASIS__feedback, TIMESTEPS__feedback)
    bump_basis_feedback_ = trim_basis_reduce_condition_number(bump_basis_feedback_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_feedback_rev = np.ascontiguousarray(bump_basis_feedback_[:, ::-1])

    ####################################################
    A__coupling = 4.5 # was 3.0
    C__coupling = 1.0
    N_BASIS__coupling = 15 # was 8
    TIMESTEPS__coupling = np.r_[0:250]

    bump_basis_coupling_ = make_cosine_bump_family(A__coupling, C__coupling,
                                                   N_BASIS__coupling, TIMESTEPS__coupling)
    bump_basis_coupling_ = trim_basis_reduce_condition_number(bump_basis_coupling_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_coupling__rev = np.ascontiguousarray(bump_basis_coupling_[:, ::-1])

    # parasols first, since we have done the hyperparam
    # tuning for these already
    parasol_neighboring_cell_distance = {'ON parasol': 5,
                                         'OFF parasol': 5,
                                         'ON midget': 3,
                                         'OFF midget': 3}

    midget_neighboring_cell_distance = {'ON parasol': 5,
                                        'OFF parasol': 5,
                                        'ON midget': 3,
                                        'OFF midget': 3}

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    # These hyperparameters have been optimized already with a fairly rigorous
    # grid search, so no need to further tune them
    ret_dict = {
        'ON parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              parasol_neighboring_cell_distance,
                                              wn_relative_downsample_factor=8,
                                              n_iter_outer=2),
        'OFF parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               bump_basis_feedback_rev,
                                               bump_basis_coupling__rev,
                                               parasol_neighboring_cell_distance,
                                               wn_relative_downsample_factor=8,
                                               n_iter_outer=2),
        'ON midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             bump_basis_feedback_rev,
                                             bump_basis_coupling__rev,
                                             midget_neighboring_cell_distance,
                                             wn_relative_downsample_factor=8,
                                             n_iter_outer=2),
        'OFF midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              midget_neighboring_cell_distance,
                                              wn_relative_downsample_factor=8,
                                              n_iter_outer=2),
    }

    return ret_dict


def make_5ms_cropped_basis_2018_08_07_5_hyperparams() -> Dict[str, GLMModelHyperparameters]:

    A__timecourse = 4.5
    C__timecourse = 1.0

    N_BASIS__timecourse = 15
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:250]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                         N_BASIS__timecourse,
                                                         TIMESTEPS__timecourse)
    bump_basis_timecourse_temp_downsample = bump_basis_timecourse_full[:, ::5]
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_temp_downsample,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_temp_downsample[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 3)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])

    ###################################################
    A__feedback = 4.5
    C__feedback = 1.0
    N_BASIS__feedback = 15
    TIMESTEPS__feedback = np.r_[0:250]

    bump_basis_feedback_ = make_cosine_bump_family(A__feedback, C__feedback,
                                                   N_BASIS__feedback, TIMESTEPS__feedback)
    bump_basis_feedback_ = bump_basis_feedback_[:, ::5]
    bump_basis_feedback_ = trim_basis_reduce_condition_number(bump_basis_feedback_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_feedback_rev = np.ascontiguousarray(bump_basis_feedback_[:, ::-1])

    ####################################################
    A__coupling = 4.5 # was 3.0
    C__coupling = 1.0
    N_BASIS__coupling = 15 # was 8
    TIMESTEPS__coupling = np.r_[0:250]

    bump_basis_coupling_ = make_cosine_bump_family(A__coupling, C__coupling,
                                                   N_BASIS__coupling, TIMESTEPS__coupling)
    bump_basis_coupling_ = bump_basis_coupling_[:, ::5]
    bump_basis_coupling_ = trim_basis_reduce_condition_number(bump_basis_coupling_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_coupling__rev = np.ascontiguousarray(bump_basis_coupling_[:, ::-1])

    # parasols first, since we have done the hyperparam
    # tuning for these already
    parasol_neighboring_cell_distance = {'ON parasol': 8,
                                         'OFF parasol': 8,
                                         'ON midget': 5,
                                         'OFF midget': 5}

    midget_neighboring_cell_distance = {'ON parasol': 8,
                                        'OFF parasol': 8,
                                        'ON midget': 5,
                                        'OFF midget': 5}

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    # These hyperparameters have been optimized already with a fairly rigorous
    # grid search, so no need to further tune them
    ret_dict = {
        'ON parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              parasol_neighboring_cell_distance,
                                              wn_relative_downsample_factor=4),
        'OFF parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               bump_basis_feedback_rev,
                                               bump_basis_coupling__rev,
                                               parasol_neighboring_cell_distance,
                                               wn_relative_downsample_factor=4),
        'ON midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             bump_basis_feedback_rev,
                                             bump_basis_coupling__rev,
                                             midget_neighboring_cell_distance,
                                             wn_relative_downsample_factor=4),
        'OFF midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              midget_neighboring_cell_distance,
                                              wn_relative_downsample_factor=4),
    }

    return ret_dict



def make_joint_wn_flashed_cropped_basis_2018_08_07_5_hyperparams() \
        -> Dict[str, GLMModelHyperparameters]:

    A__timecourse = 4.5
    C__timecourse = 1.0

    N_BASIS__timecourse = 15
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:250]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                     N_BASIS__timecourse,
                                                     TIMESTEPS__timecourse)
    bump_basis_timecourse_temp_downsample = bump_basis_timecourse_full[:, ::5]
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_temp_downsample,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_full[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 25)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])


    ###################################################
    A__feedback = 4.5
    C__feedback = 1.0
    N_BASIS__feedback = 15
    TIMESTEPS__feedback = np.r_[0:250]

    bump_basis_feedback_ = make_cosine_bump_family(A__feedback, C__feedback,
                                                   N_BASIS__feedback, TIMESTEPS__feedback)
    bump_basis_feedback_ = trim_basis_reduce_condition_number(bump_basis_feedback_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_feedback_rev = np.ascontiguousarray(bump_basis_feedback_[:, ::-1])

    ####################################################
    A__coupling = 4.5 # was 3.0
    C__coupling = 1.0
    N_BASIS__coupling = 15 # was 8
    TIMESTEPS__coupling = np.r_[0:250]

    bump_basis_coupling_ = make_cosine_bump_family(A__coupling, C__coupling,
                                                   N_BASIS__coupling, TIMESTEPS__coupling)
    bump_basis_coupling_ = trim_basis_reduce_condition_number(bump_basis_coupling_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_coupling__rev = np.ascontiguousarray(bump_basis_coupling_[:, ::-1])

    # parasols first, since we have done the hyperparam
    # tuning for these already
    parasol_neighboring_cell_distance = {'ON parasol': 8,
                                         'OFF parasol': 8,
                                         'ON midget': 5,
                                         'OFF midget': 5}

    midget_neighboring_cell_distance = {'ON parasol': 8,
                                        'OFF parasol': 8,
                                        'ON midget': 5,
                                        'OFF midget': 5}

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    # These hyperparameters have been optimized already with a fairly rigorous
    # grid search, so no need to further tune them
    ret_dict = {
        'ON parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              parasol_neighboring_cell_distance,
                                              l21_reg_const=1e-5,
                                              l1_spat_sparse_reg_const=1e-6,
                                              wn_relative_downsample_factor=4,
                                              wn_model_weight=5e-2,
                                              n_iter_outer=2),
        'OFF parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               bump_basis_feedback_rev,
                                               bump_basis_coupling__rev,
                                               parasol_neighboring_cell_distance,
                                               l21_reg_const=5.62e-5,
                                               l1_spat_sparse_reg_const=1e-6,
                                               wn_relative_downsample_factor=4,
                                               wn_model_weight=5e-2,
                                               n_iter_outer=2),
        'ON midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             bump_basis_feedback_rev,
                                             bump_basis_coupling__rev,
                                             midget_neighboring_cell_distance,
                                             l21_reg_const=1e-5,
                                             l1_spat_sparse_reg_const=1e-6,
                                             wn_relative_downsample_factor=4,
                                             wn_model_weight=5e-2,
                                             n_iter_outer=2),
        'OFF midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              midget_neighboring_cell_distance,
                                              l21_reg_const=5.62e-5,
                                              l1_spat_sparse_reg_const=3.16e-6,
                                              wn_relative_downsample_factor=4,
                                              wn_model_weight=5e-2,
                                              n_iter_outer=2),
    }

    return ret_dict


def make_joint_wn_flashed_cropped_basis_2018_11_12_5_hyperparams() \
        -> Dict[str, GLMModelHyperparameters]:
    A__timecourse = 4.5
    C__timecourse = 1.0

    N_BASIS__timecourse = 15
    MAX_CONDITION_NUMBER = 1e6
    TIMESTEPS__timecourse = np.r_[0:250]

    bump_basis_timecourse_full = make_cosine_bump_family(A__timecourse, C__timecourse,
                                                         N_BASIS__timecourse,
                                                         TIMESTEPS__timecourse)
    bump_basis_timecourse_temp_downsample = bump_basis_timecourse_full[:, ::5]
    trim_basis_ix = trim_basis_by_cond_index(bump_basis_timecourse_temp_downsample,
                                             MAX_CONDITION_NUMBER)
    trimmed_full_basis = basis_shift_forward(bump_basis_timecourse_full[trim_basis_ix:, :])
    bump_basis_timecourse__bs = backshift_n_samples(trimmed_full_basis, 15)
    bump_basis_timecourse__rev = np.ascontiguousarray(bump_basis_timecourse__bs[:, ::-1])

    ###################################################
    A__feedback = 4.5
    C__feedback = 1.0
    N_BASIS__feedback = 15
    TIMESTEPS__feedback = np.r_[0:250]

    bump_basis_feedback_ = make_cosine_bump_family(A__feedback, C__feedback,
                                                   N_BASIS__feedback, TIMESTEPS__feedback)
    bump_basis_feedback_ = trim_basis_reduce_condition_number(bump_basis_feedback_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_feedback_rev = np.ascontiguousarray(bump_basis_feedback_[:, ::-1])

    ####################################################
    A__coupling = 4.5  # was 3.0
    C__coupling = 1.0
    N_BASIS__coupling = 15  # was 8
    TIMESTEPS__coupling = np.r_[0:250]

    bump_basis_coupling_ = make_cosine_bump_family(A__coupling, C__coupling,
                                                   N_BASIS__coupling, TIMESTEPS__coupling)
    bump_basis_coupling_ = trim_basis_reduce_condition_number(bump_basis_coupling_,
                                                              MAX_CONDITION_NUMBER)
    bump_basis_coupling__rev = np.ascontiguousarray(bump_basis_coupling_[:, ::-1])

    # parasols first, since we have done the hyperparam
    # tuning for these already
    parasol_neighboring_cell_distance = {'ON parasol': 8,
                                         'OFF parasol': 8,
                                         'ON midget': 5,
                                         'OFF midget': 5}

    midget_neighboring_cell_distance = {'ON parasol': 8,
                                        'OFF parasol': 8,
                                        'ON midget': 5,
                                        'OFF midget': 5}

    # note that we cannot precompute the spatial basis, since that depends on the
    # crop... We therefore just output the parameters of the basis generation
    ret_dict = {
        'ON parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              parasol_neighboring_cell_distance,
                                              wn_relative_downsample_factor=5),
        'OFF parasol': GLMModelHyperparameters(((45, 45), 'cr'),
                                               bump_basis_timecourse__rev,
                                               bump_basis_feedback_rev,
                                               bump_basis_coupling__rev,
                                               parasol_neighboring_cell_distance,
                                               wn_relative_downsample_factor=5),
        'ON midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                             bump_basis_timecourse__rev,
                                             bump_basis_feedback_rev,
                                             bump_basis_coupling__rev,
                                             midget_neighboring_cell_distance,
                                             wn_relative_downsample_factor=5),
        'OFF midget': GLMModelHyperparameters(((35, 35), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              midget_neighboring_cell_distance,
                                              wn_relative_downsample_factor=5),
    }

    return ret_dict


def make_hyperparameters_2018_08_07_5(spatial_downsample_factor: int,
                                      temporal_binsize: int) \
        -> Dict[str, GLMModelHyperparameters]:
    '''

    :param spatial_downsample_factor:
    :param temporal_binsize:
    :return:
    '''
    ret_dict = {}  # type: Dict[str, GLMModelHyperparameters]

    interval_step_time = temporal_binsize * 0.05

    if spatial_downsample_factor == 2:
        ####################################################
        parasol_spline_basis = build_spline_matrix((32, 32), (11, 11), 'cr').reshape(32 * 32, 11 * 11)

        ####################################################
        A_parasol_timecourse = 5.5
        C_parasol_timecourse = 1.0
        N_BASIS_parasol_timecourse = 18
        TIMESTEPS_parasol_timecourse = np.r_[0:250:interval_step_time]

        bump_basis_timecourse_parasol = make_cosine_bump_family(A_parasol_timecourse, C_parasol_timecourse,
                                                                N_BASIS_parasol_timecourse,
                                                                TIMESTEPS_parasol_timecourse)
        bump_basis_timecourse_parasol = bump_basis_timecourse_parasol[8:]
        bump_basis_timecourse_parasol_bs = backshift_n_samples(bump_basis_timecourse_parasol,
                                                               int(500 / temporal_binsize))
        bump_basis_timecourse_parasol_rev = np.ascontiguousarray(bump_basis_timecourse_parasol_bs[:, ::-1])

        ###################################################
        A_parasol_feedback = 5.5
        C_parasol_feedback = 1.0
        N_BASIS_parasol_feedback = 18
        TIMESTEPS_parasol_feedback = np.r_[0:250:interval_step_time]

        bump_basis_feedback_parasol = make_cosine_bump_family(A_parasol_feedback, C_parasol_feedback,
                                                              N_BASIS_parasol_feedback, TIMESTEPS_parasol_feedback)
        bump_basis_feedback_parsol_rev = np.ascontiguousarray(bump_basis_feedback_parasol[:, ::-1])

        ####################################################
        A_parasol_coupling = 3.2
        C_parasol_coupling = 1.0
        N_BASIS_parasol_coupling = 10
        TIMESTEPS_parasol_coupling = np.r_[0:250:interval_step_time]

        bump_basis_coupling_parasol = make_cosine_bump_family(A_parasol_coupling, C_parasol_coupling,
                                                              N_BASIS_parasol_coupling, TIMESTEPS_parasol_coupling)
        bump_basis_coupling_parasol_rev = np.ascontiguousarray(bump_basis_coupling_parasol[:, ::-1])

        #####################################################
        # parasols first, since we have done the hyperparam
        # tuning for these already
        parasol_neighboring_cell_distance = {'ON parasol': 8,
                                             'OFF parasol': 8,
                                             'ON midget': 5,
                                             'OFF midget': 5}

        on_parasol_hyperparams = GLMModelHyperparameters(parasol_spline_basis,
                                                         bump_basis_timecourse_parasol_rev,
                                                         bump_basis_feedback_parsol_rev,
                                                         bump_basis_coupling_parasol_rev,
                                                         parasol_neighboring_cell_distance,
                                                         l21_reg_const=5e-2)
        ret_dict['ON parasol'] = on_parasol_hyperparams

        off_parasol_hyperparams = GLMModelHyperparameters(parasol_spline_basis,
                                                          bump_basis_timecourse_parasol_rev,
                                                          bump_basis_feedback_parsol_rev,
                                                          bump_basis_coupling_parasol_rev,
                                                          parasol_neighboring_cell_distance,
                                                          l21_reg_const=5e-2)

        ret_dict['OFF parasol'] = off_parasol_hyperparams

        #####################################################
        # midget hyperparams
        midget_spline_basis = build_spline_matrix((24, 24), (13, 13), 'cr')
        midget_neighboring_cell_distance = {'ON parasol': 8,
                                            'OFF parasol': 8,
                                            'ON midget': 5,
                                            'OFF midget': 5}

        on_midget_hyperparams = GLMModelHyperparameters(midget_spline_basis,
                                                        bump_basis_timecourse_parasol_rev,
                                                        bump_basis_feedback_parsol_rev,
                                                        bump_basis_coupling_parasol_rev,
                                                        midget_neighboring_cell_distance,
                                                        l21_reg_const=5e-2)
        ret_dict['ON midget'] = on_midget_hyperparams

        off_midget_hyperparams = GLMModelHyperparameters(midget_spline_basis,
                                                         bump_basis_timecourse_parasol_rev,
                                                         bump_basis_feedback_parsol_rev,
                                                         bump_basis_coupling_parasol_rev,
                                                         midget_neighboring_cell_distance,
                                                         l21_reg_const=5e-2)

        ret_dict['OFF midget'] = off_midget_hyperparams

    return ret_dict


CROPPED_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE2 = {
    ('2018-08-07-5', 'data000') : {
        1: make_joint_wn_flashed_cropped_basis_2018_08_07_5_hyperparams,
        5: make_5ms_cropped_basis_2018_08_07_5_hyperparams,
    },
    ('2017-11-29-0', 'data001') : {
        1: make_joint_wn_flashed_cropped_basis_2017_11_29_0_hyperparams
    },
    ('2017-12-04-5', 'data005') : {
        1: make_joint_wn_flashed_cropped_basis_2017_12_04_5_hyperparams
    },
    ('2019-11-07-0', 'data000') : {
        1: make_joint_wn_flashed_cropped_basis_2019_11_07_0_hyperparams,
    },
    ('2019-11-07-0', 'data003') : {
        1: make_joint_wn_flashed_cropped_basis_2019_11_07_0_hyperparams,
    },
    ('2018-11-12-5', 'data008') : {
        1: make_joint_wn_flashed_cropped_basis_2018_11_12_5_hyperparams
    },
    ('2018-03-01-0', 'data010') : {
        1: make_joint_wn_flashed_cropped_basis_2018_03_01_0_hyperparams,
    }
}
