from typing import Dict

import numpy as np

from basis_functions.time_basis_functions import make_cosine_bump_family, trim_basis_by_cond_index, basis_shift_forward, \
    backshift_n_samples, trim_basis_reduce_condition_number
from lib.dataset_specific_hyperparams.glm_hyperparameters import GLMModelHyperparameters


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
        'ON parasol': GLMModelHyperparameters(((25, 25), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              parasol_neighboring_cell_distance),
        'OFF parasol': GLMModelHyperparameters(((25, 25), 'cr'),
                                               bump_basis_timecourse__rev,
                                               bump_basis_feedback_rev,
                                               bump_basis_coupling__rev,
                                               parasol_neighboring_cell_distance),
        'ON midget': GLMModelHyperparameters(((13, 13), 'cr'),
                                             bump_basis_timecourse__rev,
                                             bump_basis_feedback_rev,
                                             bump_basis_coupling__rev,
                                             midget_neighboring_cell_distance),
        'OFF midget': GLMModelHyperparameters(((13, 13), 'cr'),
                                              bump_basis_timecourse__rev,
                                              bump_basis_feedback_rev,
                                              bump_basis_coupling__rev,
                                              midget_neighboring_cell_distance),
    }

    return ret_dict


SUPERVISED_GLM_HYPERPARAMETERS_FN_BY_PIECE = {
    ('2018-08-07-5', 'data000') : {
        1: make_joint_wn_flashed_cropped_basis_2018_08_07_5_hyperparams,
    },
}

