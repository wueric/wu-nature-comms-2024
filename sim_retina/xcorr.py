import numpy as np
from typing import Tuple, List


def crosscorr_hist(a_spikes_trial: np.ndarray,
                   b_spikes_trial: np.ndarray,
                   window: int = 50):
    '''

    :param a_spikes_trial: shape (batch, n_bins)
    :param b_spikes_trial: shape (batch, n_bins)
    :param window:
    :return:
    '''
    _max_size = 2 * window + 1
    times = np.r_[-window-1:window+1] + 0.5
    hist_count = np.zeros((_max_size,))
    for i in range(a_spikes_trial.shape[0]):
        a_times = np.nonzero(a_spikes_trial[i, :])[0]
        b_times = np.nonzero(b_spikes_trial[i, :])[0]

        diff = (a_times[:, None] - b_times[None, :]).reshape(-1)
        hist_count += np.histogram(diff, bins=times)[0]

    return hist_count / a_spikes_trial.shape[0], times


def crosscorr_hist_with_stim_structure_correction(
        repeat_a_spikes_trial: np.ndarray,
        repeat_b_spikes_trial: np.ndarray,
        window: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    '''
    :param repeat_a_spikes_trial: shape (n_repeats, batch, n_bins)
    :param repeat_b_spikes_trial: shape (n_repeats, batch, n_bins)
    '''

    stimulus_mismatch_sel = np.r_[0:repeat_b_spikes_trial.shape[1]] - 3
    mismatch_stim_b_trials = repeat_b_spikes_trial[:, stimulus_mismatch_sel, :]

    a_trial_flat = repeat_a_spikes_trial.reshape(-1, repeat_a_spikes_trial.shape[-1])
    b_trial_flat = repeat_b_spikes_trial.reshape(-1, repeat_b_spikes_trial.shape[-1])

    mismatch_b_flat = mismatch_stim_b_trials.reshape(-1, mismatch_stim_b_trials.shape[-1])

    raw_cc, times = crosscorr_hist(a_trial_flat,
                                   b_trial_flat,
                                   window=window)

    correction_cc, _ = crosscorr_hist(a_trial_flat,
                                      mismatch_b_flat,
                                      window=window)

    corrected_cc = raw_cc - correction_cc

    return corrected_cc, times


def batched_equal_bin_shuffle_repeats(unshuffled: np.ndarray) -> np.ndarray:
    '''

    :param unshuffled: shape (n_repeats, n_stimuli, ...)
    :return:
    '''

    n_repeats = unshuffled.shape[0]
    shift_selector = np.r_[0:n_repeats] - 3
    return unshuffled[shift_selector, ...]


def unequal_bin_xcorr_hist(
        a_spikes_trial: List[np.ndarray],
        b_spikes_trial: List[np.ndarray],
        window: int = 50):
    '''

    :param a_spikes_trial: List of shape (n_bins, ), where each
        entry might have a different value for n_bins

        Each entry of the list corresponds to a different stimulus
            presentation or repeat presentation
    :param b_spikes_trial: List of shape (n_bins, ), where each
        entry might have a different value for n_bins

        Each entry of the list corresponds to a different stimulus
            presentation or repeat presentation
    :param window:
    :return:
    '''
    _max_size = 2 * window + 1
    times = np.r_[-window-1:window+1] + 0.5
    hist_count = np.zeros((_max_size,))
    batch = len(a_spikes_trial)
    for i in range(batch):
        a_times = np.nonzero(a_spikes_trial[i])[0]
        b_times = np.nonzero(b_spikes_trial[i])[0]

        diff = (a_times[:, None] - b_times[None, :]).reshape(-1)

        hist_count += np.histogram(diff, bins=times)[0]

    return hist_count / batch, times


def unequal_bin_flatten_stim_repeat_lists(
        batched_repeat_spikes: List[List[np.ndarray]]) \
        -> List[np.ndarray]:
    ret_list = []
    for list_array in batched_repeat_spikes:
        ret_list.extend(list_array)
    return ret_list


def unequal_bin_xcorr_hist_with_stim_structure_correation(
        repeat_a_spikes_trial: List[List[np.ndarray]],
        repeat_b_spikes_trial: List[List[np.ndarray]],
        window: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    '''

    This function shuffles between the stimuli to compute the
        stimulus structure correction

    This function DOES NOT shuffle over repeats of the same stimulus;
        that is the caller's responsibility to do

    :param repeat_a_spikes_trial: List[List[np.ndarray]], each np.ndarray
        has shape (n_bins, ) where n_bins may be slightly different
        for each entry

        Outer list is over presentation of different stimuli

        Inner list is over presentation of the same stimulus
            but different repeats

    :param repeat_b_spikes_trial: List[List[np.ndarray]], each np.ndarray
        has shape (n_bins, ) where n_bins may be slightly different
        for each entry

        Outer list is over presentation of different stimuli

        Inner list is over presentation of the same stimulus
            but different repeats
    :param window:
    :return:
    '''
    n_stimuli = len(repeat_a_spikes_trial)
    n_repeats = len(repeat_a_spikes_trial[0])

    stimulus_mismatch_sel = list(np.r_[0:n_stimuli] - 5)

    # shuffle cell b trials using stimulus_mismatch_sel
    repeat_b_shuffled_by_stimulus = [
        repeat_b_spikes_trial[ix] for ix in stimulus_mismatch_sel
    ]

    a_spikes_trial_batch_flat = unequal_bin_flatten_stim_repeat_lists(
        repeat_a_spikes_trial
    )

    b_spikes_trial_batch_flat = unequal_bin_flatten_stim_repeat_lists(
        repeat_b_spikes_trial
    )

    stim_shuffled_b_spikes_trial_batch_flat = unequal_bin_flatten_stim_repeat_lists(
        repeat_b_shuffled_by_stimulus
    )

    uncorrected, times = unequal_bin_xcorr_hist(
        a_spikes_trial_batch_flat,
        b_spikes_trial_batch_flat,
        window=window
    )

    correction, _ = unequal_bin_xcorr_hist(
        a_spikes_trial_batch_flat,
        stim_shuffled_b_spikes_trial_batch_flat,
        window=window
    )

    return uncorrected - correction, times


def unequal_bin_shuffle_repeats(unshuffled: List[List[np.ndarray]]) \
        -> List[List[np.ndarray]]:
    '''
    Shuffles between the repeats (the inner list in the nested list)
    :param unshuffled:
    :return:
    '''

    n_stimuli = len(unshuffled)
    n_repeats = len(unshuffled[0])

    shuffled_ix_sel = list(np.r_[0:n_repeats] - 3)

    shuffled_output = [] # type: List[List[np.ndarray]]
    for stim_ix in range(n_stimuli):

        spikes_for_stim = unshuffled[stim_ix]
        shuffled_for_stim = [spikes_for_stim[rep_ix] for rep_ix in shuffled_ix_sel]
        shuffled_output.append(shuffled_for_stim)

    return shuffled_output
