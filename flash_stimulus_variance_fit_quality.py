import numpy as np
import pickle

import argparse

import tqdm

from fastconv.corr1d import single_filter_multiple_data_correlate1D

from jitter_stimulus_variance_fit_quality import construct_PSTH_gaussian_kernel


def compute_flashed_repeats_psth(spike_trains: np.ndarray,
                                 psth_kernel: np.ndarray):
    '''
    This function computes the meanm PSTH over repeats of the same stimulus

    For flashed stimulus every trial has exactly the same length, so this is
        comically simple to implement

    :param spike_trains: shape (n_stimuli, n_repeats, n_bins)
    :param psth_kernel: shape (n_bins_kernel, )
    :return: shape (n_stimuli, n_bins - n_bins_kernel + 1)
    '''

    n_repeats, n_stimuli, n_bins = spike_trains.shape
    spike_trains_flat = spike_trains.reshape(-1, n_bins)

    psth_flat = single_filter_multiple_data_correlate1D(spike_trains_flat,
                                                        psth_kernel)

    return np.mean(psth_flat.reshape(n_repeats, n_stimuli, -1), axis=0)


def compute_flashed_frac_variance_explained(simulated_psth: np.ndarray,
                                            ground_truth_psth: np.ndarray) -> float:
    '''

    :param simulated_psth: shape (n_stimuli, n_bins)
    :param ground_truth_psth:  shpae (n_stimuli, n_bins)
    :return:
    '''

    sim_psth_flat = simulated_psth.reshape(-1)
    gt_psth_flat = ground_truth_psth.reshape(-1)

    psth_error = sim_psth_flat - gt_psth_flat
    mean_gt_psth = np.mean(gt_psth_flat)

    ss_tot = np.sum(np.square(gt_psth_flat - mean_gt_psth))
    ss_error = np.sum(np.square(psth_error))

    return 1.0 - (ss_error / ss_tot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Estimate fit quality of flashed GLMs by computing fraction of PSTH variance explained over repeat simulations")
    parser.add_argument('simulated_spike_trains', type=str,
                        help='path to simulation file; should also contain ground truth')
    parser.add_argument('output', type=str, help='path to output file')
    parser.add_argument('-w', '--width', type=float, default=2.0, help='SD of Gaussian kernel for computing PSTH')
    parser.add_argument('-n', '--nsamples', type=int, default=10,
                        help='One side # samples for Gaussian kernel for PSTH')

    args = parser.parse_args()

    with open(args.simulated_spike_trains, 'rb') as pfile:
        sim_contents = pickle.load(pfile)

    psth_kernel = construct_PSTH_gaussian_kernel(args.width, args.nsamples)

    simulated_spike_trains = sim_contents['simulated']
    ground_truth_spike_trains = sim_contents['ground truth']

    pbar = tqdm.tqdm(total=len(simulated_spike_trains))
    var_explained_typed = {}
    for center_cell_id in simulated_spike_trains.keys():

        sim_psth = compute_flashed_repeats_psth(simulated_spike_trains[center_cell_id], psth_kernel)
        gt_psth = compute_flashed_repeats_psth(ground_truth_spike_trains[center_cell_id], psth_kernel)

        var_explained_typed[center_cell_id] = compute_flashed_frac_variance_explained(sim_psth, gt_psth)
        pbar.update(1)
    pbar.close()

    with open(args.output, 'wb') as pfile:
        pickle.dump(var_explained_typed, pfile)
