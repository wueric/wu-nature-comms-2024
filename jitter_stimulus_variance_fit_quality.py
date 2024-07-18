import numpy as np
import pickle

import argparse
from typing import List, Dict

import tqdm


def construct_PSTH_gaussian_kernel(width: float,
                                   half_samples: float) -> np.ndarray:
    kernel_time = np.r_[-half_samples:half_samples + 1]
    psth_kernel = np.exp(-0.5 * np.square(kernel_time / width))
    return psth_kernel / np.sum(psth_kernel)


def compute_repeats_psth(spike_trains: List[List[np.ndarray]],
                         psth_kernel: np.ndarray) -> List[np.ndarray]:
    '''
    This function computes the mean PSTH over the repeats of the same stimulus

    Implementation here is more subtle than would be expected, since each repeat
        of the same stimulus may be longer or shorter than the others in terms of
        number of electrical samples

    :param spike_trains:
    :param psth_kernel:
    :return:
    '''

    mean_psth_each_stim = []
    for repeat_list in spike_trains:
        min_len = float('inf')
        conv_temp = []
        for repeat_spike_train in repeat_list:
            blurred = np.convolve(repeat_spike_train, psth_kernel)
            conv_temp.append(blurred)
            min_len = min(min_len, blurred.shape[0])

        mean_psth = np.mean(np.stack([x[:min_len] for x in conv_temp], axis=1), axis=1)
        mean_psth_each_stim.append(mean_psth)
    return mean_psth_each_stim


def compute_frac_variance_explained(simulated_psth: List[np.ndarray],
                                    ground_truth_psth: List[np.ndarray]) -> float:
    '''
    Take the mean over repeats
    :param simulated_psth:
    :param ground_truth_psth:
    :return:
    '''

    sim_concat_psth = np.concatenate(simulated_psth, axis=0)
    gt_concat_psth = np.concatenate(ground_truth_psth, axis=0)

    psth_error = gt_concat_psth - sim_concat_psth
    mean_gt_psth = np.mean(gt_concat_psth)

    ss_tot = np.sum(np.square(gt_concat_psth - mean_gt_psth))
    ss_error = np.sum(np.square(psth_error))

    return 1.0 - (ss_error / ss_tot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Estimate fit quality of jitter GLMs by computing fraction of PSTH variance explained over repeat simulations")
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

    total_variance_explained = {} # type: Dict[str, Dict[int, float]]
    for center_cell_type in simulated_spike_trains.keys():

        print(f"Evaluating fit for {center_cell_type}")

        sim_cells_typed_dict = simulated_spike_trains[center_cell_type]
        gt_cells_typed_dict = ground_truth_spike_trains[center_cell_type]

        pbar = tqdm.tqdm(total=len(sim_cells_typed_dict))
        var_explained_typed = {}
        for center_cell_id in sim_cells_typed_dict.keys():

            sim_psth = compute_repeats_psth(sim_cells_typed_dict[center_cell_id], psth_kernel)
            gt_psth = compute_repeats_psth(gt_cells_typed_dict[center_cell_id], psth_kernel)

            var_explained_typed[center_cell_id] = compute_frac_variance_explained(sim_psth, gt_psth)
            pbar.update(1)
        pbar.close()

        total_variance_explained[center_cell_type] = var_explained_typed

    with open(args.output, 'wb') as pfile:
        pickle.dump(total_variance_explained, pfile)
