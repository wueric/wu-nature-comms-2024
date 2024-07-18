import numpy as np
from typing import List, Set, Dict, Tuple
import visionloader as vl
import spikebinning

SAMPLE_RATE = 20e3

#ACFT1 = 0.5e-3  # units seconds
ACFT1 = 0  # units seconds
ACFT2 = 1e-3  # units seconds

ACFT1_SAMPLES = ACFT1 * SAMPLE_RATE
ACFT2_SAMPLES = ACFT2 * SAMPLE_RATE


def count_refractory_violations(spike_times_sorted: np.ndarray) -> int:
    isi = spike_times_sorted[1:] - spike_times_sorted[:-1]

    n_violations = np.sum(np.logical_and.reduce([
        isi >= ACFT1_SAMPLES,
        isi <= ACFT2_SAMPLES
    ]))

    return int(n_violations)


def calculate_contamination(spike_times_sorted: np.ndarray) -> float:
    n_spikes = spike_times_sorted.shape[0]

    if n_spikes == 0:
        return 0.0

    n_refractory_violations = count_refractory_violations(spike_times_sorted)
    approx_duration = spike_times_sorted[-1]

    return (n_refractory_violations * approx_duration) / (n_spikes * n_spikes * (ACFT2_SAMPLES - ACFT1_SAMPLES))


def contamination_search_duplicate_merge(vision_dataset: vl.VisionCellDataTable,
                                         candidate_set: Set[int],
                                         max_refractory_violations: int) -> Tuple[Set[int], Set[int]]:
    # we'll do brute force search, since the size of each merge candidate set is small (~3 entries or so)
    print(candidate_set)
    def bit_mask_to_idx_list(number: int):
        idx_list = []
        shift_amount = 0
        while number != 0:
            if (number & 0x1) == 1:
                idx_list.append(shift_amount)
            shift_amount += 1
            number = (number >> 1)

        return idx_list

    def brute_force_search_contamination(spike_times_by_cell_id: Dict[int, np.ndarray]):
        cell_id_order = np.array(list(spike_times_by_cell_id.keys()))

        n_cells = len(cell_id_order)

        violations_by_cell_id = {cell_id: count_refractory_violations(val) for cell_id, val in \
                                 spike_times_by_cell_id.items()}
        violations_by_idx = np.array([violations_by_cell_id[cell_id] for cell_id in cell_id_order])
        nspikes_by_idx = np.array([spike_times_by_cell_id[cell_id].shape[0] for cell_id in cell_id_order])

        best_idx_list = [0, ]
        max_size = nspikes_by_idx[0]

        for i in range(1, (0b1 << n_cells)):
            idx_list = bit_mask_to_idx_list(i)

            orig_refractory_violations = np.sum(violations_by_idx[idx_list])
            n_spikes_merge = np.sum(nspikes_by_idx[idx_list])

            refractory_violations = count_refractory_violations(
                spikebinning.merge_multiple_sorted_array([spike_times_by_cell_id[cell_id_order[idx]]
                                                          for idx in idx_list]))

            if (refractory_violations - orig_refractory_violations) < max_refractory_violations and \
                    n_spikes_merge > max_size:
                best_idx_list = idx_list
                max_size = n_spikes_merge

        return set(cell_id_order[best_idx_list])

    spike_times = {cell_id: vision_dataset.get_spike_times_for_cell(cell_id) for cell_id in candidate_set}

    best_subset = brute_force_search_contamination(spike_times)

    # dump out the remaining sets, as single element sets
    return best_subset, {x for x in candidate_set if x not in best_subset}
