import numpy as np
from numpy import linalg

import visionloader as vl

from typing import List, Tuple, Dict, Sequence, Optional, Set
import functools


def get_minmax_significant_ei_vectors (dataset : vl.VisionCellDataTable,
                                       good_cell_list : Sequence[int],
                                       electrode_threshold : float,
                                       significant_electrodes: int) -> Tuple[np.ndarray, np.ndarray]:
    above_threshold_ref_ei_vectors = []  # type: List[np.ndarray]
    above_threshold_ref_cell_ids = []  # type: List[int]
    for cell_id in good_cell_list:
        ei_container = dataset.get_ei_for_cell(cell_id)
        ei_matrix = ei_container.ei

        amin_ei = np.amin(ei_matrix, axis=1)
        amax_ei = np.amax(ei_matrix, axis=1)
        abs_ei = np.max(np.abs(ei_matrix), axis=1)

        ei_minmax = np.concatenate([amin_ei, amax_ei], axis=0)

        exceeds_threshold = (abs_ei > electrode_threshold)

        if np.sum(exceeds_threshold) > significant_electrodes:
            above_threshold_ref_ei_vectors.append(ei_minmax)
            above_threshold_ref_cell_ids.append(cell_id)

    return np.array(above_threshold_ref_cell_ids), np.array(above_threshold_ref_ei_vectors)


def get_significant_ei_vectors(dataset: vl.VisionCellDataTable,
                               good_cell_list: Sequence[int],
                               electrode_threshold: float,
                               significant_electrodes: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get feature vectors of shape (n_electrodes, ) for a list of EIs

    Entry for each electrode corresponds to extremum (either maximum or minimum of the waveform,
        depending on whether maximum or minimum has the largest absolute deviation from zero)
    :param dataset: vl.VisionCellDataTable for dataset
    :param good_cell_list: Sequence of good cell ids
    :param electrode_threshold: threshold in order for an electrode to be considered significant
    :param significant_electrodes: minimum number of electrodes that have abs amplitude greater than electrode_threshold
            if a cell does not meet this criterion, we discard the cell outright
    :return: Tuple[np.array, np.array], first is list of good cell ids, shape (n_good_cells, )
            second is featurized EIs of shape (n_good_cells, n_electrodes)
    '''
    above_threshold_ref_ei_vectors = []  # type: List[np.ndarray]
    above_threshold_ref_cell_ids = []  # type: List[int]
    for cell_id in good_cell_list:
        ei_container = dataset.get_ei_for_cell(cell_id)
        ei_matrix = ei_container.ei

        amin_ei = np.amin(ei_matrix, axis=1)
        amax_ei = np.amax(ei_matrix, axis=1)
        abs_ei = np.max(np.abs(ei_matrix), axis=1)

        use_max = (abs_ei >= amax_ei)
        amin_ei[use_max] = amax_ei[use_max]

        ei_min_space_only = amin_ei

        exceeds_threshold = (abs_ei > electrode_threshold)

        if not np.any(np.isnan(ei_min_space_only)) and np.sum(exceeds_threshold) > significant_electrodes:
            above_threshold_ref_ei_vectors.append(ei_min_space_only)
            above_threshold_ref_cell_ids.append(cell_id)

    return np.array(above_threshold_ref_cell_ids), np.array(above_threshold_ref_ei_vectors)


def get_significant_ei_spacetime(dataset: vl.VisionCellDataTable,
                                 good_cell_list: Sequence[int],
                                 electrode_threshold: float,
                                 significant_electrodes: int) -> Tuple[np.ndarray, np.ndarray]:
    above_threshold_ref_ei_spacetime = []  # type: List[np.ndarray]
    above_threshold_ref_cell_ids = []  # type: List[int]
    for cell_id in good_cell_list:
        ei_container = dataset.get_ei_for_cell(cell_id)
        ei_matrix = ei_container.ei
        ei_space_only = np.min(ei_matrix, axis=1)

        exceeds_threshold = (ei_space_only >= np.abs(electrode_threshold))
        if np.sum(exceeds_threshold) > significant_electrodes:
            above_threshold_ref_ei_spacetime.append(ei_matrix)
            above_threshold_ref_cell_ids.append(cell_id)

    return np.array(above_threshold_ref_cell_ids), np.array(above_threshold_ref_ei_spacetime)


def cosine_similarity_find_hard_pairs_between_lists(reference_dataset: vl.VisionCellDataTable,
                                                    list_a_cell_ids: List[int],
                                                    list_b_cell_ids: List[int],
                                                    significant_electrodes: int = 10,
                                                    most_confusing_fraction: float = 0.10) -> List[Tuple[int, int]]:
    above_threshold_list_a_ids, above_threshold_list_a_vectors = get_significant_ei_vectors(
        reference_dataset,
        list_a_cell_ids,
        2.0,
        significant_electrodes
    )

    above_threshold_list_b_ids, above_threshold_list_b_vectors = get_significant_ei_vectors(
        reference_dataset,
        list_b_cell_ids,
        2.0,
        significant_electrodes
    )

    cosine_similarity_a_to_b = cosine_similarity(above_threshold_list_a_vectors, above_threshold_list_b_vectors)

    most_confused_ordering = np.argsort(cosine_similarity_a_to_b, axis=None)
    n_examples_to_take = int(most_confused_ordering.shape[0] * most_confusing_fraction)

    idx_list_a, idx_list_b = np.unravel_index(most_confused_ordering[-n_examples_to_take:],
                                              cosine_similarity_a_to_b.shape)

    confusing_pairs = []
    for idx_a, idx_b in zip(idx_list_a, idx_list_b):
        confusing_pairs.append((above_threshold_list_a_ids[idx_a], above_threshold_list_b_ids[idx_b]))

    return confusing_pairs


def cosine_similarity(vec_a: np.ndarray,
                      vec_b: np.ndarray,
                      axis: int = 1,
                      epsilon: float = 1e-8) -> np.ndarray:
    '''

    :param vec_a: np.ndarray, shape (n_cells_a, n_features)
    :param vec_b:  np.ndarray, shape (n_cells_b, n_features)
    :param axis:
    :param epsilon:
    :return: np.ndarray, shape (n_cells_a, n_cells_b)
    '''

    a_mag = np.linalg.norm(vec_a, axis=axis)  # shape (n_cells_a, )
    b_mag = np.linalg.norm(vec_b, axis=axis)  # shape (n_cells_b, )

    numerator = vec_a @ vec_b.T
    denominator = (a_mag[:, None] * b_mag[None, :] + epsilon)

    return numerator / denominator


@functools.total_ordering
class MatchPair:

    def __init__(self, orig_id: int, map_id: int, score: float):
        self.orig_id = orig_id
        self.map_id = map_id
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return 'MatchPair({0}, {1}, {2})'.format(self.orig_id, self.map_id, self.score)


def assign_match_until_threshold(good_cell_id_list: Sequence[int],
                                 other_cell_id_list: Sequence[int],
                                 similarity_matrix: np.ndarray,
                                 give_up_thresh: float = 0.95) -> Tuple[Dict[int, List[MatchPair]], List[int]]:
    # peel off the best matches, until we either reach give_up_thresh
    # or everything has a match
    match_dict = {}

    previously_matched_other = set()

    x_idx_sorted, y_idx_sorted = np.unravel_index(np.argsort((-1 * similarity_matrix).ravel()), similarity_matrix.shape)
    for x_idx, y_idx in zip(x_idx_sorted, y_idx_sorted):

        similarity_score = similarity_matrix[x_idx, y_idx]
        if similarity_score < give_up_thresh:
            break

        orig_cell = good_cell_id_list[x_idx]
        matched_cell = other_cell_id_list[y_idx]

        if matched_cell not in previously_matched_other:
            if orig_cell not in match_dict:
                match_dict[orig_cell] = []
            match_dict[orig_cell].append(MatchPair(orig_cell, other_cell_id_list[y_idx], similarity_score))
            previously_matched_other.add(matched_cell)

    return match_dict, [x for x in good_cell_id_list if x not in match_dict]