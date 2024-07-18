import numpy as np

import visionloader as vl

from typing import Tuple, List, Dict, Union, Set, Any

from collections import Counter

from lib.data_utils.contamination import contamination_search_duplicate_merge
from lib.data_utils.cosine_similarity import MatchPair
import lib.data_utils.sta_metadata as sta_metadata


def median_nnd_cell(vision_dset: vl.VisionCellDataTable,
                    good_cell_id_list: List[int]) -> float:
    nnd_list = []

    for i, cell_id_a in enumerate(good_cell_id_list):

        stafit_a = vision_dset.get_stafit_for_cell(cell_id_a)
        center_a = np.array([stafit_a.center_x, stafit_a.center_y])

        nnd = np.inf

        for cell_id_b in good_cell_id_list:

            if cell_id_b != cell_id_a:
                stafit_b = vision_dset.get_stafit_for_cell(cell_id_b)
                center_b = np.array([stafit_b.center_x, stafit_b.center_y])

                dist = np.linalg.norm(center_a - center_b)
                nnd = min(nnd, dist)

        nnd_list.append(nnd)

    return np.median(nnd_list)


def get_neighbors_within_list(vision_dataset: vl.VisionCellDataTable,
                              good_cells_list: List[int],
                              neighbor_max_dist: float) -> List[Tuple[int, int]]:
    '''

    :param vision_dataset:
    :param good_cells_list:
    :param neighbor_max_dist:
    :return: List of tuple of cell id
    '''

    neighbors = []  # type: List[Tuple[int, int]]
    for i, cell_id in enumerate(good_cells_list):

        stafit = vision_dataset.get_stafit_for_cell(cell_id)
        center = np.array([stafit.center_x, stafit.center_y])

        for other_cell_id in good_cells_list[i + 1:]:

            other_stafit = vision_dataset.get_stafit_for_cell(other_cell_id)
            other_center = np.array([other_stafit.center_x, other_stafit.center_y])

            if np.linalg.norm(center - other_center) < neighbor_max_dist:
                neighbors.append((cell_id, other_cell_id))

    return neighbors


def get_neighbors_between_lists(vision_dataset: vl.VisionCellDataTable,
                                cell_list_a: List[int],
                                cell_list_b: List[int],
                                neighbor_max_dist: float) -> List[Tuple[int, int]]:
    neighbors = []
    for cell_id_a in cell_list_a:

        stafit = vision_dataset.get_stafit_for_cell(cell_id_a)
        center = np.array([stafit.center_x, stafit.center_y])

        for cell_id_b in cell_list_b:

            other_stafit = vision_dataset.get_stafit_for_cell(cell_id_b)
            other_center = np.array([other_stafit.center_x, other_stafit.center_y])

            if np.linalg.norm(center - other_center) < neighbor_max_dist and cell_id_a != cell_id_b:
                neighbors.append((cell_id_a, cell_id_b))

    return neighbors


def calculate_mosaic_interaction_info_from_neighbor_pairs(vision_dataset: vl.VisionCellDataTable,
                                                          first_cell_type: str,
                                                          second_cell_type: str,
                                                          neighbors_list: List[Tuple[int, int]]) \
        -> List[Dict[str, float]]:
    '''
    It may be possible to parameterize the strength of the interaction between cells of known types
        using only the distance between the cells in the mosaic, the degree of overlap in their receptive
        fields, and some other stuff. This computes those parameters.

    Keys and values:
        * 'distance' -> float, distance between the RF centers, using the same units that Vision uses
        * 'inner_prod' -> float, absolute value of the (normalized) inner product between the sig stixels
            of the receptive fields. Requires computing sig stixels + timecourses for every cell in question

    :param vision_dataset:
    :param first_cell_type:
    :param second_cell_type:
    :param neighbors_list:
    :return:
    '''

    #### Initialize the data structure of quantities that we want to return ###########################
    distance_overlap_parameters = [{} for _ in range(len(neighbors_list))]  # type: List[Dict[str, float]]

    #### Compute the Vision distances between each cell ###############################################
    for i, (cell_id_first, cell_id_second) in enumerate(neighbors_list):
        sta_fit_first, sta_fit_second = vision_dataset.get_stafit_for_cell(
            cell_id_first), vision_dataset.get_stafit_for_cell(cell_id_second)

        first_coord = np.array([sta_fit_first.center_x, sta_fit_first.center_y])
        second_coord = np.array([sta_fit_second.center_x, sta_fit_second.center_y])

        distance = np.linalg.norm(first_coord - second_coord)
        distance_overlap_parameters[i]['distance'] = distance

    #### Compute STA inner products ###################################################################
    # Load the sig stixel STAs, normalize so everything has L2 norm = 1
    first_cell_type_all = vision_dataset.get_all_cells_of_type(first_cell_type)
    first_id_to_position = {cell_id: idx for idx, cell_id in enumerate(first_cell_type_all)}  # type: Dict[int, int]

    # shape (n_cells_first_type, width, height)
    sig_stixels_first_type = sta_metadata.load_sigstixels_spatial_only_stas(vision_dataset,
                                                                            first_cell_type_all)
    sig_stixels_first_type_flat = sig_stixels_first_type.reshape((sig_stixels_first_type.shape[0], -1))
    sig_stixels_first_type_flat = sig_stixels_first_type_flat / np.linalg.norm(sig_stixels_first_type_flat, axis=1, keepdims=True)

    if first_cell_type != second_cell_type:
        second_cell_type_all = vision_dataset.get_all_cells_of_type(second_cell_type)
        second_id_to_position = {cell_id: idx for idx, cell_id in
                                 enumerate(second_cell_type_all)}  # type: Dict[int, int]
        # shape (n_cells_second_type, width, height)
        sig_stixels_second_type = sta_metadata.load_sigstixels_spatial_only_stas(vision_dataset,
                                                                                 second_cell_type_all)
        sig_stixels_second_type_flat = sig_stixels_second_type.reshape((sig_stixels_second_type.shape[0], -1))
        sig_stixels_second_type_flat = sig_stixels_second_type_flat / np.linalg.norm(sig_stixels_second_type_flat, axis=1, keepdims=True)
    else:
        second_id_to_position = first_id_to_position
        sig_stixels_second_type_flat = sig_stixels_first_type_flat

    # translate the neighbors list into indices, so we can select STAs and do the inner product in batch form
    neighbors_sel_indices = [(first_id_to_position[id_a], second_id_to_position[id_b]) for id_a, id_b in neighbors_list]

    # now compute inner products in batch
    neighbors_sel_as_array = np.array(neighbors_sel_indices, dtype=np.int32)
    sig_stixels_first_sel = sig_stixels_first_type_flat[neighbors_sel_as_array[:, 0], ...]
    sig_stixels_second_sel = sig_stixels_second_type_flat[neighbors_sel_as_array[:, 1], ...]

    # shape (n_interactions, )
    inner_product = np.sum(sig_stixels_first_sel * sig_stixels_second_sel, axis=1)

    for i, inner_prod_value in enumerate(inner_product):
        distance_overlap_parameters[i]['inner_prod'] = inner_product[i]

    return distance_overlap_parameters


def get_neighbors_of_center_cells(vision_dataset: vl.VisionCellDataTable,
                                  center_cell_list: List[int],
                                  interacting_cell_list: List[int],
                                  neighbor_max_distance: float) -> List[Tuple[int, List[int]]]:
    output_list = []  # type: List[Tuple[int, List[int]]]

    for center_cell_id in center_cell_list:

        stafit = vision_dataset.get_stafit_for_cell(center_cell_id)
        center_coord = np.array([stafit.center_x, stafit.center_y])

        neighbors_of_center = []  # type: List[int]

        for possible_neighbor_id in interacting_cell_list:

            neighbor_stafit = vision_dataset.get_stafit_for_cell(possible_neighbor_id)
            neighbor_center_coord = np.array([neighbor_stafit.center_x, neighbor_stafit.center_y])

            if np.linalg.norm(center_coord - neighbor_center_coord) < neighbor_max_distance:
                neighbors_of_center.append(possible_neighbor_id)

        output_list.append((center_cell_id, neighbors_of_center))

    return output_list


def cell_id_to_idx_mapping(good_cell_id_list: List[int]) \
        -> Dict[int, int]:
    cell_id_to_idx = {}

    for idx, cell_id in enumerate(good_cell_id_list):
        cell_id_to_idx[cell_id] = idx

    return cell_id_to_idx


def map_ei_dataset_pairs(reference_dataset: vl.VisionCellDataTable,
                         good_cell_ids: List[int],
                         second_dataset: vl.VisionCellDataTable,
                         electrode_threshold: float = -5.0,
                         significant_electrodes: int = 10,
                         corr_threshold: float = 0.95,
                         space_only: bool = True) -> Tuple[Dict[int, List[int]], List[int]]:
    '''
    EI-based mapping method for matching cell id in a reference dataset to
        cell id in a second dataset

    Performs one-to-many mapping, i.e. the set of cell ids that we care about in
        the reference dataset should not contain any duplicate cells, but may be mapped
        to multiple cell id in the second dataset

    basic algorithm:

    Find electrodes that are larger amplitude than the threshold
    Only include cells that have more than the minimum number of electrodes that exceed the
        threshold. This is a filter to remove cells that don't appear with large amplitude
        on many electrodes

    Then flatten the EI and put into matrix, do this for cells from both datasets. When doing
        this we include all of the electrodes, whether or not they exceed threshold

    Calculate correlation between the two matrices

    Followed by some sort of mapping algorithm between correlation scores...
        (probably peel off in order of highest confidence)


    '''

    second_dataset_all_cell_ids = second_dataset.get_cell_ids()

    if space_only:

        # find the relevant EIs for the reference dataset
        above_threshold_ref_ei_vectors = []  # type: List[np.ndarray]
        above_threshold_ref_cell_ids = []  # type: List[int]
        for cell_id in good_cell_ids:
            ei_container = reference_dataset.get_ei_for_cell(cell_id)
            ei_matrix = ei_container.ei
            ei_space_only = np.amin(ei_matrix, axis=1)

            exceeds_threshold = (ei_space_only < electrode_threshold)
            if np.sum(exceeds_threshold) > significant_electrodes:
                above_threshold_ref_ei_vectors.append(ei_space_only)
                above_threshold_ref_cell_ids.append(cell_id)

        # find the relevant EIs for every cell in the second dataset
        above_threshold_second_ei_vectors = []  # type: List[np.ndarray]
        above_threshold_second_cell_ids = []  # type: List[int]

        for cell_id in second_dataset_all_cell_ids:
            ei_container = second_dataset.get_ei_for_cell(cell_id)
            ei_matrix = ei_container.ei
            ei_space_only = np.amin(ei_matrix, axis=1)

            exceeds_threshold = (ei_space_only < electrode_threshold)
            if np.sum(exceeds_threshold) > significant_electrodes:
                above_threshold_second_ei_vectors.append(ei_space_only)
                above_threshold_second_cell_ids.append(cell_id)

        above_threshold_ref_ei_matrix = np.array(above_threshold_ref_ei_vectors)
        above_threshold_second_ei_matrix = np.array(above_threshold_second_ei_vectors)

        _, n_electrodes = above_threshold_ref_ei_matrix.shape

        # above_threshold_ref_ei_matrix has shape (n_cells_ref, n_electrodes)
        # above_threshold_second_ei_matrix has shape (n_cells_second, n_electrodes)

        # calculate a correlation matrix
        ref_mean_sub = above_threshold_ref_ei_matrix - np.mean(above_threshold_ref_ei_matrix, axis=1, keepdims=True)
        second_mean_sub = above_threshold_second_ei_matrix - np.mean(above_threshold_second_ei_matrix, axis=1,
                                                                     keepdims=True)

        std_ref = np.std(above_threshold_ref_ei_matrix, axis=1)
        std_second = np.std(above_threshold_second_ei_matrix, axis=1)

        std_matrix = std_ref[:, np.newaxis] @ std_second[np.newaxis, :]

        similarity_matrix = (ref_mean_sub @ second_mean_sub.T) / (std_matrix * n_electrodes)

        print(similarity_matrix.shape)

        # similarity_matrix has shape (n_ref_cells, n_second_dset_cells)

        # now greedy algorithm to pull off most similar EIs
        # we do one-to-many, so the naive algorithm is that for each of the reference cells,
        # we match the reference cell with any second dataset cell with >= corr_threshold
        # correlation coefficient

        mapping_dict = {}  # type: Dict[int, List[int]]
        unmatched_reference_cells = []  # type: List[int]

        second_dataset_all_cell_ids_aa = np.array(above_threshold_second_cell_ids)
        for i, cell_id in enumerate(above_threshold_ref_cell_ids):
            good_matches = (similarity_matrix[i, :] >= corr_threshold)
            matches = list(second_dataset_all_cell_ids_aa[good_matches])

            if len(matches) > 0:
                mapping_dict[cell_id] = matches
            else:
                unmatched_reference_cells.append(cell_id)

        return mapping_dict, unmatched_reference_cells


def merge_matches(keyed_match: Dict[str, Dict[int, Any]]) -> Dict[int, Dict[str, Any]]:
    '''
    Merge multiple one-to-one wn->nscenes mappings such that we only include
        cells that are found in all of the datasets

    That way we can combine distinct runs of data
    
    :param args: Dict[int, int], key is reference wn cell id, val is nscenes cell id
    :return: Dict[int, List[int]], key is reference wn cell id, val is series of 
        nscenes cell id, in the same order as the arguments
    '''

    n_dsets_to_merge = len(keyed_match)

    cell_id_counter = Counter()
    for ds_name, mapping in keyed_match.items():
        for key in mapping:
            cell_id_counter[key] += 1

    ref_cell_ids_to_merge = [key for key, val in cell_id_counter.items() if val == n_dsets_to_merge]

    output_dict = {}
    for cell_id in ref_cell_ids_to_merge:
        output_dict[cell_id] = {key: mapping_dict[cell_id] for key, mapping_dict in
                                keyed_match.items()}

    return output_dict


def merge_duplicate_matches_by_spike_count(second_dataset: vl.VisionCellDataTable,
                                           one_to_many_mapping: Dict[int, List[MatchPair]],
                                           max_additional_refractory_violations: int = 0) -> Dict[int, List[int]]:
    merged_mapping = {}
    for ref_cell_id, matched_ids in one_to_many_mapping.items():
        merge_candidate_set = {x.map_id for x in matched_ids}
        best_merge, _ = contamination_search_duplicate_merge(second_dataset,
                                                             merge_candidate_set,
                                                             max_additional_refractory_violations)
        merged_mapping[ref_cell_id] = [x for x in best_merge]
    return merged_mapping


def parse_manual_matching(manual_matching: str) \
        -> Tuple[Dict[int, Dict[str, List[int]]],
                 Dict[str, Set[int]]]:

    manual_matching_lines = manual_matching.split('\n')

    wn_ns_match_ret_dict = {} # type: Dict[int, Dict[str, List[int]]]
    ns_taken_ret_dict = {} # type: Dict[str, Set[int]]
    for line in manual_matching_lines:

        if line.startswith('#'):
            continue

        ref_id, remaining = line.split(':')
        ref_id = int(ref_id)

        wn_ns_match_ret_dict[ref_id] = {}

        per_dataset_match_strings = remaining.split(';')
        for per_dataset_str in per_dataset_match_strings:
            stuff = per_dataset_str.split(',')
            ds_name = str(stuff[0])
            match_ids = [int(x) for x in stuff[1:]]

            wn_ns_match_ret_dict[ref_id][ds_name] = match_ids

            if ds_name not in ns_taken_ret_dict:
                ns_taken_ret_dict[ds_name] = set()
            for match_id in match_ids:
                ns_taken_ret_dict[ds_name].add(match_id)

    return wn_ns_match_ret_dict, ns_taken_ret_dict
