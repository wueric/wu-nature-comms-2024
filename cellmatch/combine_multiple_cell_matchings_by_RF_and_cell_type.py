import numpy as np
import visionloader as vl
import argparse
import pickle
from typing import List, Tuple, Dict

# USES pycpd as a dependency, https://github.com/siavashk/pycpd
# you can install this with "pip install pycpd"
from pycpd import RigidRegistration

from lib.dataset_config_parser.dataset_config_parser import read_config_file, SettingsSection, \
    awsify_piece_name_and_datarun_lookup_key
import lib.data_utils.matched_cells_struct as mcs


def greedy_match_rf_centers(reference_points: np.ndarray,
                            transformed_matched_points: np.ndarray,
                            bailout_distance: float) -> List[Tuple[int, int]]:
    '''

    :param reference_points: shape (M, 2)
    :param transformed_matched_points: shape (N, 2)
    :param bailout_distance:
    :return:
    '''

    # shape (M, 1, 2) - (1, N, 2)
    # -> (M, N, 2) -> (M, N)
    distances = np.linalg.norm(reference_points[:, None, :] - transformed_matched_points[None, :, :],
                              axis=2)

    ref_indices, transform_indices = np.unravel_index(np.argsort(distances, axis=None),
                                                      distances.shape)

    already_matched_ref = set()
    already_matched_transform = set()

    good_matches = [] # type: List[Tuple[int, int]]
    for ref_ix, transform_ix in zip(ref_indices, transform_indices):
        if ref_ix not in already_matched_ref and \
                transform_ix not in already_matched_transform:
            # this is a new point
            dist = distances[ref_ix, transform_ix]
            if dist >= bailout_distance:
                break

            good_matches.append((ref_ix, transform_ix))
            already_matched_ref.add(ref_ix)
            already_matched_transform.add(transform_ix)

    return good_matches


def get_stafit_centers_ordered(vision_dataset: vl.VisionCellDataTable,
                               cell_order: List[int]) -> np.ndarray:
    coordinates = []
    for cell_id in cell_order:
        stafit = vision_dataset.get_stafit_for_cell(cell_id)
        coordinates.append([stafit.center_x, stafit.center_y])
    return np.array(coordinates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Combine multiple matched cell data structures based on reference dataset RFs')
    parser.add_argument('cfg_path_A', type=str, help='path to config file for match A')
    parser.add_argument('cfg_path_B', type=str, help='path to config file for match B')
    parser.add_argument('dist_threshold', type=float,
                        help='distance threshold, above this give up. Defined relative to A')
    parser.add_argument('output_path', type=str, help='path to save pickled combined OrderedMatchedCellsStruct')

    args = parser.parse_args()

    #### Load the first (reference) dataset ##############################
    config_settings_dict_A = read_config_file(args.cfg_path_A)
    ref_lookup_key_A = awsify_piece_name_and_datarun_lookup_key(config_settings_dict_A['ReferenceDataset'].path,
                                                                config_settings_dict_A['ReferenceDataset'].name)

    ref_dataset_info_A = config_settings_dict_A['ReferenceDataset']
    ref_dataset_A = vl.load_vision_data(ref_dataset_info_A.path,
                                        ref_dataset_info_A.name,
                                        include_params=True)
    with open(config_settings_dict_A['responses_ordered'], 'rb') as ordered_cells_file:
        ordered_matches_a = pickle.load(ordered_cells_file)  # type: mcs.OrderedMatchedCellsStruct
    cell_types_ordered = ordered_matches_a.get_cell_types()

    #### Load the second (transform) dataset ##############################
    config_settings_dict_B = read_config_file(args.cfg_path_B)
    ref_lookup_key_B = awsify_piece_name_and_datarun_lookup_key(config_settings_dict_B['ReferenceDataset'].path,
                                                                config_settings_dict_B['ReferenceDataset'].name)

    ref_dataset_info_B = config_settings_dict_B['ReferenceDataset']
    ref_dataset_B = vl.load_vision_data(ref_dataset_info_B.path,
                                        ref_dataset_info_B.name,
                                        include_params=True)
    with open(config_settings_dict_B['responses_ordered'], 'rb') as ordered_cells_file:
        ordered_matches_b = pickle.load(ordered_cells_file)  # type: mcs.OrderedMatchedCellsStruct


    #### Do the matching separately for each cell type ####################
    #### Update the matching data structure accordingly ###################
    supermatch_struct = mcs.OrderedMatchedCellsStruct()
    for cell_type in cell_types_ordered:
        dataset_A_cell_ids = ordered_matches_a.get_reference_cell_order(cell_type)
        dataset_B_cell_ids = ordered_matches_b.get_reference_cell_order(cell_type)

        dataset_A_rf_centers = get_stafit_centers_ordered(ref_dataset_A, dataset_A_cell_ids)
        dataset_B_rf_centers = get_stafit_centers_ordered(ref_dataset_B, dataset_B_cell_ids)

        reg = RigidRegistration(X=dataset_A_rf_centers, Y=dataset_B_rf_centers)
        centers_TB, (s_reg, R_reg, t_reg) = reg.register()

        matched_rf_centers = greedy_match_rf_centers(dataset_A_rf_centers,
                                                     centers_TB,
                                                     args.dist_threshold)

        print(f'Matched {len(matched_rf_centers)} {cell_type} out of ({len(dataset_A_cell_ids)}, {len(dataset_B_cell_ids)})')

        for a_ix, b_ix in matched_rf_centers:
            a_cell_id, b_cell_id = dataset_A_cell_ids[a_ix], dataset_B_cell_ids[b_ix]

            # we are going to have to break the abstraction barrier here
            a_nscenes_matching = ordered_matches_a.main_datadump[cell_type][a_ix] # type: mcs.CellMatch
            b_nscenes_matching = ordered_matches_b.main_datadump[cell_type][b_ix] # type: mcs.CellMatch

            # we will use the WN id from dataset A, the first argument to the script
            combined_matching_dict = {} # type: Dict[str, List[int]]
            for ds_key, nscenes_matched_cells in a_nscenes_matching.matched_nscenes_ids.items():
                combined_matching_dict[ds_key] = nscenes_matched_cells
            for ds_key, nscenes_matched_cells in b_nscenes_matching.matched_nscenes_ids.items():
                combined_matching_dict[ds_key] = nscenes_matched_cells

            supermatch_struct.add_typed_match(cell_type,
                                              a_nscenes_matching.wn_cell_id,
                                              combined_matching_dict)

    with open(args.output_path, 'wb') as pfile:
        pickle.dump(supermatch_struct, pfile)

    print('done')
