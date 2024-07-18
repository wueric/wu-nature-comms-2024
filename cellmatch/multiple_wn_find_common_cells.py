import numpy as np
import argparse
import pickle

from typing import Dict, List, Tuple

import visionloader as vl

from cellmatch.combine_multiple_cell_matchings_by_RF_and_cell_type import get_stafit_centers_ordered, \
    greedy_match_rf_centers
from lib.dataset_config_parser.dataset_config_parser import read_config_file, SettingsSection, \
    awsify_piece_name_and_datarun_lookup_key
import lib.data_utils.matched_cells_struct as mcs

from pycpd import RigidRegistration


def construct_wn_to_wn_dictdict(matching: List[Tuple[int, int]],
                                dataset_a: str,
                                dataset_b: str) -> Dict[str, Dict[int, int]]:
    '''
    each entry in matching is assumed to be (dataset_a cell_id, dataset_b cell_id)
    :param matching:
    :param dataset_a:
    :param dataset_b:
    :return:
    '''

    a_to_b, b_to_a = {}, {}
    for a_cell_id, b_cell_id in matching:
        a_to_b[a_cell_id] = b_cell_id
        b_to_a[b_cell_id] = a_cell_id

    return {dataset_a: a_to_b, dataset_b: b_to_a}


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
    common_cells_by_type = {}
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

        matched_cell_ids = [(dataset_A_cell_ids[i], dataset_B_cell_ids[j]) for i, j in matched_rf_centers]

        common_cells_by_type[cell_type] = construct_wn_to_wn_dictdict(
            matched_cell_ids,
            ref_dataset_info_A.name,
            ref_dataset_info_B.name
        )

        print(f'Matched {len(matched_cell_ids)} {cell_type} out of ({len(dataset_A_cell_ids)}, {len(dataset_B_cell_ids)})')

    with open(args.output_path, 'wb') as pfile:
        pickle.dump(common_cells_by_type, pfile)
    print('done')