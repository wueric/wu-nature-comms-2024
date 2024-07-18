import argparse
import pickle
from typing import List
import os

import visionloader as vl

import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct, RFCenterStruct
from lib.data_utils.sta_metadata import calculate_center_from_sta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts a subset of matched cells')
    parser.add_argument('orig_cfg_path', type=str, help='path to config file')
    parser.add_argument('reduced_cfg_path', type=str, help='path to reduced cell type cfg path')

    args = parser.parse_args()

    full_config_settings_dict = dcp.read_config_file(args.orig_cfg_path)
    reduced_config_settings_dict = dcp.read_config_file(args.reduced_cfg_path)

    ref_lookup_key = dcp.awsify_piece_name_and_datarun_lookup_key(reduced_config_settings_dict['ReferenceDataset'].path,
                                                              reduced_config_settings_dict['ReferenceDataset'].name)

    ref_dataset_info = reduced_config_settings_dict['ReferenceDataset']

    ref_dataset = vl.load_vision_data(ref_dataset_info.path,
                                      ref_dataset_info.name,
                                      include_params=True,
                                      include_sta=True)

    if ref_dataset_info.classification_file is not None:
        ref_dataset.update_cell_type_classifications_from_text_file(ref_dataset_info.classification_file)


    ################################################################
    # figure out what the available nscenes datasets were
    available_nscenes_keys = []
    if dcp.NScenesMovieDatasetSection.OUTPUT_KEY in full_config_settings_dict:
        for nscenes_dataset_info in full_config_settings_dict[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]:
            available_nscenes_keys.append(nscenes_dataset_info.name)
    else:
        for nscenes_dataset_info in full_config_settings_dict[dcp.NScenesFlashedDatasetSection.OUTPUT_KEY]:
            available_nscenes_keys.append(nscenes_dataset_info.name)

    ################################################################
    # Load the cell types and matching
    with open(full_config_settings_dict['responses_ordered'], 'rb') as ordered_cells_file:
        full_cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct

    reduced_cell_types = reduced_config_settings_dict['CellTypes']  # type: List[str]

    reduced_cells_ordered = OrderedMatchedCellsStruct()
    for cell_type in reduced_cell_types:

        full_matched_ids = full_cells_ordered.get_reference_cell_order(cell_type)
        for full_matched_id in full_matched_ids:

            ns_matched_dict = {
                nscenes_key: full_cells_ordered.get_match_ids_for_ds(full_matched_id, nscenes_key)
                for nscenes_key in available_nscenes_keys
            }

            reduced_cells_ordered.add_typed_match(cell_type, full_matched_id, ns_matched_dict)

    # calculate RF center coordinates for matched cells only
    full_cells_by_type = {cell_type: full_cells_ordered.get_reference_cell_order(cell_type) \
                               for cell_type in full_cells_ordered.get_cell_types()}
    rf_centers_matched_only = calculate_center_from_sta(ref_dataset,
                                                        full_cells_by_type,
                                                        sig_stixel_threshold=4.0)
    rf_center_struct = RFCenterStruct(rf_centers_matched_only)

    ######## SAVE THE OUTPUTS TO THE SPECIFIED LOCATIONS USING PICKLE #############

    # make the folders if necessary
    responses_folder = os.path.dirname(reduced_config_settings_dict['responses_ordered'])
    os.makedirs(responses_folder, exist_ok=True)
    with open(reduced_config_settings_dict['responses_ordered'], 'wb') as picklefile:
        pickle.dump(reduced_cells_ordered, picklefile)
        pickle.dump(rf_center_struct, picklefile)
