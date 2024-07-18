import numpy as np
import visionloader as vl
import argparse
import pickle
import os

import lib.data_utils.matched_cells_struct as mcs

import lib.data_utils.cell_curation as curate
import lib.data_utils.cosine_similarity as cossim
import lib.data_utils.sta_metadata
from lib.dataset_config_parser.dataset_config_parser import read_config_file, SettingsSection, \
    awsify_piece_name_and_datarun_lookup_key
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.sta_metadata import calculate_center_from_sta

from typing import List, Dict, Tuple

from lib.dataset_specific_init_cell_match.dataset_specific_cell_matches import MANUAL_CELL_MATCHES
from lib.dataset_specific_init_cell_match.brownian_cell_matches import BROWNIAN_CELL_MATCHES

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Match cells between WN and nscenes with cosine similarity')
    parser.add_argument('cfg_path', type=str, help='path to config file')
    parser.add_argument('-m', '--manual_init', action='store_true', help='use optional built-in manual matches',
                        default=False)
    parser.add_argument('-b', '--brownian', action='store_true',
                        help='Use Brownian jitter cell matches instead of flashed', default=False)
    args = parser.parse_args()

    config_settings_dict = read_config_file(args.cfg_path)

    ref_lookup_key = awsify_piece_name_and_datarun_lookup_key(config_settings_dict['ReferenceDataset'].path,
                                                              config_settings_dict['ReferenceDataset'].name)

    cell_types = config_settings_dict['CellTypes']  # type: List[str]

    ref_dataset_info = config_settings_dict['ReferenceDataset']

    ref_dataset = vl.load_vision_data(ref_dataset_info.path,
                                      ref_dataset_info.name,
                                      include_neurons=True,
                                      include_params=True,
                                      include_ei=True,
                                      include_sta=True)
    if ref_dataset_info.classification_file is not None:
        ref_dataset.update_cell_type_classifications_from_text_file(ref_dataset_info.classification_file)

    keyed_nscenes_datasets = {}  # type: Dict[str, vl.VisionCellDataTable]

    is_jittered_movie = args.brownian

    if is_jittered_movie:
        for nscenes_dataset_info in config_settings_dict[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]:
            keyed_nscenes_datasets[nscenes_dataset_info.name] = vl.load_vision_data(nscenes_dataset_info.path,
                                                                                    nscenes_dataset_info.name,
                                                                                    include_neurons=True,
                                                                                    include_params=True,
                                                                                    include_ei=True)
    else:
        for nscenes_dataset_info in config_settings_dict[dcp.NScenesFlashedDatasetSection.OUTPUT_KEY]:
            keyed_nscenes_datasets[nscenes_dataset_info.name] = vl.load_vision_data(nscenes_dataset_info.path,
                                                                                    nscenes_dataset_info.name,
                                                                                    include_neurons=True,
                                                                                    include_params=True,
                                                                                    include_ei=True)

    # we assume that the reference dataset was curated such that there are no duplicates
    # in the white noise classification

    # first take care of all of the manual matches if those are specified
    if args.manual_init:

        if args.brownian:
            manual_wn_ns_matches, ns_taken_dict = curate.parse_manual_matching(
                BROWNIAN_CELL_MATCHES[ref_lookup_key])
        else:
            print(ref_dataset_info.path)
            manual_wn_ns_matches, ns_taken_dict = curate.parse_manual_matching(
                MANUAL_CELL_MATCHES[ref_lookup_key])

        good_cell_ids_by_type = {}
        for cell_type in cell_types:
            temp_ref_ids = ref_dataset.get_all_cells_of_type(cell_type)
            good_cell_ids_by_type[cell_type] = [x for x in temp_ref_ids if x not in manual_wn_ns_matches]
    else:
        good_cell_ids_by_type = {cell_type: ref_dataset.get_all_cells_of_type(cell_type) for cell_type in cell_types}

    # then go back to automatic matching

    # get feature vectors for the reference dataset
    cells_to_match = np.array(
        np.concatenate([good_cell_ids_by_type[cell_type] for cell_type in cell_types])
    )

    matchable_ref_cells, ref_ei_feature_vectors = cossim.get_minmax_significant_ei_vectors(ref_dataset,
                                                                                           cells_to_match,
                                                                                           config_settings_dict[
                                                                                               SettingsSection.SIG_EL_CUTOFF],
                                                                                           config_settings_dict[
                                                                                               SettingsSection.N_SIG_EL])  # FIXME last two args

    # perform EI matching to the wn dataset for each nscenes dataset
    keyed_nscenes_one_to_many = {}  # type: Dict[str, Dict[int, List[int]]]
    for nscenes_name, nscenes_dataset in keyed_nscenes_datasets.items():
        # match cells using min/max cosine similarity
        # this works better than the linear correlation that we were using before
        # might replace this with a neural net embedding at some point
        # but currently that isn't a clean-cut winner over min/max cosine similarity

        # now match cells between the reference dataset and the natural scenes dataset by EI similarity
        # (works great for parasols in 60 um)

        # Note that we have to exclude cells that were already taken by the manual matching process,
        # if the manual matching process is specified
        if args.manual_init:
            _all_nscenes_cell_ids_list = [x for x in nscenes_dataset.get_cell_ids()
                                          if x not in ns_taken_dict[nscenes_name]]
            all_nscenes_cells = np.array(_all_nscenes_cell_ids_list, dtype=np.int64)
        else:
            all_nscenes_cells = np.array(nscenes_dataset.get_cell_ids(), dtype=np.int64)

        matchable_nscenes_cells, nscenes_ei_feature_vectors = cossim.get_minmax_significant_ei_vectors(
            nscenes_dataset,
            all_nscenes_cells,
            config_settings_dict[SettingsSection.SIG_EL_CUTOFF],
            config_settings_dict[SettingsSection.N_SIG_EL]
        )

        # now do the matching using cosine similarity, and load the cutoff threshold from the dictionary
        # calculate cosine similarity
        cosine_sim_matrix = cossim.cosine_similarity(ref_ei_feature_vectors, nscenes_ei_feature_vectors)

        # match cells
        one_to_many_match, unmatched = cossim.assign_match_until_threshold(
            matchable_ref_cells,
            matchable_nscenes_cells,
            cosine_sim_matrix,
            give_up_thresh=config_settings_dict[SettingsSection.CELL_MATCH_THRESH]
        )

        # Do we merge the one-to-many match by taking the match with the most spikes?
        # Or do we merge by combining all of the matches? This might be preferred because of the YASS temporal
        #   splitting problem
        # Alternatively we do maximum size subset contamination merging, which is probably the safest

        merged_one_to_many = curate.merge_duplicate_matches_by_spike_count(nscenes_dataset,
                                                                           one_to_many_match)

        keyed_nscenes_one_to_many[nscenes_name] = merged_one_to_many

    # output has format {(wn cell id) : { (nscenes dset name) : (nscenes cell id)}}
    intersection_one_to_one_map = curate.merge_matches(
        keyed_nscenes_one_to_many)  # type: Dict[int, Dict[str, List[int]]]

    # Collate the matched single cells. The order generated here is the order that
    # will be used in the model
    ordered_cells_to_convert = mcs.OrderedMatchedCellsStruct()

    # first add the manually matched cells, if we specified those
    if args.manual_init:
        for matched_cell_id, wn_match_dict in manual_wn_ns_matches.items():
            cell_type_name = ref_dataset.get_cell_type_for_cell(matched_cell_id)
            ordered_cells_to_convert.add_typed_match(cell_type_name, matched_cell_id,
                                                     wn_match_dict)

    for cell_type_name, good_wn_cell_ids in good_cell_ids_by_type.items():
        for wn_cell_id in good_wn_cell_ids:
            if wn_cell_id in intersection_one_to_one_map:  # we have a matched cell
                ordered_cells_to_convert.add_typed_match(cell_type_name, wn_cell_id,
                                                         intersection_one_to_one_map[wn_cell_id])

    # calculate RF center coordinates for matched cells only
    reference_cells_by_type = {cell_type: ordered_cells_to_convert.get_reference_cell_order(cell_type) \
                               for cell_type in ordered_cells_to_convert.get_cell_types()}
    rf_centers_matched_only = calculate_center_from_sta(ref_dataset,
                                                        reference_cells_by_type,
                                                        sig_stixel_threshold=4.0)
    rf_center_struct = mcs.RFCenterStruct(rf_centers_matched_only)

    ######## SAVE THE OUTPUTS TO THE SPECIFIED LOCATIONS USING PICKLE #############

    # make the folders if necessary
    responses_folder = os.path.dirname(config_settings_dict['responses_ordered'])
    os.makedirs(responses_folder, exist_ok=True)
    with open(config_settings_dict['responses_ordered'], 'wb') as picklefile:
        pickle.dump(ordered_cells_to_convert, picklefile)
        pickle.dump(rf_center_struct, picklefile)

    dict_to_pickle = {}  # type: Dict[str, np.ndarray]
    for cell_type in ordered_cells_to_convert.get_cell_types():
        included_ref_cells_in_order = ordered_cells_to_convert.get_reference_cell_order(cell_type)
        dict_to_pickle[cell_type] = lib.data_utils.sta_metadata.load_sigstixels_spatial_only_stas(ref_dataset,
                                                                                                  included_ref_cells_in_order,
                                                                                                  sig_stixels_threshold=4.0)

    for cell_type in cell_types:
        print("Matched {0} / {1} of {2}".format(len(ordered_cells_to_convert.get_reference_cell_order(cell_type)),
                                                len(ref_dataset.get_all_cells_of_type(cell_type)),
                                                cell_type))
