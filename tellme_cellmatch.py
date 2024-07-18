import numpy as np
import visionloader as vl
import argparse

import lib.data_utils.cosine_similarity as cossim

from lib.dataset_config_parser.dataset_config_parser import read_config_file, SettingsSection
import lib.dataset_config_parser.dataset_config_parser as dcp

from typing import List, Dict

from tabulate import tabulate

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Match cells between WN and nscenes with cosine similarity')
    parser.add_argument('cfg_path', type=str, help='path to config file')
    parser.add_argument('reference_id', type=int, help='reference dataset cell id to match')
    parser.add_argument('--top_n', type=int, default=5, help='top N cells to match')
    args = parser.parse_args()

    config_settings_dict = read_config_file(args.cfg_path)

    cell_types = config_settings_dict['CellTypes']  # type: List[str]

    ref_dataset_info = config_settings_dict['ReferenceDataset']

    ref_dataset = vl.load_vision_data(ref_dataset_info.path,
                                      ref_dataset_info.name,
                                      include_params=True,
                                      include_ei=True)

    ref_ei_matchable, ref_ei_minmax_features = cossim.get_minmax_significant_ei_vectors(
        ref_dataset,
        np.array([args.reference_id], dtype=np.int64),
        config_settings_dict[SettingsSection.SIG_EL_CUTOFF],
        config_settings_dict[SettingsSection.N_SIG_EL]
    )

    keyed_nscenes_datasets = {}  # type: Dict[str, vl.VisionCellDataTable]
    if dcp.NScenesFlashedDatasetSection.OUTPUT_KEY in config_settings_dict:
        nscenes_dataset_list = config_settings_dict[dcp.NScenesFlashedDatasetSection.OUTPUT_KEY]
    else:
        nscenes_dataset_list = config_settings_dict[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]



    for nscenes_dataset_info in nscenes_dataset_list:
        keyed_nscenes_datasets[nscenes_dataset_info.name] = vl.load_vision_data(nscenes_dataset_info.path,
                                                                                nscenes_dataset_info.name,
                                                                                include_params=True,
                                                                                include_ei=True)

    for nscenes_name, nscenes_dataset in keyed_nscenes_datasets.items():
        # match cells using min/max cosine similarity
        # this works better than the linear correlation that we were using before
        # might replace this with a neural net embedding at some point
        # but currently that isn't a clean-cut winner over min/max cosine similarity

        # now match cells between the reference dataset and the natural scenes dataset by EI similarity
        # (works great for parasols in 60 um)
        all_nscenes_cells = np.array(nscenes_dataset.get_cell_ids())

        matchable_nscenes_cells, nscenes_ei_feature_vectors = cossim.get_minmax_significant_ei_vectors(
            nscenes_dataset,
            all_nscenes_cells,
            config_settings_dict[SettingsSection.SIG_EL_CUTOFF],
            config_settings_dict[SettingsSection.N_SIG_EL]
        )

        # now do the matching using cosine similarity, and load the cutoff threshold from the dictionary
        # calculate cosine similarity
        cosine_sim_matrix = cossim.cosine_similarity(ref_ei_minmax_features, nscenes_ei_feature_vectors).squeeze(0)
        best_ix = (np.argsort(cosine_sim_matrix)[-args.top_n:])[::-1]

        vals = [['ids', *list(matchable_nscenes_cells[best_ix])], ['score', *list(cosine_sim_matrix[best_ix])]]
        print(tabulate(vals, headers=[f'{nscenes_name}', *[i+1 for i in range(args.top_n)]]))

        print('\n\n')
