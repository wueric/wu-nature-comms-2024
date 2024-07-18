import numpy as np
import visionloader as vl
import argparse

import lib.data_utils.contamination as contamination

import spikebinning

from lib.dataset_config_parser.dataset_config_parser import read_config_file, SettingsSection
import lib.dataset_config_parser.dataset_config_parser as dcp

from typing import List, Dict, Tuple

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Match cells between WN and nscenes with cosine similarity')
    parser.add_argument('cfg_path', type=str, help='path to config file')
    parser.add_argument('dataset', type=str, help='relevant dataset')
    parser.add_argument('cell_ids', type=int, nargs='*', help='cell_ids to check contamination')
    args = parser.parse_args()

    config_settings_dict = read_config_file(args.cfg_path)
    ds_name = args.dataset

    if dcp.NScenesFlashedDatasetSection.OUTPUT_KEY in config_settings_dict:
        nscenes_dataset_list = config_settings_dict[dcp.NScenesFlashedDatasetSection.OUTPUT_KEY]
    else:
        nscenes_dataset_list = config_settings_dict[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]


    for nscenes_dataset_info in nscenes_dataset_list:
        if nscenes_dataset_info.name == ds_name:
            to_check = vl.load_vision_data(nscenes_dataset_info.path,
                                           nscenes_dataset_info.name,
                                           include_neurons=True,
                                           include_params=True)

            spike_time_merge_list = [to_check.get_spike_times_for_cell(cell_id)
                                     for cell_id in args.cell_ids]

            merged_spike_times = spikebinning.merge_multiple_sorted_array(spike_time_merge_list)
            is_sorted = np.any(merged_spike_times[1:] > merged_spike_times[:-1])
            print(f'is sorted: {is_sorted}')

            refractory_period_violations = contamination.count_refractory_violations(merged_spike_times)
            n_spikes = merged_spike_times.shape[0]
            contamination_score = contamination.calculate_contamination(merged_spike_times)

            print(f'cells {args.cell_ids}: {n_spikes} total spikes, {refractory_period_violations} violations, total contam. {contamination_score}')

            break
