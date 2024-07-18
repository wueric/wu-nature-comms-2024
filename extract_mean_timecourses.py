import numpy as np
import h5py

import argparse
import pickle

from typing import Dict

import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils import sta_metadata
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Compute the mean STA timecourse for initializing LNBRC stimulus temporal filter')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    args = parser.parse_args()

    config_settings = dcp.read_config_file(args.cfg_file)

    # Load Vision dataset and frames for the reference dataset first
    ref_dataset_info = config_settings['ReferenceDataset'] # type: dcp.DatasetInfo

    ##########################################################################
    # Load precomputed cell matches, crops, etc.
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]
    bin_width_time_ms = int(np.around(samples_per_bin / 20, decimals=0))
    stimulus_onset_time_length = int(np.around(100 / bin_width_time_ms, decimals=0))

    ###### Compute the STA timecourse, and use that as the initial guess for ###
    ###### the timecourse ######################################################
    # Load the timecourse initial guess
    mean_timecourse_by_type = {} # type: Dict[str, np.ndarray]
    with h5py.File(ref_dataset_info.hires_sta, 'r') as sta_file:
        for cell_type in ct_order:
            relev_cell_ids = cells_ordered.get_reference_cell_order(cell_type)
            guess_cell_ids = relev_cell_ids[:min(20, len(relev_cell_ids))]
            stas_relevant_cells_by_id = {}
            for cell_id in guess_cell_ids:
                stas_relevant_cells_by_id[cell_id] = np.array(sta_file[f"{cell_id}"], dtype=np.float32)

            avg_timecourse = sta_metadata.calculate_average_timecourse_highres(
                stas_relevant_cells_by_id,
                guess_cell_ids,
                sig_stixels_cutoff=5.0,
                return_rgb=False
            )[::bin_width_time_ms]

            mean_timecourse_by_type[cell_type] = avg_timecourse

    with open(config_settings[dcp.OutputSection.INITIAL_GUESS_TIMECOURSE], 'wb') as pfile:
        pickle.dump(mean_timecourse_by_type, pfile)
