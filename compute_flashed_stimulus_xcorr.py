'''

We want to compute several different kinds of cross-correlations, so that we can isolate
    the stimulus-driven and noise contributions to the cross-correlations

(1) Data cross-correlations
    * Compute center cell vs neighbors, apply the stimulus-presentation correction by shuffling
        with two trials away
(2) Shuffled data cross-correlations
    * Center cell and its neighbors like before, but shuffling the responses of each cell between
        the trials; apply the correction the same way

(3) Simulated full GLM cross-correlations
    * Simulate center cells with neighbors, where the neighbor spike trains are all taken from
        real data trials. Compute the stimulus correction in the same way

(4) Simulated full GLM cross-correlations with shuffled neighbors
    * Simulate center cell, where the neighbor spike trains are all taken from real data trials

        Then compute cross-correlations between the simulated spike train and shuffled data

        If we subtract (4) from (3) we should see the noise-correlation component of the GLM simulation

(5) Simulated uncoupled GLM cross-correlations
    * Simulate center cells with neighbors, where the neighbor spike trains are all taken from
        real data trials. Compute the stimulus correction in the same way

(6) Simulated uncoupled GLM cross-correlations with shuffled neighbors
    * Shuffle the neighbors when computing the cross-correlation. Should give the same
        result as (5), but I want to show realistic variability rather than just write down 0

        If we subtract (6) from (5) we should get exactly 0, but with noise
'''


import argparse
import pickle
from typing import List, Tuple, Dict

import numpy as np
import tqdm

import lib.dataset_config_parser.dataset_config_parser as dcp
import lib.data_utils.dynamic_data_util as ddu
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.dataset_specific_hyperparams.glm_hyperparameters import CROPPED_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE2
from sim_retina.xcorr import crosscorr_hist_with_stim_structure_correction, batched_equal_bin_shuffle_repeats

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Compute cross correlations for flashed image presentations")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('cell_type', type=str, help='cell type to compute distribution over')
    parser.add_argument('output', type=str, help='path to save path')
    parser.add_argument('-s', '--simulated_spike_trains', type=str,
                        default=None, help='path to simulated spike trains')

    args = parser.parse_args()

    config_settings = dcp.read_config_file(args.cfg_file)

    ref_dataset_info = config_settings[dcp.ReferenceDatasetSection.OUTPUT_KEY]  # type: dcp.DatasetInfo

    ################################################################
    # Load the cell types and matching
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

    ####### Load the previously identified interactions #########################
    with open(config_settings['featurized_interactions_ordered'], 'rb') as picklefile:
        pairwise_interactions = pickle.load(picklefile)  # type: InteractionGraph

    #################################################################
    n_bins_before = config_settings[dcp.TimebinningSection.NBINS_BEFORE_TRANS]
    n_bins_after = config_settings[dcp.TimebinningSection.NBINS_AFTER_TRANS]
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]
    bin_width_time_ms = int(np.around(samples_per_bin / 20, decimals=0))
    stimulus_onset_time_length = int(np.around(100 / bin_width_time_ms, decimals=0))

    ref_lookup_key = dcp.generate_lookup_key_from_dataset_info(ref_dataset_info)
    model_fit_hyperparameters = \
        CROPPED_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE2[ref_lookup_key][bin_width_time_ms]()[args.cell_type]

    cell_distances_by_type = model_fit_hyperparameters.neighboring_cell_dist

    # Load the natural scenes Vision datasets
    # we only care about the repeats in this case, since that's the only
    # way we can characterize noise correlations
    nscenes_dataset_info_list = config_settings[dcp.NScenesFlashedDatasetSection.OUTPUT_KEY]

    create_test_dataset = (dcp.TestFlashedSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutFlashedSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestFlashedSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutFlashedSection.MOVIE_BLOCK_DESCRIPTOR]

    nscenes_dset_list = ddu.load_nscenes_dataset_and_timebin_blocks3(
        nscenes_dataset_info_list,
        samples_per_bin,
        n_bins_before,
        n_bins_after,
        stimulus_onset_time_length,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
    ) # note that we don't care about the stimulus in this case
    # since the spikes have already been simulated

    data_repeats_response_vector = ddu.timebin_load_repeats_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        nscenes_dset_list,
    )

    n_repeats, n_stimuli, n_cells, n_bins = data_repeats_response_vector.shape

    cell_ids_of_type = cells_ordered.get_reference_cell_order(args.cell_type)
    cells_pbar = tqdm.tqdm(total=len(cell_ids_of_type))
    xcorr_by_cell_id = {}

    if args.simulated_spike_trains is None:
        for center_cell_id in cell_ids_of_type:

            center_ix_sel = cells_ordered.get_concat_idx_for_cell_id(center_cell_id)
            # shape (n_repeats, n_stimuli, n_bins)
            center_spike_trains = data_repeats_response_vector[:, :, center_ix_sel, :]

            ###### Get the cells that we want to compute the xcorr with #################
            ##### These are cells whose distance is less than what was specified ########
            ##### in args.coupling_dist, of any cell type
            ct_order = cells_ordered.get_cell_types()
            all_coupled_cell_ids_ordered = []  # type: List[int]
            coupled_by_type = {}
            for coupled_cell_type in ct_order:
                interaction_edges = pairwise_interactions.query_cell_interaction_edges(center_cell_id,
                                                                                       coupled_cell_type)
                coupled_cell_ids = [x.dest_cell_id for x in interaction_edges if
                                    x.additional_attributes['distance'] < cell_distances_by_type[coupled_cell_type]]
                all_coupled_cell_ids_ordered.extend(coupled_cell_ids)
                coupled_by_type[coupled_cell_type] = coupled_cell_ids

            real_xcorr_dict, shuffle_xcorr_dict = {}, {}
            pbar = tqdm.tqdm(total=len(all_coupled_cell_ids_ordered))
            for coupled_cell_id in all_coupled_cell_ids_ordered:
                coupled_ix_sel = cells_ordered.get_concat_idx_for_cell_id(coupled_cell_id)

                # shape (n_repeats, n_stimuli, n_bins)
                coupled_spike_trains = data_repeats_response_vector[:, :, coupled_ix_sel, :]

                real_xcorr_dict[coupled_cell_id] = crosscorr_hist_with_stim_structure_correction(
                    center_spike_trains, coupled_spike_trains
                )

                shuffle_xcorr_dict[coupled_cell_id] = crosscorr_hist_with_stim_structure_correction(
                    center_spike_trains, batched_equal_bin_shuffle_repeats(coupled_spike_trains)
                )

                pbar.update(1)
            pbar.close()

            xcorr_by_cell_id[center_cell_id] = {
                'cell_id': center_cell_id,
                'coupled': all_coupled_cell_ids_ordered,
                'real_xcorr': real_xcorr_dict,
                'shuffled_xcorr': shuffle_xcorr_dict,
                'by_type': coupled_by_type
            }

            cells_pbar.update(1)

    else:

        with open(args.simulated_spike_trains, 'rb') as pfile:
            simulation_contents = pickle.load(pfile)
            all_simulated_spike_trains = simulation_contents['simulated']

        for center_cell_id in cell_ids_of_type:

            center_ix_sel = cells_ordered.get_concat_idx_for_cell_id(center_cell_id)
            # shape (n_repeats, n_stimuli, n_bins)
            center_spike_trains = data_repeats_response_vector[:, :, center_ix_sel, :]

            ###### Get the cells that we want to compute the xcorr with #################
            ##### These are cells whose distance is less than what was specified ########
            ##### in args.coupling_dist, of any cell type
            ct_order = cells_ordered.get_cell_types()
            all_coupled_cell_ids_ordered = []  # type: List[int]
            coupled_by_type = {}

            for coupled_cell_type in ct_order:
                interaction_edges = pairwise_interactions.query_cell_interaction_edges(center_cell_id,
                                                                                       coupled_cell_type)
                coupled_cell_ids = [x.dest_cell_id for x in interaction_edges if
                                    x.additional_attributes['distance'] < cell_distances_by_type[coupled_cell_type]]
                all_coupled_cell_ids_ordered.extend(coupled_cell_ids)
                coupled_by_type[coupled_cell_type] = coupled_cell_ids

            simulated_spike_trains = all_simulated_spike_trains[center_cell_id]

            sim_real_xcorr_dict, sim_shuffle_xcorr_dict = {}, {}
            pbar = tqdm.tqdm(total=len(all_coupled_cell_ids_ordered))
            for coupled_cell_id in all_coupled_cell_ids_ordered:
                coupled_ix_sel = cells_ordered.get_concat_idx_for_cell_id(coupled_cell_id)

                # shape (n_repeats, n_stimuli, n_bins)
                coupled_spike_trains = data_repeats_response_vector[:, :, coupled_ix_sel, :]

                shuffled_coupled_spike_trains = batched_equal_bin_shuffle_repeats(
                    coupled_spike_trains
                )

                sim_real_xcorr_dict[coupled_cell_id] = crosscorr_hist_with_stim_structure_correction(
                    simulated_spike_trains, coupled_spike_trains
                )

                sim_shuffle_xcorr_dict[coupled_cell_id] = crosscorr_hist_with_stim_structure_correction(
                    simulated_spike_trains, shuffled_coupled_spike_trains
                )

                pbar.update(1)
            pbar.close()

            xcorr_by_cell_id[center_cell_id] = {
                'cell_id': center_cell_id,
                'coupled': all_coupled_cell_ids_ordered,
                'simulated_real_xcorr': sim_real_xcorr_dict,
                'simulated_shuffled_xcorr': sim_shuffle_xcorr_dict,
                'by_type': coupled_by_type
            }

            cells_pbar.update(1)

    cells_pbar.close()

    with open(args.output, 'wb') as pfile:
        pickle.dump(xcorr_by_cell_id, pfile)
