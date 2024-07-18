'''
We want to compute 4 different kinds of cross correlations

(1) Data cross-correlations
    * Compute center cell vs. its neighbors, apply the stimulus-presentation correction by shuffling
        with two trials away
(2) Shuffled data cross-correlations
    * Center cell and its neighbors like before, but shuffling the responses of the cells between
        the trials; apply the correction in the same way
(3) Simulated full GLM cross-correlations
(4) Simulated uncoupled GLM cross-correlation

These are expensive to evaluate, so rather than compute them for every pair of cells, we specify
    a maximum coupling distance

CPU only implementation

We compute the cross-correlation over the entire two-stimulus pair (i.e. roughly the same
    amount of spikes as was used for the reconstruction). To avoid double-counting, we
    compute for every other stimulus

'''

import argparse
import pickle
from typing import List, Tuple, Dict

import numpy as np
import tqdm

import lib.dataset_config_parser.dataset_config_parser as dcp
import lib.data_utils.dynamic_data_util as ddu
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.dataset_specific_hyperparams.jitter_glm_hyperparameters import \
    CROPPED_JITTER_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE
from sim_retina.jitter_sim_helper import SingleCellRepeatsJitteredMovieDataloader
from sim_retina.xcorr import unequal_bin_xcorr_hist_with_stim_structure_correation, unequal_bin_shuffle_repeats


def fetch_real_spikes_for_each_repeat(sim_dataloader: SingleCellRepeatsJitteredMovieDataloader,
                                      center_cell_id: int,
                                      coupled_cell_ids: List[int]) \
        -> Tuple[List[List[np.ndarray]], Dict[int, List[List[np.ndarray]]]]:
    n_repeats_total = sim_dataloader.n_repeats
    n_stimuli = sim_dataloader.n_stimuli

    # do every other stimulus, since we do the computation
    # over pairs of stimuli to make it like the reconstruction case
    center_cells_acc = []
    coupled_cells_acc = {cell_id: [] for cell_id in coupled_cell_ids}
    for stim_ix in range(0, n_stimuli, 2):

        center_cells_acc.append([])
        for cell_id in coupled_cell_ids:
            coupled_cells_acc[cell_id].append([])

        for repeat_ix in range(n_repeats_total):

            center_spikes, coupled_spikes, _, _ = sim_dataloader.fetch_data_repeat_spikes_and_bin_times(
                stim_ix,
                repeat_ix,
                center_cell_id,
                coupled_cell_ids=coupled_cell_ids
            )

            center_cells_acc[-1].append(center_spikes)
            for ii, coupled_cell_id in enumerate(coupled_cell_ids):
                coupled_cells_acc[coupled_cell_id][-1].append(coupled_spikes[ii, :])

    return center_cells_acc, coupled_cells_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Compute cross-correlations for eye movements stimulus")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('cell_type', type=str, help='cell type to compute distribution over')
    parser.add_argument('output', type=str, help='path to save path')
    parser.add_argument('-s', '--simulated_spike_trains', type=str,
                        default=None, help='path to simulated spike trains')

    args = parser.parse_args()

    config_settings = dcp.read_config_file(args.cfg_file)

    ref_dataset_info = config_settings[dcp.ReferenceDatasetSection.OUTPUT_KEY]  # type: dcp.DatasetInfo

    piece_name, ref_datarun_name = dcp.awsify_piece_name_and_datarun_lookup_key(
        ref_dataset_info.path,
        ref_dataset_info.name)

    #### Now load the natural scenes dataset #################################
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]

    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]
    bin_width_time_ms = samples_per_bin // 20

    ref_ds_lookup_key = dcp.generate_lookup_key_from_dataset_info(ref_dataset_info)
    model_fit_hyperparameters = CROPPED_JITTER_JOINT_WN_GLM_HYPERPARAMTERS_FN_BY_PIECE[ref_ds_lookup_key][bin_width_time_ms]()[args.cell_type]

    cell_distances_by_type = model_fit_hyperparameters.neighboring_cell_dist

    ################################################################
    # Load the natural scenes Vision datasets and determine what the
    # train and test partitions are
    nscenes_dataset_info_list = config_settings[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]

    create_test_dataset = (dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]

    # calculate timing for the nscenes dataset
    loaded_synchronized_datasets = ddu.load_jittered_nscenes_dataset_and_timebin(
        nscenes_dataset_info_list,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
    )

    ##########################################################################
    # Load precomputed cell matches, crops, etc.
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct

    ####### Load the previously identified interactions #########################
    with open(config_settings['featurized_interactions_ordered'], 'rb') as picklefile:
        pairwise_interactions = pickle.load(picklefile)  # type: InteractionGraph

    ##########################################################################
    # Construct the dataloader for fetching repeat stuff
    single_cell_repeats_sim_dataloader = SingleCellRepeatsJitteredMovieDataloader(
        loaded_synchronized_datasets,
        cells_ordered,
        samples_per_bin,
        image_rescale_lambda=None,  # don't need the stimulus
    )

    cell_ids_of_type = cells_ordered.get_reference_cell_order(args.cell_type)
    xcorr_by_cell_id = {}
    cells_pbar = tqdm.tqdm(total=len(cell_ids_of_type))
    if args.simulated_spike_trains is None:
        for center_cell_id in cell_ids_of_type:

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

            if args.simulated_spike_trains is None:
                # compute the shuffled and real data cross correlations
                center_cell_spikes, coupled_cell_spikes = fetch_real_spikes_for_each_repeat(
                    single_cell_repeats_sim_dataloader,
                    center_cell_id,
                    all_coupled_cell_ids_ordered
                )

                real_xcorr_dict, shuffle_xcorr_dict = {}, {}
                pbar = tqdm.tqdm(total=len(all_coupled_cell_ids_ordered))
                for coupled_cell_id, coupled_spike_trains in coupled_cell_spikes.items():
                    real_xcorr_dict[coupled_cell_id] = unequal_bin_xcorr_hist_with_stim_structure_correation(
                        center_cell_spikes,
                        coupled_spike_trains
                    )

                    shuffle_xcorr_dict[coupled_cell_id] = unequal_bin_xcorr_hist_with_stim_structure_correation(
                        center_cell_spikes,
                        unequal_bin_shuffle_repeats(coupled_spike_trains)
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

        # open simulated spikes file
        with open(args.simulated_spike_trains, 'rb') as pfile:
            simulated_spikes_contents = pickle.load(pfile)

        for center_cell_id in cell_ids_of_type:
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

            simulated_center_spikes = simulated_spikes_contents['simulated'][center_cell_id]

            # compute the shuffled and real data cross correlations
            _, coupled_cell_spikes = fetch_real_spikes_for_each_repeat(
                single_cell_repeats_sim_dataloader,
                center_cell_id,
                all_coupled_cell_ids_ordered
            ) # this was every other stimulus to avoid double-counting

            sim_xcorr_dict, sim_shuffle_xcorr_dict = {}, {}
            pbar = tqdm.tqdm(total=len(all_coupled_cell_ids_ordered))
            for coupled_cell_id, coupled_spike_trains in coupled_cell_spikes.items():
                sim_xcorr_dict[coupled_cell_id] = unequal_bin_xcorr_hist_with_stim_structure_correation(
                    simulated_center_spikes[::2], # do every other stimulus to avoid double counting
                    coupled_spike_trains
                )

                sim_shuffle_xcorr_dict[coupled_cell_id] = unequal_bin_xcorr_hist_with_stim_structure_correation(
                    simulated_center_spikes[::2], # do every other stimulus to avoid double counting
                    unequal_bin_shuffle_repeats(coupled_spike_trains)
                )

                pbar.update(1)
            pbar.close()

            xcorr_by_cell_id[center_cell_id] = {
                'cell_id': center_cell_id,
                'coupled': all_coupled_cell_ids_ordered,
                'simulated_real_xcorr': sim_xcorr_dict,
                'simulated_shuffled_xcorr': sim_shuffle_xcorr_dict,
                'by_type': coupled_by_type
            }

            cells_pbar.update(1)

    cells_pbar.close()

    with open(args.output, 'wb') as pfile:
        pickle.dump(xcorr_by_cell_id, pfile)
