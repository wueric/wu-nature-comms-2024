import numpy as np

import argparse
import pickle
from typing import List

from lib.data_utils.dynamic_data_util import JitteredMovieBatchDataloader
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.dataset_config_parser.dataset_config_parser as dcp
import lib.data_utils.dynamic_data_util as ddu
import lib.data_utils.data_util as du
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute number of spikes for each stimulus presentation. Skips the first' + \
                                     'stimulus in each block to mimic the reconstruction')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('partition', type=str, help='Either "test", "heldout"')
    parser.add_argument('output_path', type=str, help='path to save metrics dict')
    parser.add_argument('-o', '--onset', type=int, default=0,
                        help='onset time length, in units of bins set in the config, if nonzero break up into onset and remaining')

    args = parser.parse_args()

    config_settings = read_config_file(args.cfg_file)

    ################################################################
    # Load the cell types and matching
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

    ################################################################
    # Load some of the model fit parameters
    # Not strictly necessary but we're lazy
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

    #################################################################
    # Load the raw data
    create_test_dataset = (dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]


    nscenes_dataset_info_list = config_settings[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]

    loaded_synchronized_datasets = ddu.load_jittered_nscenes_dataset_and_timebin(
        nscenes_dataset_info_list,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
    )

    jitter_dataloader = JitteredMovieBatchDataloader(
        loaded_synchronized_datasets,
        cells_ordered,
        ddu.PartitionType.HELDOUT_PARTITION if args.partition == 'heldout' \
            else ddu.PartitionType.TEST_PARTITION,
        samples_per_bin,
        image_rescale_lambda=image_rescale_lambda,
        crop_w_ix=(crop_width_low, 320 - crop_width_low), # FIXME
        time_jitter_spikes=0.0
    )

    print("Counting spikes")
    pbar = tqdm.tqdm(total=len(jitter_dataloader))
    onset_spikes_by_cell_type = {ct: [] for ct in ct_order}
    remaining_spike_counts_by_cell_type = {ct : [] for ct in ct_order}
    n_cells_by_type = cells_ordered.get_n_cells_by_type()
    for i in range(len(jitter_dataloader)):

        history_frames, target_frames, snippet_transitions, spike_bins, binned_spikes = jitter_dataloader[i]

        # need to figure out which bin to start summing over
        appearance_time = snippet_transitions[60]
        first_bin_after_transition = np.argmax(spike_bins > appearance_time)
        subset_spikes = binned_spikes[:, first_bin_after_transition:]

        onset_spikes = np.sum(subset_spikes[:, :args.onset], axis=1)
        remaining_spikes = np.sum(subset_spikes[:, args.onset:], axis=1)

        # we only care about the binned spikes
        # have to partition by cell type
        # this is a bit of a hack, but easier than doing it properly
        offset = 0
        for ct in ct_order:
            n_cells_of_type = n_cells_by_type[ct]
            sliced_remaining_spikes = remaining_spikes[offset:offset+n_cells_of_type]
            sliced_onset_spikes = onset_spikes[offset:offset+n_cells_of_type]

            remaining_spike_counts_by_cell_type[ct].append(sliced_remaining_spikes)
            onset_spikes_by_cell_type[ct].append(sliced_onset_spikes)

            offset += n_cells_of_type

        pbar.update(1)
    pbar.close()

    print("Writing output")
    onset_spike_counts_by_cell_type_array = {ct: np.array(val) for ct, val in onset_spikes_by_cell_type.items()}
    remaining_spike_counts_by_cell_type_array = {ct: np.array(val) for ct, val in remaining_spike_counts_by_cell_type.items()}
    with open(args.output_path, 'wb') as pfile:
        pickle.dump({
            'onset spike count': onset_spike_counts_by_cell_type_array,
            'remaining spike count': remaining_spike_counts_by_cell_type_array,
            'onset bins': args.onset
        }, pfile)

    print("done")
