import torch
import numpy as np

import argparse
import pickle
from typing import List

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Generate linear reconstructions for flashed images')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('pretrained_linear_model', type=str, help='path to pickled linear model .pth')
    parser.add_argument('save_path', type=str, help='path to save generated images')
    parser.add_argument('-hh', '--heldout', action='store_true', default=False, help='Use heldout; default use test')
    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    ####### Load info about the cell ids, cell types, and matches #############
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

    #### Now load the natural scenes dataset #################################
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

    n_bins_before = config_settings[dcp.TimebinningSection.NBINS_BEFORE_TRANS]
    n_bins_after = config_settings[dcp.TimebinningSection.NBINS_AFTER_TRANS]
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    # Load the natural scenes Vision datasets and determine what the
    # train and test partitions are
    nscenes_dataset_info_list = config_settings[dcp.NScenesFlashedDatasetSection.OUTPUT_KEY]

    create_test_dataset = (dcp.TestFlashedSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutFlashedSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestFlashedSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutFlashedSection.MOVIE_BLOCK_DESCRIPTOR]

    bin_width_time_ms = samples_per_bin // 20
    stimulus_onset_time_length = 100 // bin_width_time_ms

    nscenes_dset_list = ddu.load_nscenes_dataset_and_timebin_blocks3(
        nscenes_dataset_info_list,
        samples_per_bin,
        n_bins_before,
        n_bins_after,
        stimulus_onset_time_length,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
        downsample_factor=nscenes_downsample_factor,
        crop_w_low=crop_width_low,
        crop_w_high=crop_width_high,
        crop_h_low=crop_height_low,
        crop_h_high=crop_height_high
    )

    for item in nscenes_dset_list:
        item.load_frames_from_disk()

    ###############################################################
    # Load and optionally downsample/crop the stimulus frames
    if args.heldout:
        test_heldout_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                            ddu.PartitionType.HELDOUT_PARTITION)
    else:
        test_heldout_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                            ddu.PartitionType.TEST_PARTITION)

    test_heldout_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(test_heldout_flashed_patches))
    print(test_heldout_frames.shape)

    binom_response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        test_heldout_flashed_patches
    ).squeeze(2)

    binom_response_vector_torch = torch.tensor(binom_response_vector, dtype=torch.float32, device=device)

    ####### Load the pretrained linear model ###################################
    linear_decoder = torch.load(args.pretrained_linear_model, map_location=device)

    with torch.no_grad():
        reconstructed_images = linear_decoder(binom_response_vector_torch).detach().cpu().numpy()

    with open(args.save_path, 'wb') as pfile:
        save_data = {
            'ground_truth': test_heldout_frames,
            'linear': reconstructed_images
        }

        pickle.dump(save_data, pfile)

    print('done')

