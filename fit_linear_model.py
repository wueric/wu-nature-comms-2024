import argparse
import pickle
from typing import List

import numpy as np
import torch

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu

from linear_decoding_models.linear_decoding_models import ClosedFormLinearModel


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Fit linear reconstruction model for flashed images')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('model_output_path', type=str, help='path to pickled linear model .pth')
    parser.add_argument('-m', '--max_training_examples', type=int, default=-1,
                        help='specify maximum number of training examples')
    parser.add_argument('-l', '--l2_reg', type=float, default=0.0, help='optional l2 regularization')
    parser.add_argument('-s', '--state_dict', action='store_true', help='save linear model weights as state dict')
    args = parser.parse_args()

    _max_n_samples = args.max_training_examples
    use_limited_data = (args.max_training_examples != -1)

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

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

    ####### Load info about the cell ids, cell types, and matches #############
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

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

    # Load and optionally downsample/crop the stimulus frames
    train_flashed_patches=ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                               ddu.PartitionType.TRAIN_PARTITION)

    train_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(train_flashed_patches))

    if use_limited_data:
        train_frames = train_frames[:_max_n_samples, ...]

    n_train_frames = train_frames.shape[0]
    height, width = train_frames.shape[1:]

    ####### Bin spikes ########################################################3
    print('Binning spikes')
    train_binom_response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        train_flashed_patches,
    ).squeeze(2)

    if use_limited_data:
        train_binom_response_vector = train_binom_response_vector[:_max_n_samples, ...]

    linear_decoder = ClosedFormLinearModel(
        train_binom_response_vector.shape[1],
        height,
        width
    ).to(device)

    train_binom_response_torch = torch.tensor(train_binom_response_vector, dtype=torch.float32,
                                              device=device)

    # copy the train frames to GPU
    train_frames_torch = torch.tensor(train_frames, dtype=torch.float32,
                                      device=device)

    with torch.no_grad():

        if args.l2_reg == 0.0:
            linear_decoder.solve(train_binom_response_torch, train_frames_torch)
        else:
            linear_decoder.solve_l2reg(train_binom_response_torch, train_frames_torch,
                                       args.l2_reg)

    del train_binom_response_torch, train_frames_torch

    print("Saving fitted linear decoder")
    if args.state_dict:
        torch.save(linear_decoder.state_dict(), args.model_output_path)
    else:
        torch.save(linear_decoder, args.model_output_path)
