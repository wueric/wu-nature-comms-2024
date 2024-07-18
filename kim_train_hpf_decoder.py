import argparse
import os
import pickle
from typing import List

import numpy as np

import torch
import torch.utils.data as torch_data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu

from kim_et_al_networks.ns_decoder import Parallel_NN_Decoder, ThinDatasetWrapper, train_parallel_NN_decoder

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Fit Kim-style linear model with L1 regularization from config; config must be single-bin binom, not GLM')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('target_train_path', type=str, help='path to target HPF images pickle')
    parser.add_argument('coeffs_file_path', type=str, help='path to pickled linear model coefficients')
    parser.add_argument('trained_decoder_path', type=str, help='path to save trained HPF decoder')
    parser.add_argument('tboard', type=str, help='path to Tensorboard output')
    parser.add_argument('-k', '--k_dim', type=int, default=25, help='Number of cells per pixel')
    parser.add_argument('-dh', '--h_dim', type=int, default=40, help='Number of cells per pixel')
    parser.add_argument('-f', '--f_dim', type=int, default=5, help='Number of cells per pixel')
    parser.add_argument('-e', '--n_epochs', type=int, default=16, help='Number of training epochs')
    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    #####################################################
    k_dim= args.k_dim
    h_dim = args.h_dim
    f_dim= args.f_dim

    #####################################################
    # load the HPF train frames
    print("Loading frames")
    with open(args.target_train_path, 'rb') as pfile:
        all_frames = pickle.load(pfile)
        train_hpf_frames = all_frames['train']
        test_hpf_frames= all_frames['test']
    n_frames, height, width = train_hpf_frames.shape

    #####################################################
    # load the pixel weights from the L1 problem
    target_pickle_file = os.path.join(args.coeffs_file_path)
    with open(target_pickle_file, 'rb') as pfile:
        summary_dict = pickle.load(pfile)
        reconstruction_coeffs = summary_dict['coeffs']

    # figure out which cells are assigned to which pixels
    abs_recons_coeffs = np.abs(reconstruction_coeffs)
    argsort_by_cell = np.argsort(abs_recons_coeffs, axis=0)

    # shape (k_dim, height, width)
    biggest_coeffs_ix = argsort_by_cell[-k_dim:, ...]

    # shape (n_pix, k_dim)
    flattened_coeff_sel = biggest_coeffs_ix.reshape(biggest_coeffs_ix.shape[0], -1).transpose(1, 0)

    #####################################################
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

    ####### Load info about the cell ids, cell types, and matches #############
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

    ####### Bin spikes ########################################################3
    print('Binning spikes')
    train_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                 ddu.PartitionType.TRAIN_PARTITION)

    test_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                 ddu.PartitionType.TEST_PARTITION)


    train_kim_decoder_response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        train_flashed_patches,
    )

    test_kim_decoder_response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        test_flashed_patches,
    )

    ###### Construct datasets and dataloaders ##################################
    train_dataset= ThinDatasetWrapper(train_hpf_frames,
                                      train_kim_decoder_response_vector)
    test_dataset = ThinDatasetWrapper(test_hpf_frames,
                                      test_kim_decoder_response_vector)

    train_dataloader = torch_data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = torch_data.DataLoader(test_dataset, batch_size=8, shuffle=True)

    ####### Build the decoder model ##########################################3
    decoder_nn_model = Parallel_NN_Decoder(flattened_coeff_sel,
                                           train_kim_decoder_response_vector.shape[1],
                                           train_kim_decoder_response_vector.shape[2],
                                           k_dim,
                                           h_dim,
                                           height * width,
                                           f_dim).to(device)

    summary_writer_path = os.path.join(args.tboard, 'tensorboard')
    summary_writer = SummaryWriter(summary_writer_path)

    print('Begin training')

    trained_decoder_nn_model= train_parallel_NN_decoder(
        decoder_nn_model,
        train_dataloader,
        test_dataloader,
        nn.MSELoss(),
        device,
        summary_writer,
        n_epochs=args.n_epochs
    )

    print("Saving fitted Kim NSDecoder")
    torch.save({'decoder': decoder_nn_model.state_dict()}, args.trained_decoder_path)

