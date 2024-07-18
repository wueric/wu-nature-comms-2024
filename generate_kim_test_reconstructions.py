import argparse
import os
import pickle
from typing import List, Callable

import numpy as np

import torch
import torch.nn as nn

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu

from kim_et_al_networks.ns_decoder import Parallel_NN_Decoder
from kim_et_al_networks.nn_deblur import ResnetGenerator

from linear_decoding_models.linear_decoding_models import ClosedFormLinearModel


def generate_full_kim_examples(linear_lpf_decoder: nn.Module,
                               hpf_decoder: nn.Module,
                               deblur_network: nn.Module,
                               ground_truth_images: np.ndarray,
                               observed_spikes: np.ndarray,
                               device: torch.device,
                               batch_size: int = 32) -> np.ndarray:
    n_tot_images, height, width = ground_truth_images.shape

    generated_images_buffer = np.zeros((n_tot_images, height, width), dtype=np.float32)
    for low in range(0, n_tot_images, batch_size):
        high = min(low + batch_size, n_tot_images)

        with torch.no_grad():
            # shape (batch, n_cells, n_bins)
            batched_spikes_torch = torch.tensor(observed_spikes[low:high, ...],
                                                dtype=torch.float32, device=device)
            batched_spikes_1bin_torch = torch.sum(batched_spikes_torch, dim=2)

            linear_reconstructed = linear_lpf_decoder(batched_spikes_1bin_torch)
            hpf_decoded = hpf_decoder(batched_spikes_torch).reshape(-1, height, width)

            combined = linear_reconstructed + hpf_decoded

            deblurred = deblur_network(combined[:, None, :, :]).squeeze(1)
            generated_images_buffer[low:high] = deblurred.detach().cpu().numpy()

    return generated_images_buffer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Generate flashed reconstructions using the Kim et al ANN benchmark method')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('coeffs_file_path', type=str, help='path to pickled linear model coefficients')
    parser.add_argument('lpf_linear_decoder_path', type=str, help='path to pickled linear model coefficients')
    parser.add_argument('hpf_nn_decoder_path', type=str, help='path to trained HPF decoder')
    parser.add_argument('deblur_path', type=str, help='path to deblur network')
    parser.add_argument('image_save_path', type=str, help='path to Tensorboard output')
    parser.add_argument('-k', '--k_dim', type=int, default=25, help='Number of cells per pixel')
    parser.add_argument('-dh', '--h_dim', type=int, default=40, help='Number of cells per pixel')
    parser.add_argument('-f', '--f_dim', type=int, default=5, help='Number of cells per pixel')
    parser.add_argument('-hh', '--heldout', action='store_true', default=False, help='Use heldout set')
    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    ###########################################################
    k_dim = args.k_dim
    h_dim = args.h_dim
    f_dim = args.f_dim

    #####################################################
    # bin spikes, get frames, etc.
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

    if args.heldout:
        test_heldout_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                            ddu.PartitionType.HELDOUT_PARTITION)
    else:
        test_heldout_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                            ddu.PartitionType.TEST_PARTITION)

    test_heldout_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(test_heldout_flashed_patches))

    n_test_frames = test_heldout_frames.shape[0]
    height, width = test_heldout_frames.shape[1:]

    ####### Bin spikes ########################################################3
    print('Binning spikes')
    kim_decoder_response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        test_heldout_flashed_patches
    )

    # Construct all of the models required to do the reconstructions
    #####################################################
    # load the optimal linear decoder
    with open(args.lpf_linear_decoder_path, 'rb') as pfile:
        summary_dict = pickle.load(pfile)
        lpf_decoder_filters = summary_dict['coeffs']

    # put this into a linear reconstructor module
    linear_decoder = ClosedFormLinearModel(
        lpf_decoder_filters.shape[0],
        lpf_decoder_filters.shape[1],
        lpf_decoder_filters.shape[2]
    ).to(device)

    decoder_filters_torch = torch.tensor(lpf_decoder_filters, dtype=torch.float32,
                                         device=device)
    linear_decoder.set_linear_filters(decoder_filters_torch)
    linear_decoder = linear_decoder.eval()

    #####################################################
    # load the pixel weights from the L1 problem
    with open(args.coeffs_file_path, 'rb') as pfile:
        summary_dict = pickle.load(pfile)
        reconstruction_coeffs = summary_dict['coeffs']

    # figure out which cells are assigned to which pixels
    abs_recons_coeffs = np.abs(reconstruction_coeffs)
    argsort_by_cell = np.argsort(abs_recons_coeffs, axis=0)

    # shape (k_dim, height, width)
    biggest_coeffs_ix = argsort_by_cell[-k_dim:, ...]

    # shape (n_pix, k_dim)
    flattened_coeff_sel = biggest_coeffs_ix.reshape(biggest_coeffs_ix.shape[0], -1).transpose(1, 0)

    # then load the nonlinear network$a
    hpf_decoder_nn_model = Parallel_NN_Decoder(flattened_coeff_sel,
                                               kim_decoder_response_vector.shape[1],
                                               kim_decoder_response_vector.shape[2],
                                               k_dim,
                                               h_dim,
                                               height * width,
                                               f_dim).to(device)
    saved_hpf_network = torch.load(args.hpf_nn_decoder_path,
                                   map_location=device)
    hpf_decoder_nn_model.load_state_dict(saved_hpf_network['decoder'])
    decoder_nn_model = hpf_decoder_nn_model.eval()

    deblur_network = ResnetGenerator(input_nc=1,
                                     output_nc=1,
                                     n_blocks=6).to(device)
    saved_deblur_network = torch.load(args.deblur_path, map_location=device)
    deblur_network.load_state_dict(saved_deblur_network['deblur'])
    deblur_network = deblur_network.eval()

    kim_examples = generate_full_kim_examples(
        linear_decoder,
        hpf_decoder_nn_model,
        deblur_network,
        test_heldout_frames,
        kim_decoder_response_vector,
        device
    )

    with open(args.image_save_path, 'wb') as pfile:
        save_data = {
            'ground_truth': test_heldout_frames,
            'kim_post_deblur': kim_examples,
        }

        pickle.dump(save_data, pfile)

    print('done')
