import argparse
import pickle
from typing import List, Tuple

import numpy as np
import torch

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from autoencoder.autoencoder import Neurips2017_Decoder, Neurips2017_Encoder
from lib.dataset_config_parser.dataset_config_parser import read_config_file


def generate_autoencoder_images(linear_reconstructor,
                                encoder: Neurips2017_Encoder,
                                decoder: Neurips2017_Decoder,
                                ground_truth_images: np.ndarray,
                                observed_spikes: np.ndarray,
                                device: torch.device,
                                batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:

    n_tot_images, height, width = ground_truth_images.shape
    print(n_tot_images)

    generated_image_buffer = np.zeros((n_tot_images, height, width), dtype=np.float32)
    generated_linear_image_buffer = np.zeros((n_tot_images, height, width), dtype=np.float32)
    for low in range(0, n_tot_images, batch_size):
        high = min(low + batch_size, n_tot_images)
        print(low, high)

        with torch.no_grad():
            # shape (batch, n_cells)
            batched_spikes_torch = torch.tensor(observed_spikes[low:high, ...],
                                                dtype=torch.float32, device=device)

            # shape (batch, height, width)
            linear_reconstructed = linear_reconstructor(batched_spikes_torch)

            # shape (batch, ?, ??)
            encoded_images = encoder(linear_reconstructed)

            # shape (batch, height, width)
            decoded_images_np = decoder(encoded_images).detach().cpu().numpy()

            generated_image_buffer[low:high,...] = decoded_images_np
            generated_linear_image_buffer[low:high,...] = linear_reconstructed.detach().cpu().numpy()

    return generated_image_buffer, generated_linear_image_buffer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Generate reconstructed images using L-CAE benchmark; config must be single-bin binom, not GLM')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('pretrained_linear_model', type=str, help='path to pickled linear model .pth')
    parser.add_argument('autoencoder_save_path', type=str, help='path to save trained autoencoder')
    parser.add_argument('image_save_path', type=str, help='path to reconstructed image output')
    parser.add_argument('-hh', '--heldout', action='store_true', default=False, help='generate heldout images')

    args = parser.parse_args()

    device = torch.device('cuda')

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
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

    #################################################################
    # Load the raw data
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

    n_test_frames = test_heldout_frames.shape[0]
    _, height, width = test_heldout_frames.shape

    # we only do reconstructions from single-bin data here
    # use a separate script to do GLM reconstructions once we get those going
    response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        test_heldout_flashed_patches,
    ).squeeze(2)

    ####### Load the pretrained linear model ###################################
    linear_decoder = torch.load(args.pretrained_linear_model, map_location=device)

    ####### Construct the encoder and decoder models ###########################
    ####### Load pretrained weights ############################################
    encoder = Neurips2017_Encoder(0.25).to(device)
    decoder = Neurips2017_Decoder(0.25).to(device)

    saved_autoencoder = torch.load(args.autoencoder_save_path, map_location=device)
    encoder.load_state_dict(saved_autoencoder['encoder'])
    decoder.load_state_dict(saved_autoencoder['decoder'])

    encoder.eval()
    decoder.eval()

    ######## Generate the autoencoder reconstructions ########################
    autoencoder_reconstructed_images, linear_reconstructed_images = generate_autoencoder_images(
        linear_decoder,
        encoder,
        decoder,
        test_heldout_frames,
        response_vector,
        device,
        batch_size=50
    )

    with open(args.image_save_path, 'wb') as pfile:
        save_data = {
            'ground_truth': test_heldout_frames,
            'autoencoder': autoencoder_reconstructed_images,
            'linear' : linear_reconstructed_images
        }

        pickle.dump(save_data, pfile)

    print('done')

