import argparse
import pickle
from typing import List

import numpy as np

import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Computes separate low and high pass partitions of flashed images for the Kim et al ANN benchmark method')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('lpf_output_path', type=str, help='path to pickled linear model coefficients')
    parser.add_argument('hpf_output_path', type=str, help='path to pickled linear model coefficients')
    parser.add_argument('-s', '--sigma', type=float, default=4.0, help='blur sigma, units pixels')
    parser.add_argument('-w', '--window', type=int, default=25, help='blur window, units pixels, must be integer')
    args = parser.parse_args()

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
    train_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                 ddu.PartitionType.TRAIN_PARTITION)

    test_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                ddu.PartitionType.TEST_PARTITION)

    heldout_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                   ddu.PartitionType.HELDOUT_PARTITION)

    train_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(train_flashed_patches))
    test_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(test_flashed_patches))
    heldout_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(heldout_flashed_patches))

    sigma = args.sigma
    window = args.window

    gaussian_blur_filter = du.matlab_style_gauss2D((window, window), sigma=sigma)

    print("Blurring train")
    blurred_images_train = du.batch_blur_image(train_frames,
                                               gaussian_blur_filter)

    print("Blurring test")
    blurred_images_test = du.batch_blur_image(test_frames,
                                              gaussian_blur_filter)

    print("Blurring heldout")
    blurred_images_heldout = du.batch_blur_image(heldout_frames,
                                                 gaussian_blur_filter)

    print("Computing HPF train")
    hpf_train = train_frames - blurred_images_train

    print("Computing HPF test")
    hpf_test = test_frames - blurred_images_test

    print("Computing HPF heldout")
    hpf_heldout = heldout_frames - blurred_images_heldout

    print("Writing to disk")
    with open(args.lpf_output_path, 'wb') as lpf_pfile:
        pickle.dump({
            'train': blurred_images_train,
            'test': blurred_images_test,
            'heldout': blurred_images_heldout,
        }, lpf_pfile)

    with open(args.hpf_output_path, 'wb') as lpf_pfile:
        pickle.dump({
            'train': hpf_train,
            'test': hpf_test,
            'heldout': hpf_heldout,
        }, lpf_pfile)
