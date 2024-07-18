import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import numpy as np

import argparse
import pickle
import os

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask

from autoencoder.autoencoder import Neurips2017_Decoder, Neurips2017_Encoder, ThinDatasetWrapper, train_autoencoder

from eval_fns.eval import MaskedMSELoss, MaskedSSIM

from typing import Callable, List


def make_ssim_loss_callable(ssim_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    '''
    Computes SSIM loss function value, takes care of range shifting and clipping to make
        sure that the values passed into SSIM are always valid and within the acceptable range
    :param ssim_fn:
    :return:
    '''

    hard_tanh = nn.Hardtanh()

    def wrapped_loss_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X_new_range = (hard_tanh(X[:, None, :, :]) + 1.0) / 2.0
        Y_new_range = (hard_tanh(Y[:, None, :, :]) + 1.0) / 2.0

        return -ssim_fn(X_new_range, Y_new_range)

    return wrapped_loss_fn


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Fit L-CAE benchmark on top of a pretrained linear model from config; config must be single-bin binom')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('pretrained_linear_model', type=str, help='path to pickled linear model .pth')
    parser.add_argument('autoencoder_save_path', type=str, help='path to save trained autoencoder')
    parser.add_argument('tensorboard_path', type=str, help='path to tensorboard output')
    parser.add_argument('-d', '--drop_rate', type=float, default=0.25, help='dropout hyperparameter')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size hyperparameter')
    parser.add_argument('-l', '--learning_rate', type=int, default=4e-3, help='learning rate')
    parser.add_argument('-n', '--n_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('-m', '--max_training_examples', type=int, default=-1,
                        help='specify maximum number of training examples')
    parser.add_argument('-s', '--ssim', action='store_true', default=False, help='Use SSIM loss')
    args = parser.parse_args()

    _max_n_samples = args.max_training_examples
    use_limited_data = (args.max_training_examples != -1)
    if use_limited_data:
        print(f"Using only {_max_n_samples} datapoints")

    use_ssim = args.ssim

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

    #### Load the bounding boxes and the blurred STAs ########################
    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)
        blurred_stas_by_type = pickle.load(pfile)

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

    train_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(train_flashed_patches))
    test_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(test_flashed_patches))

    if use_limited_data:
        train_frames = train_frames[:_max_n_samples, ...]

    n_train_frames, n_test_frames = train_frames.shape[0], test_frames.shape[0]
    height, width = train_frames.shape[1:]

    train_binom_response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        train_flashed_patches,
    ).squeeze(2)

    if use_limited_data:
        train_binom_response_vector = train_binom_response_vector[:_max_n_samples, ...]

    test_binom_response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        test_flashed_patches,
    ).squeeze(2)

    ####### Make torch datasets out of the loaded data #########################
    training_dataset = ThinDatasetWrapper(train_frames,
                                          train_binom_response_vector)
    test_dataset = ThinDatasetWrapper(test_frames,
                                      test_binom_response_vector)

    ####### Make the masked loss function; compute the mask ####################
    ref_lookup_key = dcp.awsify_piece_name_and_datarun_lookup_key(config_settings['ReferenceDataset'].path,
                                                                  config_settings['ReferenceDataset'].name)
    mask_matrix_bool = make_sig_stixel_loss_mask(ref_lookup_key,
                                                 blurred_stas_by_type,
                                                 downsample_factor=nscenes_downsample_factor,
                                                 crop_wlow=crop_width_low,
                                                 crop_whigh=crop_width_high)
    convex_hull_mask_matrix_bool = compute_convex_hull_of_mask(mask_matrix_bool,
                                                               shrinkage_factor=1.0)


    if use_ssim:
        invalid_mask = ~convex_hull_mask_matrix_bool
        differentiable_ssim_module = MaskedSSIM(invalid_mask.astype(np.float32),
                                                device,
                                                channel=1,
                                                data_range=1.0)
        loss_fn = make_ssim_loss_callable(differentiable_ssim_module)

    else:
        loss_fn = MaskedMSELoss(convex_hull_mask_matrix_bool).to(device)

    ####### Load the pretrained linear model ###################################
    linear_decoder = torch.load(args.pretrained_linear_model, map_location=device)

    ####### Construct the encoder and decoder models ###########################
    encoder = Neurips2017_Encoder(args.drop_rate).to(device)
    decoder = Neurips2017_Decoder(args.drop_rate).to(device)

    summary_writer_path = os.path.join(args.tensorboard_path, 'tensorboard')
    summary_writer = SummaryWriter(summary_writer_path)

    trained_encoder, trained_decoder = train_autoencoder(
        training_dataset,
        test_dataset,
        linear_decoder,
        encoder,
        decoder,
        loss_fn,
        args.n_epochs,
        args.batch_size,
        args.learning_rate,
        device,
        summary_writer,
        args.autoencoder_save_path,
        convex_hull_mask_matrix_bool,
        generate_images_nsteps=10
    )
