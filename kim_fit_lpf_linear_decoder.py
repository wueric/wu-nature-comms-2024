import argparse
import pickle
from typing import List

import numpy as np
import torch

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.dynamic_data_util as ddu

from linear_decoding_models.kim_l1_linear_problem import PixelwiseLinearL1Regression
from convex_optim_base.prox_optim import batch_parallel_prox_solve, ProxFISTASolverParams


L1_REG_PARAM_GRID = np.logspace(np.log10(1e-8), np.log10(1e-3), 11)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Fit Kim-style linear model with L1 regularization from config; config must be single-bin binom, not GLM' + \
        ' Note that we train with respect to the LPF filtered data rather than the full data')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('lpf_path', type=str, help='path to LPF stimuli.')
    parser.add_argument('model_output_path', type=str, help='path to pickled linear model coefficients')
    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    #### Now load the natural scenes parameters #################################
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

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

    train_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                 ddu.PartitionType.TRAIN_PARTITION)

    test_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                ddu.PartitionType.TEST_PARTITION)

    ####### Load info about the cell ids, cell types, and matches #############
    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct
    ct_order = cells_ordered.get_cell_types()

    cell_ids_as_ordered_list = []
    for ct in ct_order:
        cell_ids_as_ordered_list.extend(cells_ordered.get_reference_cell_order(ct))

    train_binom_response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        train_flashed_patches,
    ).squeeze(2)

    test_binom_response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        test_flashed_patches,
    ).squeeze(2)

    with open(args.lpf_path, 'rb') as pfile:
        frames_dict = pickle.load(pfile)
        train_frames = frames_dict['train']
        test_frames = frames_dict['test']

    height, width = train_frames.shape[1:]

    # shape (n_trials, n_cells)
    train_binom_response_torch = torch.tensor(train_binom_response_vector, dtype=torch.float32,
                                              device=device)
    test_binom_response_torch = torch.tensor(test_binom_response_vector, dtype=torch.float32,
                                             device=device)

    # copy the train frames to GPU
    train_frames_torch = torch.tensor(train_frames, dtype=torch.float32,
                                      device=device)
    test_frames_torch = torch.tensor(test_frames, dtype=torch.float32,
                                     device=device)


    # shape (n_pixels, n_trials)
    train_frames_flat_torch = train_frames_torch.reshape(train_frames_torch.shape[0], -1).permute(1, 0)
    test_frames_flat_torch = test_frames_torch.reshape(test_frames_torch.shape[0], -1).permute(1, 0)

    best_l1_reg = None
    best_train_loss, best_test_loss, best_filt_coeffs = float('inf'), float('inf'), None
    for l1_reg in L1_REG_PARAM_GRID:
        l1_optim_problem = PixelwiseLinearL1Regression(
            (height, width),
            len(cell_ids_as_ordered_list),
            l1_reg
        ).to(device)

        loss = batch_parallel_prox_solve(l1_optim_problem,
                                         ProxFISTASolverParams(initial_learning_rate=1.0,
                                                               max_iter=1000,
                                                               converge_epsilon=1e-9),
                                         verbose=True,
                                         images=train_frames_flat_torch,
                                         spikes=train_binom_response_torch.T)

        test_loss = l1_optim_problem.compute_test_loss(test_binom_response_torch.T,
                                                       test_frames_flat_torch).item()

        print(f"L1 reg const {l1_reg}, test_loss {test_loss}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_train_loss = loss.cpu().numpy()
            best_filt_coeffs = l1_optim_problem.get_filter_coeffs_imshape()
            best_l1_reg = l1_reg


    with open(args.model_output_path, 'wb') as pfile:
        pickle.dump({
            'coeffs': best_filt_coeffs,
            'loss': best_train_loss,
            'test_loss': best_test_loss,
            'l1_reg': best_l1_reg
        }, pfile)
