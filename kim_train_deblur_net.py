import argparse
import os
import pickle
from typing import List, Callable

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as torch_optim
import torch.utils.data as torch_data

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu

from eval_fns.perceptual_loss import VGG

from kim_et_al_networks.ns_decoder import Parallel_NN_Decoder, train_parallel_NN_decoder
from kim_et_al_networks.nn_deblur import ResnetGenerator


class ThinDatasetWrapper(torch_data.Dataset):

    def __init__(self, images: np.ndarray, spikes: np.ndarray):
        self.images = images
        self.spikes = spikes

    def __getitem__(self, index):
        return self.images[index, ...], self.spikes[index, ...]

    def __len__(self):
        return self.images.shape[0]


def make_mse_vgg_loss(vgg_weight: float,
                      device: torch.device) \
        -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    vgg_network_loss = VGG(conv_index='22').to(device)
    l1_loss = nn.L1Loss()

    def loss_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mse_component = l1_loss(a, b)
        vgg_component = vgg_network_loss(a[:, None, :, :].expand(-1, 3, -1, -1), b[:, None, :, :].expand(-1, 3, -1, -1))

        return mse_component + vgg_weight * vgg_component

    return loss_fn


def plot_deblur_examples(batch_ground_truth: np.ndarray,
                         batch_lpf_output: np.ndarray,
                         batch_hpf_output: np.ndarray,
                         batch_deblur_input: np.ndarray,
                         batch_deblur_output: np.ndarray):
    '''
    For every image in the example batch,
    :param forward_intermediates:
    :param linear_model:
    :param batched_observed_spikes:
    :return:
    '''

    batch_size = batch_ground_truth.shape[0]

    fig, axes = plt.subplots(batch_size, 5, figsize=(5 * 5, 5 * batch_size))
    for row in range(batch_size):
        ax = axes[row, 0]
        ax.imshow(batch_ground_truth[row, ...], vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

        ax = axes[row, 1]
        ax.imshow(batch_lpf_output[row, ...], vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

        ax = axes[row, 2]
        ax.imshow(batch_hpf_output[row, ...], vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

        ax = axes[row, 3]
        ax.imshow(batch_deblur_input[row, ...], vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

        ax = axes[row, 4]
        ax.imshow(batch_deblur_output[row, ...], vmin=-1.0, vmax=1.0, cmap='gray')
        ax.axis('off')

    return fig


def eval_test_loss_deblur(linear_lpf_decoder: nn.Module,
                          hpf_decoder: nn.Module,
                          deblur_network: nn.Module,
                          test_loader: torch_data.DataLoader,
                          loss_callable: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                          device: torch.device):
    loss_acc = []
    with torch.no_grad():
        for it, (test_frames, spikes_np) in enumerate(test_loader):
            frames_torch = torch.tensor(test_frames, dtype=torch.float32, device=device)
            spikes_torch = torch.tensor(spikes_np, dtype=torch.float32, device=device)

            height, width = frames_torch.shape[1:]

            # shape (batch, n_cells)
            spikes_onebin_torch = torch.sum(spikes_torch, dim=2)

            linear_reconstructed = linear_lpf_decoder(spikes_onebin_torch)
            hpf_residual = hpf_decoder(spikes_torch).reshape(-1, height, width)

            deblur_input = linear_reconstructed + hpf_residual

            deblur_output = deblur_network(deblur_input[:, None, :, :]).squeeze(1)
            loss = loss_callable(frames_torch, deblur_output)

            loss_acc.append(loss.item())

    return np.mean(loss_acc)


def train_vgg_deblur_network(linear_lpf_decoder: nn.Module,
                             hpf_decoder: nn.Module,
                             deblur_network: nn.Module,
                             training_loader: torch_data.DataLoader,
                             test_loader: torch_data.DataLoader,
                             loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                             device: torch.device,
                             summary_writer,
                             save_path_base: str,
                             learning_rate: float = 1e-4,
                             n_epochs: int = 16):
    optimizer = torch_optim.Adam(deblur_network.parameters(),
                                 lr=learning_rate)

    n_steps_per_epoch = len(training_loader)

    for ep in range(n_epochs):

        for it, (train_frames, train_spikes) in enumerate(training_loader):

            train_frame_torch = torch.tensor(train_frames, dtype=torch.float32, device=device)
            height, width = train_frame_torch.shape[1:]

            # shape (batch, n_cells, n_bins)
            train_spikes_torch = torch.tensor(train_spikes, dtype=torch.float32, device=device)

            with torch.no_grad():

                # shape (batch, n_cells)
                train_spikes_onebin_torch = torch.sum(train_spikes_torch, dim=2)

                linear_reconstructed = linear_lpf_decoder(train_spikes_onebin_torch)
                hpf_residual = hpf_decoder(train_spikes_torch).reshape(-1, height, width)

                deblur_input = linear_reconstructed + hpf_residual

            optimizer.zero_grad()

            deblur_output = deblur_network(deblur_input[:, None, :, :]).squeeze(1)
            loss = loss_fn(train_frame_torch, deblur_output)

            loss.backward()
            optimizer.step()

            # log stuff out to Tensorboard
            # loss is updated every step
            summary_writer.add_scalar('training loss', loss.item(), ep * n_steps_per_epoch + it)

            if it % 16 == 0:
                ex_fig = plot_deblur_examples(train_frames,
                                              linear_reconstructed.detach().cpu().numpy(),
                                              hpf_residual.detach().cpu().numpy(),
                                              deblur_input.detach().cpu().numpy(),
                                              deblur_output.detach().cpu().numpy())
                summary_writer.add_figure('training example images',
                                          ex_fig,
                                          global_step=ep * n_steps_per_epoch + it)

        test_loss = eval_test_loss_deblur(linear_decoder,
                                          hpf_decoder,
                                          deblur_network,
                                          test_loader,
                                          loss_fn,
                                          device)

        # log stuff out to Tensorboard
        # loss is updated every step
        summary_writer.add_scalar('test loss ', test_loss, (ep + 1) * n_steps_per_epoch)

        save_path = os.path.join(save_path_base, f"deblur_network_epoch{ep}.pth")
        torch.save({'deblur': deblur_network.state_dict()}, save_path)

    return deblur_network


from linear_decoding_models.linear_decoding_models import ClosedFormLinearModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Fit Kim-style deblur network, assuming HP NN and linear model already trained')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('coeffs_file_path', type=str, help='path to pickled linear model coefficients')
    parser.add_argument('lpf_linear_decoder_path', type=str, help='path to pickled linear model coefficients')
    parser.add_argument('hpf_nn_decoder_path', type=str, help='path to save trained HPF decoder')
    parser.add_argument('deblur_path', type=str, help='path to save deblur network')
    parser.add_argument('tboard', type=str, help='path to Tensorboard output')
    parser.add_argument('-k', '--k_dim', type=int, default=25, help='Number of cells per pixel')
    parser.add_argument('-dh', '--h_dim', type=int, default=40, help='Number of cells per pixel')
    parser.add_argument('-f', '--f_dim', type=int, default=5, help='Number of cells per pixel')
    parser.add_argument('-e', '--n_epochs', type=int, default=16, help='Number of training epochs')
    parser.add_argument('-b', '--batch', type=int, default=16, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-p', '--percept_loss', type=float, default=1e-2, help='')
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
    train_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                 ddu.PartitionType.TRAIN_PARTITION)

    test_flashed_patches = ddu.preload_bind_get_flashed_patches(nscenes_dset_list,
                                                                 ddu.PartitionType.TEST_PARTITION)


    train_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(train_flashed_patches))
    test_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(test_flashed_patches))

    n_train_frames, n_test_frames = train_frames.shape[0], test_frames.shape[0]
    height, width = train_frames.shape[1:]

    ####### Bin spikes ########################################################3
    print('Binning spikes')
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

    train_dataset = ThinDatasetWrapper(train_frames,
                                       train_kim_decoder_response_vector)
    test_dataset = ThinDatasetWrapper(test_frames,
                                      test_kim_decoder_response_vector)

    train_dataloader = torch_data.DataLoader(train_dataset,
                                             batch_size=args.batch,
                                             shuffle=True,
                                             drop_last=True)
    test_dataloader = torch_data.DataLoader(test_dataset,
                                            batch_size=args.batch,
                                            shuffle=True,
                                            drop_last=True)

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
                                               train_kim_decoder_response_vector.shape[1],
                                               train_kim_decoder_response_vector.shape[2],
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

    summary_writer_path = os.path.join(args.tboard, 'tensorboard')
    summary_writer = SummaryWriter(summary_writer_path)

    loss_with_perceptual = make_mse_vgg_loss(args.percept_loss, device)

    print("Begin training")
    trained_deblur_network = train_vgg_deblur_network(
        linear_decoder,
        decoder_nn_model,
        deblur_network,
        train_dataloader,
        test_dataloader,
        loss_with_perceptual,
        device,
        summary_writer,
        args.deblur_path,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs
    )
