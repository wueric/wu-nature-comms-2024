import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as torch_optim
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
from typing import List, Tuple, Callable
import pickle

from imagenet_dset.imagenet_dset import ImagenetMaskedDataloader
from gaussian_denoiser.dpir_models.models.network_unet import UNetRes
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask

#PATHS = [
#    ('/Volumes/Lab/Users/ericwu/yass-reconstruction/2018-08-07-5/data000',
#     '/Volumes/Lab/Users/ericwu/sample-reconstruction-data/2018-08-07-5/recurated_files/bigger_crop_bbox_with_midgets.pickle'),
#    ('/Volumes/Lab/Users/ericwu/yass-reconstruction/2017-11-29-0/analysis/data001',
#     '/Volumes/Lab/Users/ericwu/sample-reconstruction-data/2017-12-04-5/full_res_glm_bbox_with_midgets.pickle')
#]

PATHS = {
    ('2018-08-07-5', 'data000') : \
        '/Volumes/Lab/Users/ericwu/sample-reconstruction-data/2018-08-07-5/recurated_files/bigger_crop_bbox_with_midgets.pickle',
}

def compute_hardcoded_masks():

    masks_to_return = {}
    for lookup_key, blurred_bbox_sta_path in PATHS.items():

        with open(blurred_bbox_sta_path, 'rb') as pfile:
            _ = pickle.load(pfile)
            blurred_stas_by_type = pickle.load(pfile)

        valid_mask = make_sig_stixel_loss_mask(
            lookup_key,
            blurred_stas_by_type,
            crop_wlow=32,
            crop_whigh=32,
            crop_hlow=0,
            crop_hhigh=0,
            downsample_factor=1
        )

        convex_hull_valid_mask = compute_convex_hull_of_mask(valid_mask)

        masks_to_return[lookup_key] = convex_hull_valid_mask.astype(np.float32)

    return masks_to_return


PRECOMPUTED_MASKS = compute_hardcoded_masks()


@torch.jit.script
def post_masked_mse_loss(eval_input: torch.Tensor,
                         ground_truth: torch.Tensor,
                         mask: torch.Tensor):
    diff = eval_input - ground_truth
    square_diff = torch.sum(diff * diff * mask, dim=(1, 2))
    n_tot = torch.sum(mask, dim=(1, 2))
    return torch.mean(square_diff / n_tot)


@torch.jit.script
def post_masked_l1_loss(eval_input: torch.Tensor,
                        ground_truth: torch.Tensor,
                        mask: torch.Tensor):
    abs_diff = torch.abs(eval_input - ground_truth)
    mask_l1_diff = torch.sum(abs_diff * mask, dim=(1, 2))
    n_tot = torch.sum(mask, dim=(1, 2))
    return torch.mean(mask_l1_diff / n_tot)


def plot_denoiser_examples(batched_ground_truth: np.ndarray,
                           batched_denoiser_input: np.ndarray,
                           batched_denoiser_output: np.ndarray):
    batch_size = batched_ground_truth.shape[0]

    fig, axes = plt.subplots(batch_size, 3, figsize=(3 * 5, 5 * batch_size))
    for row in range(batch_size):
        ax = axes[row, 0]
        ax.imshow(batched_ground_truth[row, ...], vmin=0, vmax=255, cmap='gray')
        ax.axis('off')

        ax = axes[row, 1]
        ax.imshow(batched_denoiser_input[row, ...], vmin=0, vmax=255, cmap='gray')
        ax.axis('off')

        ax = axes[row, 2]
        ax.imshow(batched_denoiser_output[row, ...], vmin=0, vmax=255, cmap='gray')
        ax.axis('off')

    return fig


def eval_test_loss_denoiser(model: UNetRes,
                            test_dataloader: torch_data.DataLoader,
                            loss_callable: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                            device: torch.device) -> float:
    loss_acc = []
    with torch.no_grad():
        for it, (nn_input, target, mask) in enumerate(test_dataloader):
            # shape (batch, 3, height, width)
            nn_input_torch = torch.tensor(nn_input, dtype=torch.float32, device=device)

            target_torch = torch.tensor(target, dtype=torch.float32, device=device)
            mask_torch = torch.tensor(mask, dtype=torch.float32, device=device)

            output_flat = model(nn_input_torch)
            loss = loss_callable(output_flat, target_torch, mask_torch).detach().cpu().numpy()

            loss_acc.append(np.mean(loss))
    return np.mean(loss_acc)


def train_masked_denoiser(model: UNetRes,
                          train_dataloader: torch_data.DataLoader,
                          #test_dataloader: torch_data.DataLoader,
                          loss_callable: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                          device: torch.device,
                          summary_writer,
                          save_path: str,
                          init_learning_rate: float = 1e-1,
                          learning_rate_decay: Tuple[float, int] = (0.5, 100000),
                          n_epochs: int = 2,
                          print_per_steps: int = 16) -> UNetRes:
    lr_decay, lr_change_nsteps = learning_rate_decay

    optimizer = torch_optim.Adam(model.parameters(),
                                 lr=init_learning_rate)
    scheduler = torch_optim.lr_scheduler.StepLR(optimizer,
                                                step_size=lr_change_nsteps,
                                                gamma=lr_decay)

    n_steps_per_epoch = len(train_dataloader)
    try:
        for ep in range(n_epochs):
            for it, (nn_input, target, mask) in enumerate(train_dataloader):

                input_torch = torch.tensor(nn_input, dtype=torch.float32, device=device)
                batch, _, height, width = input_torch.shape

                target_torch = torch.tensor(target, dtype=torch.float32, device=device)
                mask_torch = torch.tensor(mask, dtype=torch.float32, device=device)

                optimizer.zero_grad()
                nn_denoised = model(input_torch).squeeze(1)
                loss = loss_callable(nn_denoised, target_torch, mask_torch)

                loss.backward()
                optimizer.step()
                scheduler.step()

                # log to Tensorboard
                if it % print_per_steps == 0:
                    ex_fig = plot_denoiser_examples(target, nn_input[:, 0, ...],
                                                    nn_denoised.detach().cpu().numpy())
                    summary_writer.add_figure('training example images',
                                              ex_fig,
                                              global_step=ep * n_steps_per_epoch + it)

                    summary_writer.add_scalar('train_loss', loss.item(), (it + ep * n_steps_per_epoch))

            print("Saving refitted denoiser")
            torch.save({'denoiser': model.state_dict()},
                       f'{args.save_path}_epoch{ep}.pth')

    #        test_loss = eval_test_loss_denoiser(model,
    #                                            test_dataloader,
    #                                            loss_callable,
    #                                            device)

            # log stuff out to Tensorboard
            # loss is updated every step
            #summary_writer.add_scalar('test loss ', test_loss, (ep + 1) * n_steps_per_epoch)

    except:
        pass

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Retrains DRUnet denoiser CNN with additional channel for valid region mask')
    parser.add_argument('data_input', type=str, help='path to data input')
    #parser.add_argument('test_input', type=str, help='path to data input')
    parser.add_argument('save_path', type=str, help='path to save NN weights')
    parser.add_argument('tboard', type=str, help='path to Tensorboard folder')
    parser.add_argument('-e', '--n_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-b', '--batch', type=int, default=16, help='number of training epochs')
    parser.add_argument('-n', '--noise_levels', type=float, nargs='+',
                        help='Noise levels to use')
    parser.add_argument('-ss', '--step_sched', type=int, default=100000, help='reduce learning rate after N steps')
    parser.add_argument('-sr', '--sched_reduce', type=float, default=0.5, help='reduce learning rate by M after N steps')
    parser.add_argument('-l', '--use_l1_loss', action='store_true', default=False,
                        help='Use L1 loss instead of L2 loss')

    args = parser.parse_args()

    device = torch.device('cuda')

    #################################################
    # Build the model
    N_CHANNELS = 1
    model = UNetRes(in_nc=N_CHANNELS + 2, out_nc=N_CHANNELS, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                    downsample_mode="strideconv", upsample_mode="convtranspose").to(device)
    model.train()

    #################################################
    # Set up the dataloader
    train_dataset = ImagenetMaskedDataloader(args.data_input,
                                             args.noise_levels,
                                             list(PRECOMPUTED_MASKS.values()),
                                             augment_masks=True)
    train_dataloader = torch_data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    #################################################
    # Set up tensorboard
    summary_writer_path = os.path.join(args.tboard, 'tensorboard')
    summary_writer = SummaryWriter(summary_writer_path)

    if args.use_l1_loss:
        loss_callable = post_masked_l1_loss
    else:
        loss_callable = post_masked_mse_loss

    print('Begin training')
    trained_model = train_masked_denoiser(model,
                                          train_dataloader,
                                          #test_dataloader,
                                          loss_callable,
                                          device,
                                          summary_writer,
                                          args.save_path,
                                          init_learning_rate=args.learning_rate,
                                          learning_rate_decay=(args.sched_reduce, args.step_sched),
                                          n_epochs=args.n_epochs,
                                          print_per_steps=16)
