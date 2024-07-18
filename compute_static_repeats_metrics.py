import numpy as np
import torch

import pickle

import lpips

from eval_fns.eval import Masked_MS_SSIM, compute_convolutional_max_metric, compute_convolutional_min_LPIPS
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Compute reconstruction quality for flashed repeats")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('repeats_reconstruction', type=str, help='path to repeats reconstruction')
    parser.add_argument('save_path', type=str, help='path to save reconstruction quality metrics')

    args = parser.parse_args()

    with open(args.repeats_reconstruction, 'rb') as pfile:
        repeats_recons_dict = pickle.load(pfile)

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)
    reference_piece_path = config_settings['ReferenceDataset'].path

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

    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)
        blurred_stas_by_type = pickle.load(pfile)

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_metric_rescale_lambda = du.make_torch_transform_to_recons_metric_lambda(image_rescale_low,
                                                                                  image_rescale_high)

    ref_lookup_key = dcp.awsify_piece_name_and_datarun_lookup_key(config_settings['ReferenceDataset'].path,
                                                                  config_settings['ReferenceDataset'].name)

    valid_mask = make_sig_stixel_loss_mask(
        ref_lookup_key,
        blurred_stas_by_type,
        crop_wlow=crop_width_low,
        crop_whigh=crop_width_high,
        crop_hlow=crop_height_low,
        crop_hhigh=crop_height_high,
        downsample_factor=nscenes_downsample_factor
    )

    convex_hull_mask_matrix_bool = compute_convex_hull_of_mask(valid_mask, shrinkage_factor=1.0)
    inverse_mask = ~convex_hull_mask_matrix_bool

    new_mask_float = convex_hull_mask_matrix_bool.astype(np.float32)

    new_ms_ssim_module = Masked_MS_SSIM(
        inverse_mask,
        np.array([0.07105472, 0.45297383, 0.47597145], dtype=np.float32),
        device,
        channel=1,
        data_range=1.0,
        win_size=9,
        size_average=False
    )

    loss_fn_vgg = lpips.LPIPS(net='vgg', spatial=True).to(device)

    with torch.no_grad():
        ground_truth_torch = image_metric_rescale_lambda(
            torch.tensor(repeats_recons_dict['ground_truth'], device=device, dtype=torch.float32)
        )
        data_recons_images_torch = image_metric_rescale_lambda(
            torch.tensor(repeats_recons_dict['repeats'], device=device, dtype=torch.float32)
        )

        n_repeat_trials, n_stimuli, height, width = data_recons_images_torch.shape

        ground_truth_torch_exp_flat = ground_truth_torch[None, :, :, :].expand(n_repeat_trials, -1, -1, -1) \
            .reshape(-1, height, width)
        data_torch_exp_flat = data_recons_images_torch.reshape(-1, height, width)

        ms_ssim_val_data_repeats = compute_convolutional_max_metric(
            new_ms_ssim_module,
            ground_truth_torch_exp_flat,
            data_torch_exp_flat,
            4,
            device,
            batch_size=16
        ).detach().cpu().numpy()
        ms_ssim_val_data_repeats = ms_ssim_val_data_repeats.reshape(n_repeat_trials, n_stimuli)

        lpips_val = compute_convolutional_min_LPIPS(
            loss_fn_vgg,
            ground_truth_torch_exp_flat,
            data_torch_exp_flat,
            torch.tensor(new_mask_float, dtype=torch.float32, device=device),
            4,
            device,
            batch_size=1
        ).detach().cpu().numpy()
        lpips_val = lpips_val.reshape(n_repeat_trials,  n_stimuli)

    with open(args.save_path, 'wb') as pfile:
        pickle.dump({
            'ms_ssim': ms_ssim_val_data_repeats,
            'lpips': lpips_val
        }, pfile)
