import argparse
import pickle

import numpy as np
import torch

import lib.data_utils.data_util as du
import lib.dataset_config_parser.dataset_config_parser as dcp
from eval_fns.eval import  MaskedMSELoss, apply_masked_mse
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute mean square error for flashed reconstructions')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('recons', type=str, help='path to recons file')
    parser.add_argument('key', type=str,
                        help='key to fetch out of reconstruction dict')
    parser.add_argument('output_path', type=str, help='path to save metrics dict')

    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.config)

    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct

    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)
        blurred_stas_by_type = pickle.load(pfile)

    #### Now load the natural scenes dataset #################################
    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]
    nscenes_downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

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

    print("Loading static reconstructions")
    with open(args.recons, 'rb') as pfile:
        static_reconstructions = pickle.load(pfile)

    gt = static_reconstructions['ground_truth']
    recons = static_reconstructions[args.key]

    gt_rescaled = np.clip((gt + 1.0) / 2.0, a_min=0.0, a_max=1.0)
    recons_rescaled = np.clip((recons + 1.0) / 2.0, a_min=0.0, a_max=1.0)

    new_mse_module = MaskedMSELoss(
        np.array(convex_hull_mask_matrix_bool,
                 dtype=np.float32),
        dtype=torch.float32
    ).to(device)

    print("Computing static MSE")

    static_mse = apply_masked_mse(
        new_mse_module,
        gt_rescaled,
        recons_rescaled,
        32,
        device,
    )

    print("Computing static PSNR")

    psnr = 10 * np.log10(1.0 / static_mse)

    print(f'static MS-SSIM {np.mean(static_mse)}, PSNR {np.mean(psnr)}')

    print("Writing results to disk")
    with open(args.output_path, 'wb') as pfile:
        pickle.dump({
            'mse': static_mse,
            'psnr': psnr,
        }, pfile)
