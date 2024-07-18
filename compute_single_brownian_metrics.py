import numpy as np
import torch

import argparse
import pickle

from eval_fns.eval import batched_compute_convolutional_masked_ms_ssim, Masked_MS_SSIM, \
    batched_computed_convolutional_masked_LPIPS
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.dataset_config_parser.dataset_config_parser as dcp
import lib.data_utils.data_util as du
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask

import lpips


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute image quality metrics for each eye movements model set')
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

    print("Loading reconstructions")
    with open(args.recons, 'rb') as pfile:
        recons_examples = pickle.load(pfile)

    gt = recons_examples['ground_truth']
    to_test = recons_examples[args.key]

    new_ms_ssim_module = Masked_MS_SSIM(
        ~convex_hull_mask_matrix_bool,
        np.array([0.07105472, 0.45297383, 0.47597145], dtype=np.float32),
        device,
        channel=1,
        data_range=1.0,
        win_size=9,
        size_average=False
    )

    print("Computing MS-SSIM")
    known_eye_movements_mssim = batched_compute_convolutional_masked_ms_ssim(
        gt,
        to_test,
        new_ms_ssim_module,
        5,
        device,
        batch_size=8
    )

    loss_fn_vgg = lpips.LPIPS(net='vgg', spatial=True).to(device)

    print("Computing LPIPS")
    known_eye_movements_lpips = batched_computed_convolutional_masked_LPIPS(
        gt,
        to_test,
        convex_hull_mask_matrix_bool.astype(np.float32),
        loss_fn_vgg,
        4,
        device,
        batch_size=4
    )

    print("Writing results to disk")
    with open(args.output_path, 'wb') as pfile:
        pickle.dump({
            'ms-ssim': known_eye_movements_mssim,
            'lpips': known_eye_movements_lpips
        }, pfile)
