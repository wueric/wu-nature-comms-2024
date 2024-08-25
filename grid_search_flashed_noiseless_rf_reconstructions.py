import argparse
import pickle

import numpy as np
import torch

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from denoise_inverse_alg.glm_inverse_alg import reinflate_cropped_glm_model
from eval_fns.eval import MaskedMSELoss, MS_SSIM, SSIM
from flashed_noiseless_rf_reconstructions import retrieve_eye_movements_single_frames, retrieve_flashed_stimulus_frames, \
    apply_linear_projections_gpu, make_linear_noiseless_HQS_X_prob_solver_generator_fn, \
    make_linear_noiseless_HQS_Z_prob_solver_generator_fn
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from optimization_encoder.trial_glm import load_fitted_glm_families
from reconstruction_fns.flashed_reconstruction_toplevel_fns import batch_parallel_noiseless_linear_hqs_grid_search

BATCH_SIZE = 16
HQS_MAX_ITER = 5
GRID_SEARCH_TEST_N_IMAGES = 80

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Grid search flashed image reconstruction hyperparameters, using LNBRC encoding and dCNN image prior")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('yaml_path', type=str, help='path to YAML specifying where the GLM fits are')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-n', '--noise_init', type=float, default=1e-3)
    parser.add_argument('-ls', '--lambda_start', nargs=3,
                        help='Grid search parameters for HQS prox start. Start, end, n_grid')
    parser.add_argument('-le', '--lambda_end', nargs=3,
                        help='Grid search parameters for HQS prox end. Start, end, n_grid')
    parser.add_argument('-p', '--prior', type=float, nargs='+',
                        help='Grid search parameters for prior lambda. Specify weights explicitly')
    parser.add_argument('-it', '--max_iter', type=int, default=HQS_MAX_ITER, help='Number of HQS iterations')
    parser.add_argument('-f', '--fixational', action='store_true', default=False,
                        help='is fixational eye movements stimulus')

    args = parser.parse_args()

    device = torch.device('cuda')

    config_settings = read_config_file(args.cfg_file)

    grid_search_lambda_start = np.logspace(float(args.lambda_start[0]), float(args.lambda_start[1]),
                                           int(args.lambda_start[2]))
    grid_search_lambda_end = np.logspace(float(args.lambda_end[0]), float(args.lambda_end[1]),
                                         int(args.lambda_end[2]))
    grid_search_prior = np.array(args.prior)

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
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

    ################################################################
    # Compute the valid region mask
    print("Computing valid region mask")
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

    convex_hull_valid_mask = compute_convex_hull_of_mask(valid_mask)
    invalid_mask = ~convex_hull_valid_mask

    valid_mask_float_torch = torch.tensor(convex_hull_valid_mask, dtype=torch.float32, device=device)
    inverse_mask_float_torch = torch.tensor(invalid_mask, dtype=torch.float32, device=device)

    #####################################################################
    # load the stimulus
    # different for flashed or jittered
    if args.fixational:
        target_frames = retrieve_eye_movements_single_frames(
            config_settings,
            ddu.PartitionType.TEST_PARTITION,
            cells_ordered
        )
    else:
        target_frames = retrieve_flashed_stimulus_frames(
            config_settings,
            ddu.PartitionType.TEST_PARTITION
        )

    n_images, height, width = target_frames.shape

    #####################################################################
    # get the models, extract the linear filters, renormalize them to 1
    fitted_glm_paths = parse_prefit_glm_paths(args.yaml_path)
    print(fitted_glm_paths)
    fitted_glm_families = load_fitted_glm_families(fitted_glm_paths)

    packed_glm_tensors = reinflate_cropped_glm_model(
        fitted_glm_families,
        bounding_boxes_by_type,
        cells_ordered,
        height,
        width,
        downsample_factor=nscenes_downsample_factor,
        crop_width_low=crop_width_low,
        crop_width_high=crop_width_high,
        crop_height_low=crop_height_low,
        crop_height_high=crop_height_high
    )

    linear_filters = packed_glm_tensors.spatial_filters.reshape(-1, height * width)
    linear_filters = linear_filters / np.linalg.norm(linear_filters, axis=-1, keepdims=True)
    linear_filters = linear_filters.reshape(-1, height, width)

    ################################################################
    # Construct the loss evaluation modules
    masked_mse_module = MaskedMSELoss(convex_hull_valid_mask).to(device)

    ms_ssim_module = MS_SSIM(channel=1,
                             data_range=1.0,
                             win_size=9,
                             weights=[0.07105472, 0.45297383, 0.47597145],
                             not_valid_mask=inverse_mask_float_torch).to(device)

    ssim_module = SSIM(channel=1,
                       data_range=1.0,
                       not_valid_mask=inverse_mask_float_torch).to(device)

    ################################################################
    # Do the grid search
    example_selector = np.random.choice(
            np.r_[0:target_frames.shape[0]],
            replace=False,
            size=GRID_SEARCH_TEST_N_IMAGES)

    grid_search_target_frames = target_frames[example_selector, ...]

    #####################################################################
    # perform the linear projections
    # shape (n_images, n_cells)
    linearly_projected = apply_linear_projections_gpu(linear_filters,
                                                      grid_search_target_frames,
                                                      device)

    grid_search_output_dict = batch_parallel_noiseless_linear_hqs_grid_search(
        linearly_projected,
        grid_search_target_frames,
        (image_rescale_low, image_rescale_high),
        linear_filters,
        BATCH_SIZE,
        masked_mse_module,
        ssim_module,
        ms_ssim_module,
        grid_search_lambda_start,
        grid_search_lambda_end,
        grid_search_prior,
        args.max_iter,
        make_linear_noiseless_HQS_X_prob_solver_generator_fn,
        make_linear_noiseless_HQS_Z_prob_solver_generator_fn,
        device,
        initialize_noise_level=args.noise_init,
        valid_region_mask=convex_hull_valid_mask
    )

    with open(args.save_path, 'wb') as pfile:
        pickle.dump(grid_search_output_dict, pfile)
