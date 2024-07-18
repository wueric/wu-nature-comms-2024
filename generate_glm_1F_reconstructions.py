import argparse
import pickle
from collections import namedtuple
from typing import List

import numpy as np
import torch
import tqdm

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from convex_optim_base.unconstrained_optim import FistaSolverParams
from convex_optim_base.unconstrained_optim import batch_parallel_unconstrained_solve
from denoise_inverse_alg.glm_inverse_alg import BatchParallelPatchGaussian1FPriorGLMReconstruction, \
    reinflate_cropped_glm_model
from denoise_inverse_alg.glm_inverse_alg import PackedGLMTensors, \
    make_full_res_packed_glm_tensors, noreduce_bernoulli_neg_LL
from generate_cropped_glm_hqs_reconstructions import make_glm_stim_time_component
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from optimization_encoder.trial_glm import load_fitted_glm_families
from simple_priors.gaussian_prior import make_zca_gaussian_prior_matrix

Gaussian1FHyperparameters = namedtuple('Gaussian1FHyperparameters',
                                       ['prior_weight', 'patch_height', 'patch_width'])


def batch_parallel_generate_1F_reconstructions(
        example_spikes: np.ndarray,
        packed_glm_tensors: PackedGLMTensors,
        glm_time_component: np.ndarray,
        prior_zca_matrix_imshape: np.ndarray,
        prior_lambda_weight: float,
        max_batch_size: int,
        device: torch.device,
        patch_stride: int = 2) -> np.ndarray:
    '''

    :param example_spikes:
    :param packed_glm_tensors:
    :param glm_time_component:
    :param prior_zca_matrix_imshape: shape (patch_height, patch_width, patch_height, patch_width)
    :param prior_lambda_weight:
    :param max_batch_size:
    :param device:
    :return:
    '''

    n_examples, n_cells, n_bins_observed = example_spikes.shape
    _, height, width = packed_glm_tensors.spatial_filters.shape

    example_spikes_torch = torch.tensor(example_spikes, dtype=torch.float32, device=device)

    batch_gaussian_problem = BatchParallelPatchGaussian1FPriorGLMReconstruction(
        max_batch_size,
        prior_zca_matrix_imshape,
        prior_lambda_weight,
        packed_glm_tensors,
        glm_time_component,
        noreduce_bernoulli_neg_LL,
        patch_stride=patch_stride
    ).to(device)

    # run the first N-1 iterations
    output_image_buffer_np = np.zeros((n_examples, height, width), dtype=np.float32)
    pbar = tqdm.tqdm(total=n_examples)
    for low in range(0, n_examples - max_batch_size, max_batch_size):
        high = low + max_batch_size

        glm_trial_spikes_torch = example_spikes_torch[low:high, ...]
        batch_gaussian_problem.precompute_gensig_components(glm_trial_spikes_torch)

        _ = batch_parallel_unconstrained_solve(
            batch_gaussian_problem,
            FistaSolverParams(
                initial_learning_rate=1.0,
                max_iter=250,
                converge_epsilon=1e-6,
                backtracking_beta=0.5
            ),
            verbose=False,
            observed_spikes=glm_trial_spikes_torch,
        )

        reconstructed_image = batch_gaussian_problem.get_reconstructed_image()
        output_image_buffer_np[low:high, :, :] = reconstructed_image

        pbar.update(max_batch_size)

    del batch_gaussian_problem

    # run the final iteration
    low = ((n_examples - 1) // max_batch_size) * max_batch_size
    high = n_examples

    eff_batch_size = high - low

    batch_gaussian_problem = BatchParallelPatchGaussian1FPriorGLMReconstruction(
        eff_batch_size,
        prior_zca_matrix_imshape,
        prior_lambda_weight,
        packed_glm_tensors,
        glm_time_component,
        noreduce_bernoulli_neg_LL,
        patch_stride=patch_stride
    ).to(device)

    glm_trial_spikes_torch = example_spikes_torch[low:high, ...]
    batch_gaussian_problem.precompute_gensig_components(glm_trial_spikes_torch)

    _ = batch_parallel_unconstrained_solve(
        batch_gaussian_problem,
        FistaSolverParams(
            initial_learning_rate=1.0,
            max_iter=250,
            converge_epsilon=1e-6,
            backtracking_beta=0.5
        ),
        verbose=False,
        observed_spikes=glm_trial_spikes_torch,
    )

    reconstructed_image = batch_gaussian_problem.get_reconstructed_image()
    output_image_buffer_np[low:high, :, :] = reconstructed_image

    pbar.update(eff_batch_size)
    pbar.close()

    del batch_gaussian_problem
    return output_image_buffer_np


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Generate flashed image reconstructions using LNBRC encoding and dCNN image prior.")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('yaml_path', type=str, help='path to YAML specifying encoding model weights')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-lam', '--prior_lambda', type=float, default=0.2, help='HQS prior weight')
    parser.add_argument('-b', '--batch', type=int, default=8, help='maximum batch size')
    parser.add_argument('-r', '--full_resolution', action='store_true', default=False,
                        help='Set if the GLMs pointed to by the .yaml file are full resolution')
    parser.add_argument('-hh', '--heldout', action='store_true', default=False, help='generate heldout images')
    parser.add_argument('-d', '--patchdim', type=int, default=64, help='Gaussian prior patch dimension')
    parser.add_argument('-s', '--patchstride', type=int, default=1, help='Gaussian prior patch dimension')
    args = parser.parse_args()

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

    bin_width_time_ms = int(np.around(samples_per_bin / 20, decimals=0))
    stimulus_onset_time_length = int(np.around(100 / bin_width_time_ms, decimals=0))

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
    test_heldout_nscenes_patch_list = ddu.preload_bind_get_flashed_patches(
        nscenes_dset_list,
        ddu.PartitionType.HELDOUT_PARTITION if args.heldout else ddu.PartitionType.TEST_PARTITION
    )

    test_heldout_frames = image_rescale_lambda(ddu.concatenate_frames_from_flashed_patches(
        test_heldout_nscenes_patch_list))

    _, height, width = test_heldout_frames.shape

    glm_stim_time_component = make_glm_stim_time_component(config_settings)

    #######################################################
    fitted_glm_paths = parse_prefit_glm_paths(args.yaml_path)
    fitted_glm_families = load_fitted_glm_families(fitted_glm_paths)

    if not args.full_resolution:

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
    else:
        packed_glm_tensors = make_full_res_packed_glm_tensors(
            cells_ordered,
            fitted_glm_families
        )

    # we only do reconstructions from single-bin data here
    # use a separate script to do GLM reconstructions once we get those going
    response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        test_heldout_nscenes_patch_list,
    )

    gaussian_zca_mat = make_zca_gaussian_prior_matrix((args.patchdim, args.patchdim),
                                                      dc_multiple=1.0)
    gaussian_zca_mat_imshape = (gaussian_zca_mat.reshape((args.patchdim, args.patchdim,
                                                          args.patchdim, args.patchdim)))

    target_reconstructions = batch_parallel_generate_1F_reconstructions(
        response_vector,
        packed_glm_tensors,
        glm_stim_time_component,
        gaussian_zca_mat_imshape,
        args.prior_lambda,
        args.batch,
        device,
        patch_stride=args.patchstride
    )

    with open(args.save_path, 'wb') as pfile:
        save_data = {
            'ground_truth': test_heldout_frames,
            '1/F': target_reconstructions
        }

        pickle.dump(save_data, pfile)

    print('done')
