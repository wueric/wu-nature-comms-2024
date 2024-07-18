import argparse
import pickle
from typing import List

import numpy as np
import torch

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from optimization_encoder.trial_glm import load_fitted_glm_families
from sim_retina.jitter_sim_helper import SingleCellRepeatsJitteredMovieDataloader, \
    ct_full_glm_simulate_single_cell_single_stimulus, \
    ct_uncoupled_glm_simulate_single_cell_single_stimulus
from sim_retina.load_model_for_sim import reinflate_full_glm_model_for_sim, reinflate_uncoupled_glm_model_for_sim
from sim_retina.single_cell_sim import CTSimGLM, CTSimFBOnlyGLM

N_CONSECUTIVE_STIMULI = 10 # 5 seconds, but we have to chop off the first
# 250 us since that is needed to initialize the model

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Simulate rasters for eye movements stimulus")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('yaml_path', type=str, help='path to YAML specifying encoding model weights')
    parser.add_argument('center_cell_id', type=int, help='center cell id to simulate')
    parser.add_argument('stimulus_ix', type=int, help='index of the repeat stimulus we want to start at')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-f', '--feedback_only', action='store_true', default=False,
                        help='GLM model specified by model_cfg_path is feedback-only')

    args = parser.parse_args()

    device = torch.device('cuda')

    ########################################################
    # Load the pretrained models from disk
    glm_parsed_paths = parse_prefit_glm_paths(args.yaml_path)
    glm_families = load_fitted_glm_families(glm_parsed_paths)

    # read the config
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

    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)
        blurred_stas_by_type = pickle.load(pfile)

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

    #################################################################
    # Load the raw data
    create_test_dataset = (dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)
    create_heldout_dataset = (dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR in config_settings)

    test_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]
    heldout_dataset_movie_blocks = []  # type: List[dcp.MovieBlockSectionDescriptor]

    if create_test_dataset:
        test_dataset_movie_blocks = config_settings[dcp.TestMovieSection.MOVIE_BLOCK_DESCRIPTOR]
    if create_heldout_dataset:
        heldout_dataset_movie_blocks = config_settings[dcp.HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR]

    nscenes_dataset_info_list = config_settings[dcp.NScenesMovieDatasetSection.OUTPUT_KEY]

    loaded_synchronized_datasets = ddu.load_jittered_nscenes_dataset_and_timebin(
        nscenes_dataset_info_list,
        test_dataset_movie_blocks,
        heldout_dataset_movie_blocks,
    )

    single_cell_repeats_sim_dataloader = SingleCellRepeatsJitteredMovieDataloader(
        loaded_synchronized_datasets,
        cells_ordered,
        samples_per_bin,
        image_rescale_lambda=image_rescale_lambda,
        n_stimuli_to_get=N_CONSECUTIVE_STIMULI
    )

    center_cell_id = args.center_cell_id
    center_cell_type = cells_ordered.get_cell_type_for_cell_id(center_cell_id)
    reference_order_typed = cells_ordered.get_reference_cell_order(center_cell_type)
    sel_ix = reference_order_typed.index(center_cell_id)

    print(f'Simulating {center_cell_id}')

    # get bounding box, apply to cropped stimulus
    crop_bbox = bounding_boxes_by_type[center_cell_type][sel_ix]
    crop_bounds_h, crop_bounds_w = crop_bbox.make_cropping_sliceobj(
        crop_hlow=crop_height_low,
        crop_hhigh=crop_height_high,
        crop_wlow=crop_width_low,
        crop_whigh=crop_width_high,
        downsample_factor=nscenes_downsample_factor,
        return_bounds=True
    )

    if args.feedback_only:

        uncoupled_glm_sim_parameters = reinflate_uncoupled_glm_model_for_sim(
            glm_families,
            cells_ordered,
            (center_cell_type, center_cell_id)
        )

        uncoupled_sim_glm = CTSimFBOnlyGLM(
            uncoupled_glm_sim_parameters.cropped_spatial_filter.reshape(-1),
            uncoupled_glm_sim_parameters.stimulus_timecourse,
            uncoupled_glm_sim_parameters.feedback_filter,
            uncoupled_glm_sim_parameters.bias,
            lambda x: torch.bernoulli(torch.sigmoid(x)),
            dtype=torch.float32
        ).to(device)

        simulated_spikes_for_cell = ct_uncoupled_glm_simulate_single_cell_single_stimulus(
            single_cell_repeats_sim_dataloader,
            uncoupled_sim_glm,
            args.stimulus_ix,
            center_cell_id,
            (crop_bounds_h, crop_bounds_w),
            device,
            max_batch_size=8
        )

    else:

        full_glm_sim_parameters = reinflate_full_glm_model_for_sim(
            glm_families,
            cells_ordered,
            (center_cell_type, center_cell_id),
            compute_indices=False
        )

        # create simulation model on GPU
        ct_sim_glm = CTSimGLM(
            full_glm_sim_parameters.cropped_spatial_filter.reshape(-1),
            full_glm_sim_parameters.stimulus_timecourse,
            full_glm_sim_parameters.feedback_filter,
            full_glm_sim_parameters.coupling_params[0],
            full_glm_sim_parameters.bias,
            lambda x: torch.bernoulli(torch.sigmoid(x)),
            dtype=torch.float32
        ).to(device)

        simulated_spikes_for_cell = ct_full_glm_simulate_single_cell_single_stimulus(
            single_cell_repeats_sim_dataloader,
            ct_sim_glm,
            args.stimulus_ix,
            center_cell_id,
            list(full_glm_sim_parameters.coupling_params[1]),
            (crop_bounds_h, crop_bounds_w),
            device,
            max_batch_size=8
        )

    # also want to save the ground truth spikes for the center cell for easy plotting
    ground_truth_spikes = []
    for r in range(single_cell_repeats_sim_dataloader.n_repeats):
        center_spikes, _, spike_bins, snippet_transitions = single_cell_repeats_sim_dataloader.fetch_data_repeat_spikes_and_bin_times(
            args.stimulus_ix,
            r,
            center_cell_id,
            None
        )

        ground_truth_spikes.append((center_spikes, spike_bins, snippet_transitions))


    print('Saving simulated spike trains')
    with open(args.save_path, 'wb') as pfile:
        pickle.dump({
            'simulated': simulated_spikes_for_cell,
            'ground truth': ground_truth_spikes
        }, pfile)






