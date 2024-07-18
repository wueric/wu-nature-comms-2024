import argparse
import pickle
from typing import Dict, List

import numpy as np
import torch
import tqdm

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from generate_cropped_glm_hqs_reconstructions import make_glm_stim_time_component
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from optimization_encoder.trial_glm import load_fitted_glm_families
from sim_retina.load_model_for_sim import SingleCellGLMForSim, reinflate_full_glm_model_for_sim, \
    SingleCellUncoupledGLMForSim, reinflate_uncoupled_glm_model_for_sim
from sim_retina.single_cell_sim import SimFlashedGLM, ForwardSim_FeedbackOnlyTrialGLM


def full_glm_simulate_single_cell(full_sim_glm_parameters: SingleCellGLMForSim,
                                  cropped_stimuli: np.ndarray,
                                  stimulus_time_component: np.ndarray,
                                  center_cell_spikes: np.ndarray,
                                  coupled_cell_spikes: np.ndarray,
                                  device: torch.device,
                                  n_repeats: int = 1) -> np.ndarray:
    # let the garbage collector deal with GPU resources
    # shape (batch, n_bins)
    center_cell_spikes_torch = torch.tensor(center_cell_spikes,
                                            dtype=torch.float32, device=device)

    # shape (batch, n_coupled_cells, n_bins)
    coupled_cell_spikes_torch = torch.tensor(coupled_cell_spikes,
                                             dtype=torch.float32, device=device)

    # shape (batch, n_pixels)
    cropped_stimuli_torch = torch.tensor(cropped_stimuli.reshape(cropped_stimuli.shape[0], -1),
                                         dtype=torch.float32, device=device)

    # shape (n_bins, )
    stimulus_time_component_torch = torch.tensor(stimulus_time_component,
                                                 dtype=torch.float32, device=device)

    n_bins_filter = full_sim_glm_parameters.stimulus_timecourse.shape[0]

    simulation_model = SimFlashedGLM(
        full_sim_glm_parameters.cropped_spatial_filter.reshape(-1),
        full_sim_glm_parameters.bias,
        full_sim_glm_parameters.stimulus_timecourse,
        full_sim_glm_parameters.feedback_filter,
        full_sim_glm_parameters.coupling_params[0],
        lambda x: torch.bernoulli(torch.sigmoid(x)),
        dtype=torch.float32
    ).to(device)

    simulated_spikes = simulation_model.simulate_cell(
        cropped_stimuli_torch,
        stimulus_time_component_torch,
        center_cell_spikes_torch[:, :n_bins_filter],
        coupled_cell_spikes_torch,
        n_repeats=n_repeats).detach().cpu().numpy()

    del center_cell_spikes_torch, coupled_cell_spikes_torch, cropped_stimuli_torch
    del stimulus_time_component_torch, simulation_model

    return simulated_spikes


def uncoupled_simulate_single_cell(full_sim_glm_parameters: SingleCellUncoupledGLMForSim,
                                   cropped_stimuli: np.ndarray,
                                   stimulus_time_component: np.ndarray,
                                   center_cell_spikes: np.ndarray,
                                   device: torch.device,
                                   n_repeats: int = 1) -> np.ndarray:
    # let the garbage collector deal with GPU resources
    # shape (batch, n_bins)
    center_cell_spikes_torch = torch.tensor(center_cell_spikes,
                                            dtype=torch.float32, device=device)

    # shape (batch, n_pixels)
    cropped_stimuli_torch = torch.tensor(cropped_stimuli.reshape(cropped_stimuli.shape[0], -1),
                                         dtype=torch.float32, device=device)

    # shape (n_bins, )
    stimulus_time_component_torch = torch.tensor(stimulus_time_component,
                                                 dtype=torch.float32, device=device)

    n_bins_filter = full_sim_glm_parameters.stimulus_timecourse.shape[0]

    simulation_model = ForwardSim_FeedbackOnlyTrialGLM(
        full_sim_glm_parameters.cropped_spatial_filter.reshape(-1),
        full_sim_glm_parameters.stimulus_timecourse,
        full_sim_glm_parameters.feedback_filter,
        full_sim_glm_parameters.bias,
        lambda x: torch.bernoulli(torch.sigmoid(x)),
        dtype=torch.float32
    ).to(device)

    simulated_spikes = simulation_model.simulate_cell(
        cropped_stimuli_torch,
        stimulus_time_component_torch,
        center_cell_spikes_torch[:, :n_bins_filter],
        n_repeats=n_repeats)[0].detach().cpu().numpy()

    del center_cell_spikes_torch, cropped_stimuli_torch
    del stimulus_time_component_torch, simulation_model

    return simulated_spikes


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Simulate rasters for flashed image repeat presentations")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('yaml_path', type=str, help='path to YAML specifying encoding model weights')
    parser.add_argument('cell_type', type=str, help='Cell type to simulate (break up by cell type to save memory/disk IO)')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-f', '--feedback_only', action='store_true', default=False,
                        help='GLM model specified by model_cfg_path is feedback-only')
    args = parser.parse_args()

    device = torch.device('cuda')

    ########################################################
    # Load the pretrained models from disk
    glm_parsed_paths = parse_prefit_glm_paths(args.yaml_path)
    glm_families = load_fitted_glm_families(glm_parsed_paths)

    ################################################
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

    bbox_path = config_settings['bbox_path']
    with open(bbox_path, 'rb') as pfile:
        bounding_boxes_by_type = pickle.load(pfile)
        blurred_stas_by_type = pickle.load(pfile)

    ####### Load the previously identified interactions #########################
    with open(config_settings['featurized_interactions_ordered'], 'rb') as picklefile:
        pairwise_interactions = pickle.load(picklefile)  # type: InteractionGraph

    image_rescale_low, image_rescale_high = config_settings[dcp.SettingsSection.IMAGE_RESCALE_INTERVAL]
    image_rescale_lambda = du.make_image_transform_lambda(image_rescale_low, image_rescale_high, np.float32)

    #################################################################
    # Load the raw data
    n_bins_before = config_settings[dcp.TimebinningSection.NBINS_BEFORE_TRANS]
    n_bins_after = config_settings[dcp.TimebinningSection.NBINS_AFTER_TRANS]
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    # Load the natural scenes Vision datasets
    # we only care about the repeats in this case, since that's the only
    # way we can characterize noise correlations
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

    ###############################################################
    nscenes_dset_list[0].load_repeat_frames()
    repeat_frames = nscenes_dset_list[0].repeat_frames_cached
    rescaled_repeat_frames = image_rescale_lambda(repeat_frames)
    n_repeat_frames, height, width = repeat_frames.shape

    data_repeats_response_vector = ddu.timebin_load_repeats_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        nscenes_dset_list,
    )

    glm_stim_time_component = make_glm_stim_time_component(config_settings)

    simulated_repeats_by_cell = {}  # type: Dict[int, np.ndarray]
    ground_truth_repeats_by_cell = {} # type: Dict[int, np.ndarray]

    center_cell_type = args.cell_type
    center_cell_ids = cells_ordered.get_reference_cell_order(center_cell_type)

    print(f'Simulating {center_cell_type}')
    pbar = tqdm.tqdm(total=len(center_cell_ids))
    for sel_ix, center_cell_id in enumerate(center_cell_ids):

        # get bounding box, apply to cropped stimulus
        crop_bbox = bounding_boxes_by_type[center_cell_type][sel_ix]
        crop_slice_h, crop_slice_w = crop_bbox.make_precropped_sliceobj(
            crop_hlow=crop_height_low,
            crop_hhigh=crop_height_high,
            crop_wlow=crop_width_low,
            crop_whigh=crop_width_high,
            downsample_factor=nscenes_downsample_factor
        )

        cropped_repeat_frames = rescaled_repeat_frames[:, crop_slice_h, crop_slice_w]

        center_cell_ix = cells_ordered.get_concat_idx_for_cell_id(center_cell_id)
        center_cell_spikes = data_repeats_response_vector[:, :, center_cell_ix, :]
        center_cell_spikes_unbatched = center_cell_spikes.reshape(-1, center_cell_spikes.shape[-1])

        repeated_cropped_repeat_frames = np.tile(cropped_repeat_frames[None, :, :, :],
                                                 (center_cell_spikes.shape[0], 1, 1, 1))
        repeated_cropped_repeat_frames = repeated_cropped_repeat_frames.reshape(
            -1, repeated_cropped_repeat_frames.shape[2], repeated_cropped_repeat_frames.shape[3])

        if not args.feedback_only:
            full_glm_sim_parameters = reinflate_full_glm_model_for_sim(
                glm_families,
                cells_ordered,
                (center_cell_type, center_cell_id)
            )

            coupled_cell_spikes = data_repeats_response_vector[:, :, full_glm_sim_parameters.coupling_params[1], :]
            coupled_cell_spikes_unbatched = coupled_cell_spikes.reshape(
                -1, coupled_cell_spikes.shape[-2], coupled_cell_spikes.shape[-1])

            sim_spikes = full_glm_simulate_single_cell(
                full_glm_sim_parameters,
                repeated_cropped_repeat_frames,
                glm_stim_time_component,
                center_cell_spikes_unbatched,
                coupled_cell_spikes_unbatched,
                device
            ).squeeze(1)
        else:
            uncoupled_glm_sim_parameters = reinflate_uncoupled_glm_model_for_sim(
                glm_families,
                cells_ordered,
                (center_cell_type, center_cell_id)
            )

            sim_spikes = uncoupled_simulate_single_cell(
                uncoupled_glm_sim_parameters,
                repeated_cropped_repeat_frames,
                glm_stim_time_component,
                center_cell_spikes_unbatched,
                device
            ).squeeze(1)

        sim_spikes_repeat_structure = sim_spikes.reshape(center_cell_spikes.shape[0], -1,
                                                         center_cell_spikes.shape[-1])

        simulated_repeats_by_cell[center_cell_id] = sim_spikes_repeat_structure
        ground_truth_repeats_by_cell[center_cell_id] = center_cell_spikes

        pbar.update(1)
    pbar.close()

    print('Saving simulated spike trains')
    with open(args.save_path, 'wb') as pfile:
        pickle.dump({
            'simulated': simulated_repeats_by_cell,
            'ground truth': ground_truth_repeats_by_cell
        }, pfile)
