import argparse
import pickle
from typing import List

import numpy as np
import torch
import tqdm

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.dataset_config_parser.dataset_config_parser import read_config_file
from lib.dataset_config_parser.trained_model_config_parser import parse_prefit_glm_paths
from optimization_encoder.trial_glm import load_fitted_glm_families
from sim_retina.jitter_sim_helper import SingleCellRepeatsJitteredMovieDataloader, ct_full_glm_simulate_single_cell, \
    ct_uncoupled_glm_simulate_single_cell
from sim_retina.load_model_for_sim import reinflate_full_glm_model_for_sim, reinflate_uncoupled_glm_model_for_sim


def extract_ground_truth_repeat_responses(repeats_dataloader: SingleCellRepeatsJitteredMovieDataloader,
                                          center_cell_id: int) -> List[List[np.ndarray]]:
    n_stimuli = repeats_dataloader.n_stimuli
    n_repeats = repeats_dataloader.n_repeats

    responses_outer = []
    for stim_ix in range(n_stimuli):

        rep_acc = []
        for rep_ix in range(n_repeats):
            center_spikes, _, bin_times, frame_transitions = repeats_dataloader.fetch_data_repeat_spikes_and_bin_times(
                stim_ix, rep_ix, center_cell_id, coupled_cell_ids=None)
            rep_acc.append(center_spikes)
        responses_outer.append(rep_acc)
    return responses_outer


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate simulated repeat rasters for eye movements stimulus")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('yaml_path', type=str, help='path to YAML specifying encoding model weights')
    parser.add_argument('cell_type', type=str, help='cell type')
    parser.add_argument('save_path', type=str, help='path to save pickle file')
    parser.add_argument('-f', '--feedback_only', action='store_true', default=False,
                        help='GLM model specified by model_cfg_path is feedback-only')

    args = parser.parse_args()

    device = torch.device('cuda')

    cell_type = args.cell_type

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

    # FIXME see if we can get a better way to do this
    height, width = 160, (320 - crop_width_low - crop_width_high)

    single_cell_repeats_sim_dataloader = SingleCellRepeatsJitteredMovieDataloader(
        loaded_synchronized_datasets,
        cells_ordered,
        samples_per_bin,
        image_rescale_lambda=image_rescale_lambda,
    )

    simulated_repeats_by_cell = {}
    ground_truth_repeats_by_cell = {}

    center_cell_ids = cells_ordered.get_reference_cell_order(cell_type)
    print(f'Simulating {cell_type}')
    pbar = tqdm.tqdm(total=len(center_cell_ids))
    for sel_ix, center_cell_id in enumerate(center_cell_ids):

        # get bounding box, apply to cropped stimulus
        crop_bbox = bounding_boxes_by_type[cell_type][sel_ix]
        crop_bounds_h, crop_bounds_w = crop_bbox.make_cropping_sliceobj(
            crop_hlow=crop_height_low,
            crop_hhigh=crop_height_high,
            crop_wlow=crop_width_low,
            crop_whigh=crop_width_high,
            downsample_factor=nscenes_downsample_factor,
            return_bounds=True
        )

        # grab the ground truth spike trains
        ground_truth_spikes_for_cell = extract_ground_truth_repeat_responses(
            single_cell_repeats_sim_dataloader,
            center_cell_id
        )

        if args.feedback_only:

            uncoupled_glm_sim_parameters = reinflate_uncoupled_glm_model_for_sim(
                glm_families,
                cells_ordered,
                (cell_type, center_cell_id)
            )

            simulated_spikes_for_cell = ct_uncoupled_glm_simulate_single_cell(
                uncoupled_glm_sim_parameters,
                lambda x: torch.bernoulli(torch.sigmoid(x)),
                (crop_bounds_h, crop_bounds_w),
                single_cell_repeats_sim_dataloader,
                device,
                max_batch_size=32,
            )

        else:

            full_glm_sim_parameters = reinflate_full_glm_model_for_sim(
                glm_families,
                cells_ordered,
                (cell_type, center_cell_id),
                compute_indices=False
            )

            simulated_spikes_for_cell = ct_full_glm_simulate_single_cell(
                full_glm_sim_parameters,
                lambda x: torch.bernoulli(torch.sigmoid(x)),
                (crop_bounds_h, crop_bounds_w),
                single_cell_repeats_sim_dataloader,
                device,
                max_batch_size=32
            )

        simulated_repeats_by_cell[center_cell_id] = simulated_spikes_for_cell
        ground_truth_repeats_by_cell[center_cell_id] = ground_truth_spikes_for_cell

        pbar.update(1)

    pbar.close()

    print('Saving simulated spike trains')
    with open(args.save_path, 'wb') as pfile:
        pickle.dump(
            {
                'simulated': simulated_repeats_by_cell,
                'ground truth': ground_truth_repeats_by_cell
            },
            pfile)
