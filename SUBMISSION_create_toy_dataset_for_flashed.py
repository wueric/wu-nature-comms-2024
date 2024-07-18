import argparse
import pickle
from typing import Dict, Any, List

import numpy as np

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct


def make_glm_stim_time_component(config_settings: Dict[str, Any]) \
        -> np.ndarray:
    n_bins_before = config_settings[dcp.TimebinningSection.NBINS_BEFORE_TRANS]
    n_bins_after = config_settings[dcp.TimebinningSection.NBINS_AFTER_TRANS]
    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

    ################################################################
    # Make the stimulus separable time component based on the config
    stimulus_separable_time = np.zeros((n_bins_before + n_bins_after,),
                                       dtype=np.float32)

    n_bins_high = 2000 // samples_per_bin  # FIXME make this a constant
    stimulus_separable_time[n_bins_before:n_bins_before + n_bins_high] = 1.0
    return stimulus_separable_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Mass-generate reconstructions for each method")
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('output_path', type=str, help='path to toy dataset pickle')
    parser.add_argument('example_indices', type=int, nargs='+', help='example stimulus indices to fetch')
    parser.add_argument('-hh', '--heldout', action='store_true', default=False, help='use heldout partition')

    args = parser.parse_args()

    config_settings = dcp.read_config_file(args.cfg_file)
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

    # we only do reconstructions from single-bin data here
    # use a separate script to do GLM reconstructions once we get those going
    response_vector = ddu.timebin_load_single_partition_trials_cell_id_list(
        cells_ordered,
        cell_ids_as_ordered_list,
        test_heldout_nscenes_patch_list,
    )

    stacked_subset_responses = []
    stacked_subset_stimuli = []
    for example_ix in args.example_indices:

        example_responses = response_vector[example_ix, ...]
        stacked_subset_responses.append(example_responses)

        example_stimulus = test_heldout_frames[example_ix, ...]
        stacked_subset_stimuli.append(example_stimulus)

    example_data_and_responses = {
        'spikes': np.stack(stacked_subset_responses, axis=0),
        'stimulus': np.stack(stacked_subset_stimuli, axis=0),
        'time_component': glm_stim_time_component
    }

    additional_stuff_dict = {
        'valid_region': convex_hull_valid_mask,
        'crop_width_low': crop_width_low,
        'crop_width_high': crop_width_high,
        'crop_height_low': crop_height_low,
        'crop_height_high': crop_height_high,
        'downsample_factor': nscenes_downsample_factor
    }

    with open(args.output_path, 'wb') as pfile:
        pickle.dump(
            {
                'data': example_data_and_responses,
                'metadata': additional_stuff_dict
            },
            pfile
        )




