import numpy as np

import argparse
import pickle
from typing import List

import lib.data_utils.data_util as du
import lib.data_utils.dynamic_data_util as ddu
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.sta_metadata import compute_convex_hull_of_mask
from lib.dataset_specific_hyperparams.mask_roi_region import make_sig_stixel_loss_mask
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Generate reconstructions for the joint estimation of eye movements and image")
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

    samples_per_bin = config_settings[dcp.TimebinningSection.SAMPLES_PER_BIN]

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

    jitter_dataloader = ddu.JitteredMovieBatchDataloader(
        loaded_synchronized_datasets,
        cells_ordered,
        ddu.PartitionType.HELDOUT_PARTITION if args.heldout \
            else ddu.PartitionType.TEST_PARTITION,
        samples_per_bin,
        image_rescale_lambda=image_rescale_lambda,
        crop_w_ix=(crop_width_low, 320 - crop_width_low), # FIXME
    )

    extracted_data_dict = {}
    for example_ix in args.example_indices:

        history_frames, target_frames, transition_times, spike_bin_times, binned_spikes = jitter_dataloader[example_ix]
        extracted_data_dict[example_ix] = (
            (history_frames, target_frames),
            transition_times,
            spike_bin_times,
            binned_spikes
        )

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
                'data': extracted_data_dict,
                'metadata': additional_stuff_dict
            },
            pfile
        )