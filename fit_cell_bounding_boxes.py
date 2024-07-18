import argparse
import pickle
from typing import Dict, List, Tuple, Optional

import visionloader as vl

import numpy as np

import lib.data_utils.sta_metadata
from lib.dataset_specific_stimulus_scaling.data_specific_stimulus_scaling import dispatch_stimulus_scale, \
    get_stixel_size_wn, NSCENES_STIXEL_SIZE
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import CroppedSTABoundingBox, make_bounding_box, DownsampledCroppedFullBox, \
    make_fixed_size_bounding_box
import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.dataset_config_parser.dataset_config_parser import read_config_file
import lib.data_utils.data_util as du
from fastconv import conv2d

'''
README

Saves bounding boxes and blurred STAs
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Compute bounding boxes for each cell in cropped natural scenes coordinates')
    parser.add_argument('cfg_file', type=str, help='path to cfg file')
    parser.add_argument('-f', '--fixed_size', action='store_true', default=False,
                        help='Force the bounding boxes to always be exactly the specified size')
    args = parser.parse_args()

    config_settings = read_config_file(args.cfg_file)

    ref_lookup_key = dcp.awsify_piece_name_and_datarun_lookup_key(config_settings['ReferenceDataset'].path,
                                                                  config_settings['ReferenceDataset'].name)

    ######## From the config file, load the reference dataset ###############################
    ######## for the target cells, grab sig stixel STAs, and figure out the cropping ########
    reference_dataset_info = config_settings['ReferenceDataset']  # type: dcp.DatasetInfo
    reference_dataset = vl.load_vision_data(reference_dataset_info.path,
                                            reference_dataset_info.name,
                                            include_sta=True,
                                            include_params=True)

    crop_width_low, crop_width_high = config_settings[dcp.SettingsSection.CROP_Y_LOW], config_settings[
        dcp.SettingsSection.CROP_Y_HIGH]
    crop_height_low, crop_height_high = config_settings[dcp.SettingsSection.CROP_X_LOW], config_settings[
        dcp.SettingsSection.CROP_X_HIGH]

    upsample_fn = dispatch_stimulus_scale(ref_lookup_key)
    stixel_size = get_stixel_size_wn(ref_lookup_key)

    downsample_factor = config_settings[dcp.SettingsSection.NSCENES_DOWNSAMPLE_FACTOR]

    effective_nscenes_stix_size = NSCENES_STIXEL_SIZE * downsample_factor
    nscenes_to_wn_scale = stixel_size // effective_nscenes_stix_size

    with open(config_settings['responses_ordered'], 'rb') as ordered_cells_file:
        cells_ordered = pickle.load(ordered_cells_file)  # type: OrderedMatchedCellsStruct

    n_cells_by_type = cells_ordered.get_n_cells_by_type()
    cell_type_ordering = cells_ordered.get_cell_types()

    sig_stixels_by_type = {}  # type: Dict[str, np.ndarray]
    for cell_type in cell_type_ordering:
        cell_id_list = cells_ordered.get_reference_cell_order(cell_type)

        # this has the shape of the white noise stimulus
        # which needs to be cropped
        sig_stixels_by_type[cell_type] = lib.data_utils.sta_metadata.load_sigstixels_spatial_only_stas(
            reference_dataset,
            cell_id_list,
            sig_stixels_threshold=4.0
        )

    # apply an insignificant spatial blur so we don't have artifacts from the stixel blocks
    mini_2d_gaussian = du.matlab_style_gauss2D(shape=(11, 11), sigma=1.0).astype(np.float32)

    ##### Now crop bounding boxes for each cell, fit the binomial encoding model ###############
    ##### and then put the bounding box back into the full-res filters #########################
    bounding_box_by_cell_type = {}  # type: Dict[str, List[CroppedSTABoundingBox]]
    blurred_sig_stixels_in_box_by_type = {}  # type: Dict[str, List[np.ndarray]]
    blurred_sta_dims = None # type: Optional[Tuple[int, int]]
    for cell_type, batched_filter_matrix in sig_stixels_by_type.items():
        bounding_box_by_cell_type[cell_type] = []
        blurred_sig_stixels_in_box_by_type[cell_type] = []

        cell_type_box_size = config_settings['STACropping'][cell_type]
        n_cells_batch = batched_filter_matrix.shape[0]

        batch_blurred_sig_stixels = conv2d.batch_parallel_2Dconv_same(
                batched_filter_matrix.repeat(nscenes_to_wn_scale, axis=1).repeat(nscenes_to_wn_scale, axis=2),
                mini_2d_gaussian,
                0.0
        )

        blurred_sta_dims = batched_filter_matrix.shape[1:]

        for i in range(n_cells_batch):

            if args.fixed_size:
                bounding_box = make_fixed_size_bounding_box(
                    batched_filter_matrix[i, ...],
                    cell_type_box_size,
                    stixel_size,
                    crop_width_low=crop_width_low,
                    crop_width_high=crop_width_high,
                    crop_height_low=crop_height_low,
                    crop_height_high=crop_height_high
                )
            else:
                bounding_box = make_bounding_box(batched_filter_matrix[i, ...],
                                                 cell_type_box_size,
                                                 stixel_size)

            bounding_box_by_cell_type[cell_type].append(bounding_box)
            blurred_sig_stixels_in_box_by_type[cell_type].append(batch_blurred_sig_stixels[i,...])

    #### Create the no-crop selection box ######################################################
    no_crop_sel_box = DownsampledCroppedFullBox(stixel_size, blurred_sta_dims)

    ##### Save the output file #################################################################
    with open(config_settings['bbox_path'], 'wb') as pfile:
        pickle.dump(bounding_box_by_cell_type, pfile)
        pickle.dump(blurred_sig_stixels_in_box_by_type, pfile)
        pickle.dump(no_crop_sel_box, pfile)
