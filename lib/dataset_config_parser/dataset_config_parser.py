import configparser

from typing import Dict, Any, Optional, List, Tuple, Union
import os

from lib.data_utils.interaction_hashable import FeatureInteraction

from dataclasses import dataclass

LookupKey = Tuple[str, str]


def awsify_piece_name_and_datarun_lookup_key(dataset_raw_path: str,
                                             dataset_name: str) -> LookupKey:
    '''
    Generates key pairs from config file contents so that we can easily access
        dataset-specific hyperparameters even when the paths to the actual datasets
        may be all sorts of crazy, i.e. on AWS

    Assumes that each specific datarun is spike-sorted only once, i.e. there is exactly
        one version, and so there is a single value corresponding to key (piece, datarun)

    We also assume that dataset_raw_path follows the naming convention exactly
        /stuff/to/path/2018-08-07-5/data000
    with no end slash.

    :param dataset_raw_path:
    :param dataset_name:
    :return:
    '''
    basename_piece = os.path.basename(os.path.dirname(dataset_raw_path))
    return (basename_piece, dataset_name)


@dataclass
class DatasetInfo:
    path: str
    name: str
    wn_xml: str
    hires_sta: str
    classification_file: Union[str, None]


@dataclass
class NScenesDatasetInfo:
    path: str
    name: str
    movie_path: str
    repeat_path: str


def generate_lookup_key_from_dataset_info(dataset_info: Union[DatasetInfo, NScenesDatasetInfo]) \
        -> LookupKey:

    return awsify_piece_name_and_datarun_lookup_key(dataset_info.path,
                                                    dataset_info.name)


@dataclass
class MovieBlockSectionDescriptor:
    path: str
    block_num: int
    block_low: int
    block_high: int


class DatasetSection:
    CFG_SPIKES_DS_PATH_KEY = 'spikes_ds_path'
    CFG_BINNED_DS_PATH_KEY = 'timebin_ds_path'

    CFG_GROUP = None  # type: Optional[str]
    SPIKES_DSET_KEY = None  # type: Optional[str]
    BINS_DSET_KEY = None  # type: Optional[str]


class MovieSection:
    CFG_GROUP = None
    MOVIE_BLOCK_DESCRIPTOR = None


class TestMovieSection(MovieSection):
    CFG_GROUP = 'TestMovieBlocks'
    MOVIE_BLOCK_DESCRIPTOR = 'test_movie_sections'


class HeldoutMovieSection(MovieSection):
    CFG_GROUP = 'HeldoutMovieBlocks'
    MOVIE_BLOCK_DESCRIPTOR = 'heldout_movie_sections'


class TestFlashedSection(MovieSection):
    CFG_GROUP = 'TestFlashedBlocks'
    MOVIE_BLOCK_DESCRIPTOR = 'test_flash_sections'


class HeldoutFlashedSection(MovieSection):
    CFG_GROUP = 'HeldoutFlashedBlocks'
    MOVIE_BLOCK_DESCRIPTOR = 'heldout_flash_sections'


class TimebinningSection:
    CFG_GROUP = 'TimeBinningParameters'

    NBINS_BEFORE_TRANS = 'n_bins_before_transition'
    NBINS_AFTER_TRANS = 'n_bins_after_transition'
    SAMPLES_PER_BIN = 'samples_per_bin'


def parse_timebinning_section(section_parser_obj,
                              output_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in section_parser_obj:
        if key == TimebinningSection.NBINS_BEFORE_TRANS:
            output_dict[TimebinningSection.NBINS_BEFORE_TRANS] = int(val)
        elif key == TimebinningSection.NBINS_AFTER_TRANS:
            output_dict[TimebinningSection.NBINS_AFTER_TRANS] = int(val)
        elif key == TimebinningSection.SAMPLES_PER_BIN:
            output_dict[TimebinningSection.SAMPLES_PER_BIN] = int(val)
    return output_dict


class OutputSection:
    CFG_GROUP = 'OutputFiles'

    RESPONSES_ORDERED = 'responses_ordered'
    FEATURIZED_INTERACTIONS = 'featurized_interactions_ordered'
    BBOX_PATH = 'bbox_path'
    INITIAL_GUESS_TIMECOURSE = 'timecourse_init_guess'


def parse_output_file_section(output_section_parser_obj,
                              output_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in output_section_parser_obj:
        if key == OutputSection.RESPONSES_ORDERED:
            output_dict[OutputSection.RESPONSES_ORDERED] = value
        elif key == OutputSection.FEATURIZED_INTERACTIONS:
            output_dict[OutputSection.FEATURIZED_INTERACTIONS] = value
        elif key == OutputSection.BBOX_PATH:
            output_dict[OutputSection.BBOX_PATH] = value
        elif key == OutputSection.INITIAL_GUESS_TIMECOURSE:
            output_dict[OutputSection.INITIAL_GUESS_TIMECOURSE] = value

    return output_dict


class SettingsSection:
    CFG_GROUP = 'Settings'

    CELL_MATCH_THRESH = 'cell_match_threshold'
    SIG_EL_CUTOFF = 'sig_el_cutoff'
    N_SIG_EL = 'n_sig_el_cutoff'
    VARIANCE_WINDOW_SIZE = 'variance_window_size'
    IMAGE_RESCALE_INTERVAL = 'image_rescale_interval'

    CROP_X_LOW = 'crop_x_low'
    CROP_X_HIGH = 'crop_x_high'
    CROP_Y_LOW = 'crop_y_low'
    CROP_Y_HIGH = 'crop_y_high'

    NSCENES_DOWNSAMPLE_FACTOR = 'nscenes_downsample_factor'


def parse_settings_section_add_defaults(settings_section_parser_obj,
                                        output_dict: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Modifies the output dict to add the parameters that result from parsing
        the Settings section of the config file
    :param cfg_reader_obj:
    :param output_dict:
    :return:
    '''
    for key, value in settings_section_parser_obj:
        if key == SettingsSection.CELL_MATCH_THRESH:
            output_dict[SettingsSection.CELL_MATCH_THRESH] = float(value)
        elif key == SettingsSection.SIG_EL_CUTOFF:
            output_dict[SettingsSection.SIG_EL_CUTOFF] = float(value)
        elif key == SettingsSection.N_SIG_EL:
            output_dict[SettingsSection.N_SIG_EL] = int(value)
        elif key == SettingsSection.VARIANCE_WINDOW_SIZE:
            output_dict[SettingsSection.VARIANCE_WINDOW_SIZE] = int(value)
        elif key == SettingsSection.IMAGE_RESCALE_INTERVAL:
            low_str, high_str = value.split(',')
            low, high = float(low_str), float(high_str)
            output_dict[SettingsSection.IMAGE_RESCALE_INTERVAL] = (low, high)
        elif key == SettingsSection.CROP_X_LOW:
            output_dict[SettingsSection.CROP_X_LOW] = int(value)
        elif key == SettingsSection.CROP_X_HIGH:
            output_dict[SettingsSection.CROP_X_HIGH] = int(value)
        elif key == SettingsSection.CROP_Y_LOW:
            output_dict[SettingsSection.CROP_Y_LOW] = int(value)
        elif key == SettingsSection.CROP_Y_HIGH:
            output_dict[SettingsSection.CROP_Y_HIGH] = int(value)
        elif key == SettingsSection.NSCENES_DOWNSAMPLE_FACTOR:
            output_dict[SettingsSection.NSCENES_DOWNSAMPLE_FACTOR] = int(value)

    # put in default settings if we haven't provided them
    if SettingsSection.CELL_MATCH_THRESH not in output_dict:
        output_dict[SettingsSection.CELL_MATCH_THRESH] = 0.95
    if SettingsSection.IMAGE_RESCALE_INTERVAL not in output_dict:
        output_dict[SettingsSection.IMAGE_RESCALE_INTERVAL] = (-0.5, 0.5)
    if SettingsSection.CROP_X_LOW not in output_dict:
        output_dict[SettingsSection.CROP_X_LOW] = 0
    if SettingsSection.CROP_X_HIGH not in output_dict:
        output_dict[SettingsSection.CROP_X_HIGH] = 0
    if SettingsSection.CROP_Y_LOW not in output_dict:
        output_dict[SettingsSection.CROP_Y_LOW] = 32
    if SettingsSection.CROP_Y_HIGH not in output_dict:
        output_dict[SettingsSection.CROP_Y_HIGH] = 32
    if SettingsSection.NSCENES_DOWNSAMPLE_FACTOR not in output_dict:
        output_dict[SettingsSection.NSCENES_DOWNSAMPLE_FACTOR] = 1

    return output_dict


class FeaturizedInteractionSection:
    CFG_GROUP = 'FeaturizedInteractions'
    FEATURIZED_INTERACTION_LIST = 'FeaturizedInteractions'


def parse_featurized_interaction_section(interaction_parser_obj,
                                         output_dict: Dict[str, Any]) -> Dict[str, Any]:
    interactions_list = []
    for _, info in interaction_parser_obj:
        interaction_str, str_dist = info.split(',')
        interaction_dist = float(str_dist)

        cell_type_a, cell_type_b = interaction_str.split('*')
        interactions_list.append(FeatureInteraction(cell_type_a, cell_type_b, interaction_dist))

    output_dict[FeaturizedInteractionSection.FEATURIZED_INTERACTION_LIST] = interactions_list
    return output_dict


class ReferenceDatasetSection:
    CFG_GROUP = 'ReferenceDataset'
    OUTPUT_KEY = 'ReferenceDataset'


def parse_reference_dataset_section(reference_dataset_obj,
                                    output_dict: Dict[str, Any]) -> Dict[str, Any]:
    ref_dataset_parsed_list = reference_dataset_obj['path'].split(',')
    if len(ref_dataset_parsed_list) == 5:
        # we specified a classification file to use as well
        ref_dataset_path, ref_dataset_name, wn_xml_path, hires_sta_path, class_path = ref_dataset_parsed_list
        reference_dataset_info = DatasetInfo(ref_dataset_path, ref_dataset_name, wn_xml_path,
                                             hires_sta_path, class_path)
        output_dict[ReferenceDatasetSection.OUTPUT_KEY] = reference_dataset_info

    elif len(ref_dataset_parsed_list) == 4:
        ref_dataset_path, ref_dataset_name, wn_xml_path, hires_sta_path, = ref_dataset_parsed_list
        reference_dataset_info = DatasetInfo(ref_dataset_path, ref_dataset_name, wn_xml_path,
                                             hires_sta_path, None)
        output_dict[ReferenceDatasetSection.OUTPUT_KEY] = reference_dataset_info
    else:
        assert False, "{0} section malformed".format(ReferenceDatasetSection.CFG_GROUP)

    return output_dict


class STACroppingSection:
    CFG_GROUP = 'STACropping'
    OUTPUT_KEY = 'STACropping'


def parse_sta_cropping_section(sta_cropping_parser_boj,
                               output_dict: Dict[str, Any]) -> Dict[str, Any]:
    wip_dict = {}
    for _, value in sta_cropping_parser_boj:
        cell_type_name, crop_str = value.split(',')
        wip_dict[cell_type_name] = int(crop_str)

    output_dict[STACroppingSection.OUTPUT_KEY] = wip_dict
    return output_dict


class NScenesMovieDatasetSection:
    CFG_GROUP = 'NScenesMovieDataset'
    OUTPUT_KEY = 'NScenesMovieDatasets'


class NScenesFlashedDatasetSection:
    CFG_GROUP = 'NScenesFlashedDataset'
    OUTPUT_KEY = 'NScenesFlashedDatasets'


def parse_movie_nscenes_dataset_section(nscenes_section_parser_obj,
                                        output_dict: Dict[str, Any]) -> Dict[str, Any]:
    # note that we might have multiple nscenes datasets
    ns_list = []
    for _, info in nscenes_section_parser_obj:
        ns_path, ns_name, ns_movie_path, ns_repeat_movie_path = info.split(',')
        ns_list.append(NScenesDatasetInfo(ns_path, ns_name, ns_movie_path, ns_repeat_movie_path))

    output_dict[NScenesMovieDatasetSection.OUTPUT_KEY] = ns_list
    return output_dict


def parse_flashed_nscenes_dataset_section(nscenes_section_parser_obj,
                                        output_dict: Dict[str, Any]) -> Dict[str, Any]:
    # note that we might have multiple nscenes datasets
    ns_list = []
    for _, info in nscenes_section_parser_obj:
        ns_path, ns_name, ns_movie_path, ns_repeat_movie_path = info.split(',')
        ns_list.append(NScenesDatasetInfo(ns_path, ns_name, ns_movie_path, ns_repeat_movie_path))

    output_dict[NScenesFlashedDatasetSection.OUTPUT_KEY] = ns_list
    return output_dict


def read_config_file(config_file_path: str) -> Dict[str, Any]:
    parser = configparser.ConfigParser()
    parser.read(config_file_path)

    output_dict = {}
    if ReferenceDatasetSection.CFG_GROUP in parser.sections():
        ref_dataset_info_section = parser[ReferenceDatasetSection.CFG_GROUP]
        output_dict = parse_reference_dataset_section(ref_dataset_info_section, output_dict)
    else:
        assert False, "Missing {0} section".format(ReferenceDatasetSection.CFG_GROUP)

    if NScenesMovieDatasetSection.CFG_GROUP in parser.sections():
        nscenes_items = parser.items(NScenesMovieDatasetSection.CFG_GROUP)
        output_dict = parse_movie_nscenes_dataset_section(nscenes_items, output_dict)

    if NScenesFlashedDatasetSection.CFG_GROUP in parser.sections():
        nscenes_items = parser.items(NScenesFlashedDatasetSection.CFG_GROUP)
        output_dict = parse_flashed_nscenes_dataset_section(nscenes_items, output_dict)

    if NScenesMovieDatasetSection.CFG_GROUP not in parser.sections() and \
            NScenesFlashedDatasetSection.CFG_GROUP not in parser.sections():
        assert False, "Natural scenes datasets not specified"

    if 'CellTypes' in parser.sections():
        cell_types_items = parser.items('CellTypes')
        output_dict['CellTypes'] = [cell_type_name for _, cell_type_name in cell_types_items]
    else:
        assert False, "Missing 'CellTypes' section"

    if STACroppingSection.CFG_GROUP in parser.sections():
        cell_types_items = parser.items(STACroppingSection.CFG_GROUP)
        output_dict = parse_sta_cropping_section(cell_types_items, output_dict)

    if FeaturizedInteractionSection.CFG_GROUP in parser.sections():
        interactions_items = parser.items(FeaturizedInteractionSection.CFG_GROUP)
        output_dict = parse_featurized_interaction_section(interactions_items, output_dict)

    if OutputSection.CFG_GROUP in parser.sections():
        output_items = parser.items(OutputSection.CFG_GROUP)
        output_dict = parse_output_file_section(output_items, output_dict)
    else:
        assert False, "No 'OutputFiles' section"

    if TestMovieSection.CFG_GROUP in parser.sections():
        movie_entries = parser.items(TestMovieSection.CFG_GROUP)
        movie_descriptors = []  # type: List[MovieBlockSectionDescriptor]
        for _, movie_descriptor in movie_entries:
            movie_path, block_num_str, low_high_str = movie_descriptor.split(',')
            block_num = int(block_num_str)
            frame_low, frame_high = map(lambda x: int(x), low_high_str.split(':'))
            movie_descriptors.append(MovieBlockSectionDescriptor(movie_path, block_num, frame_low, frame_high))
        output_dict[TestMovieSection.MOVIE_BLOCK_DESCRIPTOR] = movie_descriptors

    if HeldoutMovieSection.CFG_GROUP in parser.sections():
        movie_entries = parser.items(HeldoutMovieSection.CFG_GROUP)
        movie_descriptors = []  # type: List[MovieBlockSectionDescriptor]
        for _, movie_descriptor in movie_entries:
            movie_path, block_num_str, low_high_str = movie_descriptor.split(',')
            block_num = int(block_num_str)
            frame_low, frame_high = map(lambda x: int(x), low_high_str.split(':'))
            movie_descriptors.append(MovieBlockSectionDescriptor(movie_path, block_num, frame_low, frame_high))
        output_dict[HeldoutMovieSection.MOVIE_BLOCK_DESCRIPTOR] = movie_descriptors

    if TestFlashedSection.CFG_GROUP in parser.sections():
        movie_entries = parser.items(TestFlashedSection.CFG_GROUP)
        movie_descriptors = []  # type: List[MovieBlockSectionDescriptor]
        for _, movie_descriptor in movie_entries:
            movie_path, block_num_str, low_high_str = movie_descriptor.split(',')
            block_num = int(block_num_str)
            frame_low, frame_high = map(lambda x: int(x), low_high_str.split(':'))
            movie_descriptors.append(MovieBlockSectionDescriptor(movie_path, block_num, frame_low, frame_high))
        output_dict[TestFlashedSection.MOVIE_BLOCK_DESCRIPTOR] = movie_descriptors

    if HeldoutFlashedSection.CFG_GROUP in parser.sections():
        movie_entries = parser.items(HeldoutFlashedSection.CFG_GROUP)
        movie_descriptors = []  # type: List[MovieBlockSectionDescriptor]
        for _, movie_descriptor in movie_entries:
            movie_path, block_num_str, low_high_str = movie_descriptor.split(',')
            block_num = int(block_num_str)
            frame_low, frame_high = map(lambda x: int(x), low_high_str.split(':'))
            movie_descriptors.append(MovieBlockSectionDescriptor(movie_path, block_num, frame_low, frame_high))
        output_dict[HeldoutFlashedSection.MOVIE_BLOCK_DESCRIPTOR] = movie_descriptors

    if SettingsSection.CFG_GROUP in parser.sections():
        settings_items = parser.items(SettingsSection.CFG_GROUP)
        output_dict = parse_settings_section_add_defaults(settings_items, output_dict)

    if TimebinningSection.CFG_GROUP in parser.sections():
        timebin_items = parser.items(TimebinningSection.CFG_GROUP)
        output_dict = parse_timebinning_section(timebin_items, output_dict)

    return output_dict
