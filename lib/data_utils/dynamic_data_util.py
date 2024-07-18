import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Union, Dict, Tuple, Callable, Optional

import numpy as np
import rawmovie.load_movie as load_movie
import spikebinning
from movie_upsampling import compute_interval_overlaps, integer_upsample_movie
import visionloader as vl
from tqdm import tqdm

import lib.dataset_config_parser.dataset_config_parser as dcp
from lib.data_utils.data_util import batch_downsample_image, time_domain_trials_bin_spikes_multiple_cells2, \
    partition_movie_block_matches_dataset, fast_get_spike_count_multiple_cells, merge_multicell_spike_vectors, \
    merge_multicell_spike_vectors2
from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.dataset_specific_ttl_corrections.block_structure_ttl_corrections import \
    WithinBlockMatchedFrameInterval, WithinBlockMatching, RepeatMatchedFrame
from lib.dataset_specific_ttl_corrections.new_flash_structure_ttl_corrections import get_experiment_block_structure, \
    FlashBinTimeStructure, parse_flash_trial_triggers_and_assign_frames
from lib.dataset_specific_ttl_corrections.nsbrownian_ttl_structure_corrections import \
    dispatch_ns_brownian_experimental_structure_and_params, parse_structured_ns_brownian_triggers_and_assign_frames, \
    SynchronizedNSBrownianSection
from lib.dataset_specific_ttl_corrections.ttl_interval_constants import NS_BROWNIAN_N_FRAMES_PER_TRIGGER, \
    NS_BROWNIAN_FRAME_RATE, SAMPLE_RATE, NS_BROWNIAN_N_FRAMES_PER_IMAGE
from lib.dataset_specific_ttl_corrections.wn_ttl_structure import WhiteNoiseSynchroSection


class PartitionType(Enum):
    TRAIN_PARTITION = 0
    TEST_PARTITION = 1
    HELDOUT_PARTITION = 2


@dataclass
class LoadedFlashPatch:
    dataset: vl.VisionCellDataTable
    name: str
    block_matchings: List[WithinBlockMatching]
    frames_cached: np.ndarray
    stimulus_time_component: np.ndarray


class LoadedFlashedNaturalScenes:

    def __init__(self,
                 path: str,
                 name: str,
                 dataset: vl.VisionCellDataTable,
                 movie: str,
                 stimulus_time_component: np.ndarray,
                 train_block_frame: List[WithinBlockMatching],
                 test_block_frame: List[WithinBlockMatching],
                 heldout_block_frame: List[WithinBlockMatching],
                 repeat_movie: str,
                 repeat_block_frames: List[RepeatMatchedFrame],
                 downsample_factor: int = 1,
                 crop_h_low: int = 0,
                 crop_h_high: int = 0,
                 crop_w_low: int = 0,
                 crop_w_high: int = 0,
                 frames_cached: bool = False,
                 train_frames_cached: Optional[np.ndarray] = None,
                 test_frames_cached: Optional[np.ndarray] = None,
                 heldout_frames_cached: Optional[np.ndarray] = None,
                 repeats_cached: bool = False,
                 repeat_frames_cached: Optional[np.ndarray] = None):

        self.path = path
        self.name = name
        self.dataset = dataset
        self.movie = movie

        self.stimulus_time_component = stimulus_time_component

        self.train_block_frame = train_block_frame
        self.test_block_frame = test_block_frame
        self.heldout_block_frame = heldout_block_frame

        self.repeat_movie = repeat_movie
        self.repeat_block_frames = repeat_block_frames

        self.downsample_factor = downsample_factor
        self.crop_h_low = crop_h_low
        self.crop_h_high = crop_h_high
        self.crop_w_low = crop_w_low
        self.crop_w_high = crop_w_high

        self.frames_cached = frames_cached
        self.train_frames_cached = train_frames_cached
        self.test_frames_cached = test_frames_cached
        self.heldout_frames_cached = heldout_frames_cached

        self.repeats_cached = repeats_cached
        self.repeat_frames_cached = repeat_frames_cached

    def load_frames_from_disk(self):
        if not self.frames_cached:

            # load all of the frames
            print("Loading flashed images from disk", file=sys.stderr)
            with load_movie.RawMovieReader(self.movie,
                                           crop_h_low=self.crop_h_low,
                                           crop_h_high=self.crop_h_high,
                                           crop_w_low=self.crop_w_low,
                                           crop_w_high=self.crop_w_high) as raw_movie_reader:
                all_frames, n_frames = raw_movie_reader.get_all_frames_bw()

            # downsample the frames if necessary
            if self.downsample_factor != 1:
                print("Downsampling flashed images", file=sys.stderr)
                all_frames = batch_downsample_image(all_frames, self.downsample_factor)

            self.train_frames_cached = _slice_block_frames(all_frames, self.train_block_frame)
            self.test_frames_cached = _slice_block_frames(all_frames, self.test_block_frame)
            self.heldout_frames_cached = _slice_block_frames(all_frames, self.heldout_block_frame)

            self.frames_cached = True

    def load_repeat_frames(self) -> None:

        if not self.repeats_cached:
            with load_movie.RawMovieReader(self.repeat_movie,
                                           crop_h_low=self.crop_h_low,
                                           crop_h_high=self.crop_h_high,
                                           crop_w_low=self.crop_w_low,
                                           crop_w_high=self.crop_w_high) as raw_movie_reader:

                all_frames, n_frames = raw_movie_reader.get_all_frames_bw()

            if self.downsample_factor != 1:
                print("Downsampling images", file=sys.stderr)
                all_frames = batch_downsample_image(all_frames, self.downsample_factor)

            self.repeat_frames_cached = all_frames

        self.repeats_cached = True


def preload_bind_get_flashed_patches(loaded_flashed_nscenes: List[LoadedFlashedNaturalScenes],
                                     dataset_partition: PartitionType,
                                     crop_slice_h=None,
                                     crop_slice_w=None) -> List[LoadedFlashPatch]:
    ret_list = []
    for loaded_flashed_nscene in loaded_flashed_nscenes:
        loaded_flashed_nscene.load_frames_from_disk()

        if dataset_partition == PartitionType.TRAIN_PARTITION:
            relevant_frames = loaded_flashed_nscene.train_frames_cached
            relevant_blocks = loaded_flashed_nscene.train_block_frame
        elif dataset_partition == PartitionType.TEST_PARTITION:
            relevant_frames = loaded_flashed_nscene.test_frames_cached
            relevant_blocks = loaded_flashed_nscene.test_block_frame
        else:
            relevant_frames = loaded_flashed_nscene.heldout_frames_cached
            relevant_blocks = loaded_flashed_nscene.heldout_block_frame

        if crop_slice_h is None and crop_slice_w is None:
            cropped_frames = relevant_frames
        elif crop_slice_h is None:
            cropped_frames = relevant_frames[:, :, crop_slice_w]
        elif crop_slice_w is None:
            cropped_frames = relevant_frames[:, crop_slice_h, :]
        else:
            cropped_frames = relevant_frames[:, crop_slice_h, crop_slice_w]

        ret_list.append(LoadedFlashPatch(loaded_flashed_nscene.dataset, loaded_flashed_nscene.name,
                                         relevant_blocks, cropped_frames,
                                         loaded_flashed_nscene.stimulus_time_component))

    return ret_list


def concatenate_frames_from_flashed_patches(loaded_flashed_patches: List[LoadedFlashPatch]) \
        -> np.ndarray:
    ret_list = []
    for flashed_patch in loaded_flashed_patches:
        if flashed_patch.frames_cached is not None and flashed_patch.frames_cached.size > 0:
            ret_list.append(flashed_patch.frames_cached)

    return np.concatenate(ret_list, axis=0)


def make_flashed_time_component(n_bins_before: int,
                                n_bins_after: int,
                                stimulus_onset_length_bins: int) -> np.ndarray:
    stimulus_time_component = np.zeros((n_bins_before + n_bins_after,), dtype=np.float32)
    stimulus_time_component[n_bins_before:n_bins_before + stimulus_onset_length_bins] = 1.0
    return stimulus_time_component


def load_nscenes_dataset_and_timebin_blocks3(
        nscenes_info_list: List[dcp.NScenesDatasetInfo],
        samples_per_bin: Union[int, float],
        n_bins_before: int,
        n_bins_after: int,
        stimulus_onset_length_bins: int,
        test_movie_blocks: List[dcp.MovieBlockSectionDescriptor],
        heldout_movie_blocks: List[dcp.MovieBlockSectionDescriptor],
        downsample_factor: int = 1,
        crop_h_low: int = 0,
        crop_h_high: int = 0,
        crop_w_low: int = 0,
        crop_w_high: int = 0) \
        -> List[LoadedFlashedNaturalScenes]:
    '''
    Newer version of load_nscenes_dataset_and_timebin_blocks, which should help us recover more experimental
        data blocks where the triggers were screwed up
    :param nscenes_info_list:
    :param samples_per_bin: number of electrical samples per time bin, could be non-integer
    :param n_bins_before:
    :param n_bins_after:
    :param test_movie_blocks:
    :param heldout_movie_blocks:
    :return:
    '''

    loaded_dataset_list = []
    for i, nscenes_info in enumerate(nscenes_info_list):
        print("Loading natural scenes dataset {0}/{1}".format(i + 1, len(nscenes_info_list)),
              file=sys.stderr)

        nscenes_dataset = vl.load_vision_data(nscenes_info.path,
                                              nscenes_info.name,
                                              include_params=True,
                                              include_neurons=True)

        nscenes_lookup_key = dcp.generate_lookup_key_from_dataset_info(nscenes_info)
        experiment_block_structure = get_experiment_block_structure(nscenes_lookup_key)

        timebin_structure = FlashBinTimeStructure(samples_per_bin, n_bins_before, n_bins_after)

        data_blocks, repeat_blocks = parse_flash_trial_triggers_and_assign_frames(
            nscenes_dataset.get_ttl_times(),
            experiment_block_structure,
            timebin_structure
        )

        # now determine whether there are any blocks that belong to either the test
        # or heldout datasets
        train_block_frames, test_block_frames, heldout_block_frames = partition_movie_block_matches_dataset(
            nscenes_info.movie_path,
            data_blocks,
            test_movie_blocks,
            heldout_movie_blocks
        )

        loaded_dataset_list.append(LoadedFlashedNaturalScenes(
            nscenes_info.path,
            nscenes_info.name,
            nscenes_dataset,
            nscenes_info.movie_path,
            make_flashed_time_component(n_bins_before, n_bins_after, stimulus_onset_length_bins),
            train_block_frames,
            test_block_frames,
            heldout_block_frames,
            repeat_movie=nscenes_info.repeat_path,
            repeat_block_frames=repeat_blocks,
            downsample_factor=downsample_factor,
            crop_h_low=crop_h_low,
            crop_h_high=crop_h_high,
            crop_w_low=crop_w_low,
            crop_w_high=crop_w_high,
        ))

    return loaded_dataset_list


def _generate_stacked_timebinned_data(nscenes_dataset: vl.VisionCellDataTable,
                                      timebinned_times: np.ndarray,
                                      curated_typed_cells_with_dup: Dict[str, List[List[int]]]) \
        -> np.ndarray:
    acc_list = []  # type: List[List[int]]
    for key, val in curated_typed_cells_with_dup.items():
        acc_list.extend(val)

    return time_domain_trials_bin_spikes_multiple_cells2(
        nscenes_dataset,
        acc_list,
        timebinned_times
    )


def _generate_timebinned_data3(nscenes_dataset: vl.VisionCellDataTable,
                               timebinned_times: np.ndarray,
                               curated_typed_cells_with_dup: Dict[str, List[List[int]]]) \
        -> Dict[str, np.ndarray]:
    '''

    :param nscenes_dataset: natural scenes Vision dataset
    :param timebinned_times: shape (n_trials, n_time_bins + 1), integer sample time bin edges
        for each trial
    :param curated_typed_cells_with_dup:
    :return:
    '''

    binned_spikes_by_type = {}  # type: Dict[str, np.ndarray]
    for key, cells_of_type in curated_typed_cells_with_dup.items():
        binned_spikes_by_type[key] = time_domain_trials_bin_spikes_multiple_cells2(
            nscenes_dataset,
            cells_of_type,
            timebinned_times)

    return binned_spikes_by_type


def _slice_block_frames(frames: np.ndarray,
                        block_list: List[WithinBlockMatching]) -> Optional[np.ndarray]:
    if len(block_list) == 0:
        return None

    selected_frames = []
    for block_frame in block_list:
        selected_frames.extend(block_frame.frame_list)

    return frames[selected_frames, ...]


def shuffle_loaded_repeat_spikes(experimental_repeats: np.ndarray,
                                 n_synthetic_repeats: int) -> np.ndarray:
    '''

    :param experimental_repeats: shape (n_repeats, n_distinct_images, n_cells, n_bins)
    :param n_synthetic_repeats: number of synthetic repeat trials to generate
        by shuffling between trials
    :return: shape (n_synthetic_repeats, n_distinct_images, n_cells, n_bins)
    '''

    n_data_repeats, n_images, n_cells, n_bins = experimental_repeats.shape
    output = np.zeros((n_synthetic_repeats, n_images, n_cells, n_bins),
                      dtype=experimental_repeats.dtype)

    sel = np.random.randint(low=0, high=n_data_repeats,
                            size=(n_synthetic_repeats, n_images, n_cells))

    for ix in range(n_synthetic_repeats):
        for jx in range(n_images):
            for kx in range(n_cells):
                output[ix, jx, kx, :] = experimental_repeats[sel[ix, jx, kx], jx, kx, :]

    return output


def timebin_load_repeats_cell_id_list(cells_ordered: OrderedMatchedCellsStruct,
                                      target_cell_ids: List[int],
                                      loaded_nscenes_list: List[LoadedFlashedNaturalScenes]) \
        -> np.ndarray:
    repeats_multidataset_list = []  # type: List[Tuple[Dict[int, np.ndarray], List[int], np.ndarray]]

    tot_repeats = 0

    for ds_num, nscenes_dataset_wrapper in enumerate(loaded_nscenes_list):
        nscenes_dataset = nscenes_dataset_wrapper.dataset
        nscenes_name = nscenes_dataset_wrapper.name

        reduced_cells_with_ordering = [cells_ordered.get_match_ids_for_ds(target_cell, nscenes_name)
                                       for target_cell in target_cell_ids]  # type: List[List[int]]

        multicell_spikes, cell_ordering = merge_multicell_spike_vectors(nscenes_dataset,
                                                                        reduced_cells_with_ordering)

        repeat_block_frames = nscenes_dataset_wrapper.repeat_block_frames
        tot_repeats += len(repeat_block_frames)
        stacked_repeat_blocks = np.concatenate([bf.bin_cutoff_list for bf in repeat_block_frames], axis=0)

        repeats_multidataset_list.append((multicell_spikes, cell_ordering, stacked_repeat_blocks))

    # shape (n_repeat_blocks * n_images_repeats, n_cells, n_bins)
    repeat_binned_spikes_unshape = spikebinning.multidataset_bin_spikes_trials_parallel(
        repeats_multidataset_list)

    _, n_cells, n_bins = repeat_binned_spikes_unshape.shape

    repeat_binned_spikes = repeat_binned_spikes_unshape.reshape(tot_repeats, -1, n_cells, n_bins)
    return repeat_binned_spikes


def timebin_load_repeats_subset_cells(cells_ordered: OrderedMatchedCellsStruct,
                                      typed_subset_cells_wn_id: Dict[str, List[int]],
                                      loaded_nscenes_list: List[LoadedFlashedNaturalScenes],
                                      return_stacked_array: bool = False) \
        -> Union[Dict[str, np.ndarray], np.ndarray]:
    '''

    In this case, there is no test or heldout partitions, since our primary use for the
        repeats is to characterize biological and model variability

    (bit of a dirty hack here: we assume that the test repeat movie is always the same for every
        different natural scenes dataset)


    :param cells_ordered:
    :param typed_subset_cells_wn_id:
    :param loaded_nscenes_list:
    :return: Dict, key is string cell type,
        val is np.ndarray with shape (n_repeat_blocks, n_distinct_images, n_cells, n_bins)
    '''

    if return_stacked_array:

        acc_cell_id_list = []
        for cell_type in cells_ordered.get_cell_types():
            acc_cell_id_list.extend(typed_subset_cells_wn_id[cell_type])

        return timebin_load_repeats_cell_id_list(cells_ordered, acc_cell_id_list, loaded_nscenes_list)

    else:

        repeats_accumulator = []  # type: List[Dict[str, np.ndarray]]
        for ds_num, nscenes_dataset_wrapper in enumerate(loaded_nscenes_list):
            print("Processing dataset {0}/{1}".format(ds_num + 1, len(loaded_nscenes_list)),
                  file=sys.stderr)

            nscenes_dataset = nscenes_dataset_wrapper.dataset
            nscenes_name = nscenes_dataset_wrapper.name

            reduced_cells_with_types = {
                cell_type: [cells_ordered.get_match_for_ds(cell_type, ref_id, nscenes_name) for ref_id in ref_id_list]
                for cell_type, ref_id_list in typed_subset_cells_wn_id.items()
            }

            repeat_block_frames = nscenes_dataset_wrapper.repeat_block_frames
            stacked_repeat_blocks = np.concatenate([bf.bin_cutoff_list for bf in repeat_block_frames], axis=0)

            temp_binned_dict = _generate_timebinned_data3(nscenes_dataset,
                                                          stacked_repeat_blocks,
                                                          reduced_cells_with_types)

            # we need to reshape the values in the dict to accomodate the repeat structure
            repeat_binned_dict = {key: val.reshape(len(repeat_block_frames), -1, val.shape[1], val.shape[2])
                                  for key, val in temp_binned_dict.items()}

            repeats_accumulator.append(repeat_binned_dict)

        return {cell_type: np.concatenate([chunk_dict[cell_type] for chunk_dict in repeats_accumulator], axis=0)
                for cell_type in typed_subset_cells_wn_id.keys()}


def timebin_load_single_partition_trials_cell_id_list(cells_ordered: OrderedMatchedCellsStruct,
                                                      target_cell_ids: List[int],
                                                      loaded_nscenes_list: List[LoadedFlashPatch],
                                                      jitter_time_amount: Union[float, Dict[int, float]] = 0.0) \
        -> Optional[np.ndarray]:
    '''

    :param cells_ordered:
    :param target_cell_ids:
    :param loaded_nscenes_list:
    :param which_partition:
    :param return_stacked_array:
    :return:
    '''
    data_sequence = []  # type: List[Tuple[Dict[int, np.ndarray], List[int], np.ndarray]]
    for ds_num, nscenes_dataset_wrapper in enumerate(loaded_nscenes_list):
        nscenes_dataset = nscenes_dataset_wrapper.dataset
        nscenes_name = nscenes_dataset_wrapper.name

        multicell_spikes, cell_ordering = merge_multicell_spike_vectors2(
            nscenes_dataset, nscenes_name,
            cells_ordered, target_cell_ids
        )

        if jitter_time_amount != 0.0:
            multicell_spikes = jitter_spikes_dict(multicell_spikes, jitter_time_amount)

        if len(nscenes_dataset_wrapper.block_matchings) > 0:
            stacked_block_frames = np.concatenate([bf.bin_list for bf in nscenes_dataset_wrapper.block_matchings],
                                                  axis=0)
            data_sequence.append((multicell_spikes, cell_ordering, stacked_block_frames))

    if len(data_sequence) != 0:
        binned_spikes = spikebinning.multidataset_bin_spikes_trials_parallel(data_sequence)
        return binned_spikes

    return None


def timebin_load_single_partition_trials_subset_cells(cells_ordered: OrderedMatchedCellsStruct,
                                                      typed_subset_cells_wn_id: Dict[str, List[int]],
                                                      loaded_nscenes_patch_list: List[LoadedFlashPatch],
                                                      return_stacked_array: bool = True,
                                                      jitter_time_amount: float = 0.0) \
        -> Optional[Union[Dict[str, np.ndarray], np.ndarray]]:
    '''
    Implementation note: we have to be careful about the ordering of cells and cell types here
    
    :param cells_ordered: 
    :param typed_subset_cells_wn_id: 
    :param loaded_nscenes_list: 
    :param which_partition: 
    :param return_stacked_array: 
    :return: 
    '''

    if return_stacked_array:
        id_list_sequence = []
        for cell_type in cells_ordered.get_cell_types():
            if cell_type in typed_subset_cells_wn_id:
                id_list_sequence.extend(typed_subset_cells_wn_id[cell_type])

        return timebin_load_single_partition_trials_cell_id_list(cells_ordered,
                                                                 id_list_sequence,
                                                                 loaded_nscenes_patch_list,
                                                                 jitter_time_amount=jitter_time_amount)
    else:
        return {
            cell_type: timebin_load_single_partition_trials_cell_id_list(
                cells_ordered,
                cell_id_list,
                loaded_nscenes_patch_list,
                jitter_time_amount=jitter_time_amount
            ) for cell_type, cell_id_list in typed_subset_cells_wn_id.items()
        }


def construct_spike_jitter_amount_by_cell_id(jitter_time_by_type: Dict[str, float],
                                             cells_ordered: OrderedMatchedCellsStruct) -> Dict[int, float]:
    ct_order = cells_ordered.get_cell_types()
    ret_dict = {}
    for ct in ct_order:
        cell_ids = cells_ordered.get_reference_cell_order(ct)
        for cell_id in cell_ids:
            ret_dict[cell_id] = jitter_time_by_type[ct]
    return ret_dict


def jitter_spikes_dict(spikes_dict: Dict[int, np.ndarray],
                       jitter_time_amount: Union[float, Dict[int, float]]) -> Dict[int, np.ndarray]:
    jittered_spike_vector_dict = {}
    for cell_id, spike_vector in spikes_dict.items():
        scale = jitter_time_amount
        if isinstance(jitter_time_amount, dict):
            scale = jitter_time_amount[cell_id]

        jittered_amount = np.random.normal(scale=scale, size=spike_vector.shape)
        jittered_spike_vector = np.around(jittered_amount + spike_vector).astype(np.int64)
        jittered_spike_vector_dict[cell_id] = np.sort(jittered_spike_vector)

    return jittered_spike_vector_dict


def _generate_singlebinned_data(nscenes_dataset: vl.VisionCellDataTable,
                                block_frame_list: List[WithinBlockMatchedFrameInterval],
                                curated_typed_cells_with_dup: Dict[str, List[List[int]]]) \
        -> List[Dict[str, np.ndarray]]:
    spikes_dict_accumulator = []  # type: List[Dict[str, np.ndarray]]

    pbar = tqdm(total=len(block_frame_list))
    for j, block_frame in enumerate(block_frame_list):
        spikes_by_type = {
            key: fast_get_spike_count_multiple_cells(nscenes_dataset, cells_of_type,
                                                     block_frame.interval_list) for key, cells_of_type in
            curated_typed_cells_with_dup.items()
        }

        spikes_dict_accumulator.append(spikes_by_type)
        pbar.update(1)
    pbar.close()

    return spikes_dict_accumulator


def singlebin_load_all_trials2(cells_ordered: OrderedMatchedCellsStruct,
                               loaded_nscenes_list: List[LoadedFlashedNaturalScenes]) \
        -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    '''

    :param cells_ordered:
    :param loaded_nscenes_list:
    :return:
    '''
    train_spikes_accumulator = []  # type: List[Dict[str, np.ndarray]]
    test_spikes_accumulator = []  # type: List[Dict[str, np.ndarray]]
    heldout_spikes_accumulator = []  # type: List[Dict[str, np.ndarray]]

    for ds_num, nscenes_wrapper in enumerate(loaded_nscenes_list):
        print("Processing dataset {0}/{1}".format(ds_num + 1, len(loaded_nscenes_list)),
              file=sys.stderr)

        train_block_frames = nscenes_wrapper.train_block_frame
        test_block_frames = nscenes_wrapper.test_block_frame
        heldout_block_frames = nscenes_wrapper.heldout_block_frame

        nscenes_dataset = nscenes_wrapper.dataset
        nscenes_dsname = nscenes_wrapper.name

        valid_cell_types = cells_ordered.get_cell_types()
        curated_cells_with_types = {cell_type: cells_ordered.get_cell_order_for_ds_name(cell_type, nscenes_dsname) for
                                    cell_type in valid_cell_types}  # type: Dict[str, List[List[int]]]

        print("Binning training data", file=sys.stderr)
        train_frame_chunk, train_spike_chunk = _generate_singlebinned_data(
            nscenes_dataset,
            train_block_frames,
            curated_cells_with_types,
        )
        train_spikes_accumulator.extend(train_spike_chunk)

        # if we need to extract a testing dataset, do so now
        if len(test_block_frames) > 0:
            print("Binning test data", file=sys.stderr)
            test_frame_chunk, test_spike_chunk = _generate_singlebinned_data(
                nscenes_dataset,
                test_block_frames,
                curated_cells_with_types,
            )
            test_spikes_accumulator.extend(test_spike_chunk)

        # if we need to extract a heldout dataset, do so now
        if len(heldout_block_frames) > 0:
            print("Binning heldout data", file=sys.stderr)
            heldout_frame_chunk, heldout_spike_chunk = _generate_singlebinned_data(
                nscenes_dataset,
                heldout_block_frames,
                curated_cells_with_types,
            )
            heldout_spikes_accumulator.extend(heldout_spike_chunk)

    # concatenate all the stuff
    concatenated_train_spikes = {
        cell_type: np.concatenate([chunk_dict[cell_type] for chunk_dict in train_spikes_accumulator],
                                  axis=0) for cell_type in
        cells_ordered.get_cell_types()}  # type: Dict[str, np.ndarray]

    concatenated_test_spikes = {
        cell_type: np.concatenate([chunk_dict[cell_type] for chunk_dict in test_spikes_accumulator],
                                  axis=0) for cell_type in
        cells_ordered.get_cell_types()}  # type: Dict[str, np.ndarray]

    concatenated_heldout_spikes = {
        cell_type: np.concatenate([chunk_dict[cell_type] for chunk_dict in heldout_spikes_accumulator],
                                  axis=0) for cell_type in
        cells_ordered.get_cell_types()}  # type: Dict[str, np.ndarray]

    return concatenated_train_spikes, concatenated_test_spikes, concatenated_heldout_spikes


def upsample_movie(movie_frames: np.ndarray,
                   movie_bin_cutoffs: np.ndarray,
                   bin_cutoffs: np.ndarray,
                   show_pbar: bool = True) -> np.ndarray:
    '''
    Upsamples the stimulus movie up to the rate determined
        by bin_cutoffs

    IMPORTANT: assumes that the movie frame rate is lower than the
        rate determined by bin_cutoffs, such that this function
        ALWAYS performs an upsampling

    :param movie_frames: shape (frame_dim, n_frames)
    :param movie_bin_cutoffs: shape (n_frames + 1, ), start and end sample numbers for
        each movie frame
    :param bin_cutoffs: shape (n_bins + 1, ), start and end sample numbers for
        each bin
    :return:
    '''

    frame_dim, n_frames = movie_frames.shape
    n_frame_cutoffs = movie_bin_cutoffs.shape[0]
    if n_frame_cutoffs - 1 != n_frames:
        raise ValueError('n_frame_cutoffs must have 1 more entry than n_frames')

    n_bin_cutoffs = bin_cutoffs.shape[0]
    n_bins = n_bin_cutoffs - 1

    upsampled_movie = np.zeros((frame_dim, n_bins), dtype=movie_frames.dtype)

    if show_pbar:
        pbar = tqdm(total=n_bins)

    frame_idx = 0
    for us_idx in range(n_bins):

        low, high = bin_cutoffs[us_idx], bin_cutoffs[us_idx + 1]
        bin_width = high - low

        # determine which movie frames this interval overlaps with
        # note that we can do this efficiently because frame_idx...
        frame_low = frame_idx
        while frame_low < n_frames and movie_bin_cutoffs[frame_low + 1] < low:
            frame_low += 1

        frame_sel_indices = []
        frame_overlaps = []
        frame_high = frame_low - 1
        # each iteration of the loop corresponds to an overlapping frame
        while frame_high < (n_frames - 1) and movie_bin_cutoffs[frame_high + 1] < high:
            frame_high += 1

            frame_sel_indices.append(frame_high)
            curr_frame_start, curr_frame_end = movie_bin_cutoffs[frame_high], movie_bin_cutoffs[frame_high + 1]

            interval_overlap = min(curr_frame_end, high) - max(curr_frame_start, low)
            frame_overlaps.append(interval_overlap / bin_width)

        # assign the value for the interval
        # if this interval only overlaps with one frame, then just do a copy
        # otherwise we have to do matrix multiplication
        if len(frame_sel_indices) == 0:
            assert False, 'Each bin interval should overlap with at least 1 frame interval'
        if len(frame_sel_indices) == 1:
            upsampled_movie[:, us_idx] = movie_frames[:, frame_sel_indices[0]]
        else:
            # shape (frame_dim, n_overlapping_frames)
            selected_frames = movie_frames[:, frame_sel_indices]

            # shape (n_overlapping_frames, )
            frame_weight_vector = np.array(frame_overlaps)

            mul_value = (selected_frames @ frame_weight_vector[:, None]).squeeze(-1)
            upsampled_movie[:, us_idx] = mul_value

        frame_idx = frame_high

        if show_pbar:
            pbar.update(1)

    if show_pbar:
        pbar.close()

    return upsampled_movie


def movie_bin_spikes_multiple_cells2(vision_dataset: vl.VisionCellDataTable,
                                     good_cell_ordering: List[List[int]],
                                     movie_bin_edges: np.ndarray,
                                     jitter_time_amount: Union[float, Dict[int, float]] = 0.0) -> np.ndarray:
    '''

    :param vision_dataset:
    :param good_cell_ordering:
    :param movie_bin_edges:
    :return:
    '''

    merged_spike_vector_dict, cell_order = merge_multicell_spike_vectors(vision_dataset,
                                                                         good_cell_ordering)

    if jitter_time_amount != 0.0:
        merged_spike_vector_dict = jitter_spikes_dict(merged_spike_vector_dict,
                                                      jitter_time_amount)

    return spikebinning.bin_spikes_movie(merged_spike_vector_dict,
                                         cell_order,
                                         movie_bin_edges)


def _interval_contains_other(bigger_interval: Tuple[int, int],
                             smaller_interval: Tuple[int, int]) -> bool:
    return bigger_interval[1] >= smaller_interval[1] and bigger_interval[0] <= smaller_interval[0]


def _interval_same_start(int1: Tuple[int, int],
                         int2: Tuple[int, int]) -> bool:
    return int1[0] == int2[0]


def _interval_same_end(int1: Tuple[int, int],
                       int2: Tuple[int, int]) -> bool:
    return int1[1] == int2[1]


def split_synchronized_block(synchro_section: SynchronizedNSBrownianSection,
                             block_descriptor: dcp.MovieBlockSectionDescriptor) \
        -> Tuple[SynchronizedNSBrownianSection, List[SynchronizedNSBrownianSection]]:
    '''
    Splits a SynchronizedNSBrownianSection. Cuts out a piece corresponding to block_descriptor,
        and returns a list with the remaining chunks. There are at most two remaining chunks

    :param synchro_section:
    :param block_descriptor:
    :return:
    '''

    # first verify that the

    synchro_section_stimuli_interval = synchro_section.stimuli_start_stop_rel_to_exp_block_start()
    block_interval = (block_descriptor.block_low, block_descriptor.block_high)

    if _interval_contains_other(synchro_section_stimuli_interval, block_interval):

        # now carve up the section, need special logic to deal with the case that the overlapping interval
        # is at the start or at the end
        if _interval_same_start(synchro_section_stimuli_interval, block_interval) and \
                _interval_same_end(synchro_section_stimuli_interval, block_interval):

            # this case, do nothing, since the whole section belongs to the sub-interval
            return synchro_section, []

        elif _interval_same_start(synchro_section_stimuli_interval, block_interval):

            end_sample_num, prev_ttl_idx, frames_after_last_ttl = synchro_section.compute_stimulus_end(
                block_interval[1] - 1,
                since_block_start=True
            )

            stim_idx_start_beginning_referred = block_interval[0] + synchro_section.stim_idx_of_exp_block_start
            stim_idx_end_beginning_referred = block_interval[1] + synchro_section.stim_idx_of_exp_block_start

            # construct the subsection that belongs to the partition interval
            partition_section = SynchronizedNSBrownianSection(
                (stim_idx_start_beginning_referred, stim_idx_end_beginning_referred),
                synchro_section.n_frames_per_stimulus,
                synchro_section.frame_fetch_path,
                synchro_section.triggers[0:prev_ttl_idx + 1],
                synchro_section.frames_per_trigger,
                synchro_section.electrical_sample_rate,
                synchro_section.display_frame_rate,
                n_stimuli_since_trial_block_start=synchro_section.n_stimuli_since_trial_block_start,
                first_triggered_frame_offset=synchro_section.first_triggered_frame_offset,
                section_begin_sample_num=synchro_section.section_begin_sample_num,
                last_triggered_frame_remaining=frames_after_last_ttl,
                section_end_sample_num=end_sample_num
            )

            # construct the subsection that corresponds to the remainder (there is only one remainder section)
            remainder_frames_before_first_ttl = synchro_section.frames_per_trigger - frames_after_last_ttl
            if frames_after_last_ttl == 0:
                remainder_frames_before_first_ttl = 0

            remainder_section = SynchronizedNSBrownianSection(
                (stim_idx_end_beginning_referred, synchro_section.stimulus_start_stop[1]),
                synchro_section.n_frames_per_stimulus,
                synchro_section.frame_fetch_path,
                synchro_section.triggers[prev_ttl_idx:],
                synchro_section.frames_per_trigger,
                synchro_section.electrical_sample_rate,
                synchro_section.display_frame_rate,
                n_stimuli_since_trial_block_start=block_interval[1],
                first_triggered_frame_offset=remainder_frames_before_first_ttl,
                section_begin_sample_num=end_sample_num,
                last_triggered_frame_remaining=synchro_section.last_triggered_frame_remaining,
                section_end_sample_num=synchro_section.section_end_sample_num
            )

            return partition_section, [remainder_section, ]

        elif _interval_same_end(synchro_section_stimuli_interval, block_interval):
            # find out the start point of the inner interval w.r.t. the synchronized section
            start_sample_num, next_ttl_idx, frames_before_first_ttl = synchro_section.compute_stimulus_start(
                block_interval[0],
                since_block_start=True)

            stim_idx_start_beginning_referred = block_interval[0] + synchro_section.stim_idx_of_exp_block_start
            stim_idx_end_beginning_referred = block_interval[1] + synchro_section.stim_idx_of_exp_block_start

            remainder_triggers_partition_top = next_ttl_idx + 1 if frames_before_first_ttl == 0 else next_ttl_idx

            partition_section = SynchronizedNSBrownianSection(
                (stim_idx_start_beginning_referred, stim_idx_end_beginning_referred),
                synchro_section.n_frames_per_stimulus,
                synchro_section.frame_fetch_path,
                synchro_section.triggers[next_ttl_idx:],
                synchro_section.frames_per_trigger,
                synchro_section.electrical_sample_rate,
                synchro_section.display_frame_rate,
                n_stimuli_since_trial_block_start=block_interval[0],
                first_triggered_frame_offset=frames_before_first_ttl,
                section_begin_sample_num=start_sample_num,
                last_triggered_frame_remaining=synchro_section.last_triggered_frame_remaining,
                section_end_sample_num=synchro_section.section_end_sample_num
            )

            # construct the subsection that corresponds to the remainder (there is only one remainder section)
            remainder_section = SynchronizedNSBrownianSection(
                (synchro_section.stimulus_start_stop[0], stim_idx_start_beginning_referred),
                synchro_section.n_frames_per_stimulus,
                synchro_section.frame_fetch_path,
                synchro_section.triggers[:remainder_triggers_partition_top],
                synchro_section.frames_per_trigger,
                synchro_section.electrical_sample_rate,
                synchro_section.display_frame_rate,
                n_stimuli_since_trial_block_start=synchro_section.n_stimuli_since_trial_block_start,
                first_triggered_frame_offset=synchro_section.first_triggered_frame_offset,
                section_begin_sample_num=synchro_section.section_begin_sample_num,
                last_triggered_frame_remaining=synchro_section.frames_per_trigger - frames_before_first_ttl,
                section_end_sample_num=start_sample_num
            )

            print('shared end', remainder_section)

            return partition_section, [remainder_section, ]

        else:
            # here the partitioned block occurs in the middle, and so we need to split the original section
            # into three sections

            # find out the start and end points of the inner interval w.r.t. the synchronized section
            start_sample_num, next_ttl_idx, frames_before_first_ttl = synchro_section.compute_stimulus_start(
                block_interval[0],
                since_block_start=True)

            end_sample_num, prev_ttl_idx, frames_after_last_ttl = synchro_section.compute_stimulus_end(
                block_interval[1] - 1,
                since_block_start=True
            )

            stim_idx_start_beginning_referred = block_interval[0] + synchro_section.stim_idx_of_exp_block_start
            stim_idx_end_beginning_referred = block_interval[1] + synchro_section.stim_idx_of_exp_block_start

            # first construct the partition section
            partition_section = SynchronizedNSBrownianSection(
                (stim_idx_start_beginning_referred, stim_idx_end_beginning_referred),
                synchro_section.n_frames_per_stimulus,
                synchro_section.frame_fetch_path,
                synchro_section.triggers[next_ttl_idx:prev_ttl_idx + 1],
                synchro_section.frames_per_trigger,
                synchro_section.electrical_sample_rate,
                synchro_section.display_frame_rate,
                n_stimuli_since_trial_block_start=block_interval[0],
                first_triggered_frame_offset=frames_before_first_ttl,
                section_begin_sample_num=start_sample_num,
                last_triggered_frame_remaining=frames_after_last_ttl,
                section_end_sample_num=end_sample_num
            )

            beginning_remainder_triggers_partition_top = next_ttl_idx + 1 if frames_before_first_ttl == 0 else next_ttl_idx

            # then construct the before remainder
            before_remainder = SynchronizedNSBrownianSection(
                (synchro_section.stimulus_start_stop[0], stim_idx_start_beginning_referred),
                synchro_section.n_frames_per_stimulus,
                synchro_section.frame_fetch_path,
                synchro_section.triggers[:beginning_remainder_triggers_partition_top],
                synchro_section.frames_per_trigger,
                synchro_section.electrical_sample_rate,
                synchro_section.display_frame_rate,
                n_stimuli_since_trial_block_start=synchro_section.n_stimuli_since_trial_block_start,
                first_triggered_frame_offset=synchro_section.first_triggered_frame_offset,
                section_begin_sample_num=synchro_section.section_begin_sample_num,
                last_triggered_frame_remaining=synchro_section.frames_per_trigger - frames_before_first_ttl,
                section_end_sample_num=start_sample_num
            )

            # and then the after remainder
            after_remainder = SynchronizedNSBrownianSection(
                (stim_idx_end_beginning_referred, synchro_section.stimulus_start_stop[1]),
                synchro_section.n_frames_per_stimulus,
                synchro_section.frame_fetch_path,
                synchro_section.triggers[next_ttl_idx:],
                synchro_section.frames_per_trigger,
                synchro_section.electrical_sample_rate,
                synchro_section.display_frame_rate,
                n_stimuli_since_trial_block_start=block_interval[1],
                first_triggered_frame_offset=synchro_section.frames_per_trigger - frames_after_last_ttl,
                section_begin_sample_num=end_sample_num,
                last_triggered_frame_remaining=synchro_section.last_triggered_frame_remaining,
                section_end_sample_num=synchro_section.section_end_sample_num
            )

            return partition_section, [before_remainder, after_remainder]

    else:
        raise ValueError(
            f"block_descriptor {block_interval} does not overlap with synchro_section {synchro_section_stimuli_interval}")


def test_heldout_split_single_synchronized_block(synchro_section: SynchronizedNSBrownianSection,
                                                 test_block_descriptor: Optional[
                                                     dcp.MovieBlockSectionDescriptor] = None,
                                                 heldout_block_descriptor: Optional[
                                                     dcp.MovieBlockSectionDescriptor] = None) \
        -> Tuple[Union[None, SynchronizedNSBrownianSection],
                 Union[None, SynchronizedNSBrownianSection],
                 Union[None, List[SynchronizedNSBrownianSection]]]:
    '''
    Splits a SynchronizedNSBrownianSection into test, heldout, and remaining partitions.
        The first returned object corresponds to the SynchronizedNSBrownianSection that belongs to the test
        partition, the second object to the heldout partition, and the third to the remainder

    Return values are None if no such object is produced or needed.

    IMPLEMENTATION NOTE: we write this in a very crude way since we will never need
        to split an experimental block into more than just one test and one heldout chunk

    :param synchro_section:
    :param test_block_descriptor: description of the SynchronizedNSBrownianSection that belongs
        to either the test section; the chunk of recorded data that belongs to this
        will be returned as the first return value
    :param heldout_block_descriptor: description of the SynchronizedNSBrownianSection that belongs
        to either the test section; the chunk of recorded data that belongs to this
        will be returned as the first return value
    :return:
    '''

    # first split off the test block
    test_partition = None
    remainders = [synchro_section, ]
    if test_block_descriptor is not None:
        test_partition, remainders = split_synchronized_block(synchro_section, test_block_descriptor)

    if heldout_block_descriptor is None:
        return test_partition, None, remainders

    pass_through = []
    heldout_partition = None
    heldout_interval = (heldout_block_descriptor.block_low, heldout_block_descriptor.block_high)
    while len(remainders) > 0:

        rem = remainders.pop(0)

        remainder_start_stop_ref_block = rem.stimuli_start_stop_rel_to_exp_block_start()
        if _interval_contains_other(remainder_start_stop_ref_block, heldout_interval):
            heldout_partition, heldout_remainders = split_synchronized_block(rem, heldout_block_descriptor)
            pass_through.extend(heldout_remainders)
            break
        else:
            pass_through.append(rem)

    pass_through.extend(remainders)
    return test_partition, heldout_partition, pass_through


def split_train_test_heldout_synchro_blocks(data_blocks_by_block_id: Dict[int, SynchronizedNSBrownianSection],
                                            test_movie_blocks: Dict[int, dcp.MovieBlockSectionDescriptor],
                                            heldout_movie_blocks: Dict[int, dcp.MovieBlockSectionDescriptor]) \
        -> Tuple[Dict[int, List[SynchronizedNSBrownianSection]], Dict[int, SynchronizedNSBrownianSection], Dict[
            int, SynchronizedNSBrownianSection]]:
    '''
    :param data_blocks_by_block_id:
    :param test_movie_blocks:
    :param heldout_movie_blocks:
    :return:
    '''

    train_blocks, test_blocks, heldout_blocks = {}, {}, {}

    for block_id, synchro_section in data_blocks_by_block_id.items():

        test_descriptor = test_movie_blocks[block_id] if block_id in test_movie_blocks else None
        heldout_descriptor = heldout_movie_blocks[block_id] if block_id in heldout_movie_blocks else None

        test_block, heldout_block, remainders = test_heldout_split_single_synchronized_block(synchro_section,
                                                                                             test_descriptor,
                                                                                             heldout_descriptor)

        if test_block is not None:
            test_blocks[block_id] = test_block
        if heldout_block is not None:
            heldout_blocks[block_id] = heldout_block

        train_blocks[block_id] = remainders

    return train_blocks, test_blocks, heldout_blocks


def split_movie_block_descriptors_by_stimulus_and_block(block_desc_list: List[dcp.MovieBlockSectionDescriptor]) \
        -> Dict[str, Dict[int, dcp.MovieBlockSectionDescriptor]]:
    ret_dict = {}
    for block_desc in block_desc_list:
        if block_desc.path not in ret_dict:
            ret_dict[block_desc.path] = {}

        ret_dict[block_desc.path][block_desc.block_num] = block_desc

    return ret_dict


@dataclass
class LoadedBrownianMovieBlock:
    '''
    Data structure for holding together all of the information
        required to fit a GLM to a given jittered movie block:

    (1) Vision dataset, for dealing with spike trains
    (2) SynchronizedNSBrownianSection, tracking the frame transition
        times for the block
    (3) stimulus_frame_patches, np.ndarray containing the stimulus
        (already cropped, converted to BW, etc.)
    '''
    vision_name: str
    vision_dataset: vl.VisionCellDataTable
    timing_synchro: SynchronizedNSBrownianSection
    stimulus_frame_patches: np.ndarray


@dataclass
class LoadedWNMovieBlock:
    '''
    Data structure for holding all of the information required
        to fit a GLM to a given whtie noise movie block

    Note that although the frame generator is bound to a WhiteNoiseSynchroSection,
        we prefer to pre-generate the frames ahead of time since that allows us
        to save computing. Since the WhiteNoiseSynchroSection corresponds
        to the entire datarun rather than just a section, we simply keep the
        frame transition times rather than use the WhiteNoiseSynchroSection
    '''

    vision_name: str
    vision_dataset: vl.VisionCellDataTable

    timing_synchro: np.ndarray
    stimulus_frame_patches_wn_resolution: np.ndarray


@dataclass
class LoadedBrownianMovies:
    '''
    Data structure for holding onto the Vision dataset, as well as all of the
        synchronized blocks that we want to use from the Vision dataset

    Does not contain frames. We need to separately create LoadedBrownianMovieBlock
        for each block to hold the cached/cropped frames
    '''
    path: str
    name: str
    dataset: vl.VisionCellDataTable

    train_blocks: Dict[int, List[SynchronizedNSBrownianSection]]
    test_blocks: Dict[int, SynchronizedNSBrownianSection]
    heldout_blocks: Dict[int, SynchronizedNSBrownianSection]
    repeat_blocks: List[SynchronizedNSBrownianSection]


@dataclass
class LoadedWNMovies:
    path: str
    name: str
    dataset: vl.VisionCellDataTable
    wn_synchro_block: WhiteNoiseSynchroSection


@dataclass
class RepeatBrownianTrainingBlock:
    stimulus_movie: np.ndarray
    frame_transition_times: np.ndarray
    spike_times: np.ndarray
    spike_bin_edges: np.ndarray


class RepeatsJitteredMovieDataloader:
    '''
    Dataloader for getting complete repeats (i.e. fetching the data as is,
        no shuffling between repeats)

    No real tricky points here, because we're just getting the relevant data as is

    __getitem__ requires two dimensions, since we have to specify which stimulus
        as well as which repeat of that stimulus
    '''

    def __init__(self,
                 loaded_brownian_movies: List[LoadedBrownianMovies],
                 cell_matching: OrderedMatchedCellsStruct,
                 bin_width: int,
                 crop_h_ix: Optional[Tuple[int, int]] = None,
                 crop_w_ix: Optional[Tuple[int, int]] = None,
                 image_rescale_lambda: Optional[Callable[[np.ndarray], np.ndarray]] = None):

        self.cell_matching = cell_matching

        self.bin_width = bin_width
        self.crop_h_ix = crop_h_ix
        self.crop_w_ix = crop_w_ix

        self.image_rescale_lambda = image_rescale_lambda

        self.data_blocks = []  # type: List[_JitterReconstructionBrownianMovieBlock]
        self.cell_ids_to_bin = {}  # type: Dict[str, List[List[int]]]
        self.n_stimuli = 0

        for loaded_brownian_movie in loaded_brownian_movies:

            # this is guaranteed to be the repeat partition
            synchro_section_list = loaded_brownian_movie.repeat_blocks  # type: List[SynchronizedNSBrownianSection]
            for synchro_block in synchro_section_list:
                self.data_blocks.append(_JitterReconstructionBrownianMovieBlock(
                    loaded_brownian_movie.name,
                    loaded_brownian_movie.dataset,
                    synchro_block
                ))

                self.n_stimuli = synchro_block.n_stimuli - 1

            matched_cell_ids = []
            for ct in self.cell_matching.get_cell_types():
                matched_cell_ids.extend(self.cell_matching.get_cell_order_for_ds_name(
                    ct, loaded_brownian_movie.name))
            self.cell_ids_to_bin[loaded_brownian_movie.name] = matched_cell_ids

        self.n_repeats = len(self.data_blocks)

    def __len__(self):
        return self.n_stimuli

    @property
    def num_repeats(self):
        return self.n_repeats

    def __getitem__(self, to_get) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        stim_ix, repeat_ix = to_get
        assert isinstance(stim_ix, int) or isinstance(stim_ix, np.int64), 'can only get one stimulus image at a time'
        assert isinstance(repeat_ix, int) or isinstance(repeat_ix, np.int64), 'can only get one repeat image at a time'

        # this should be pretty easy, since we just have to figure out what we want to
        # get and then just get it

        relev_data_block = self.data_blocks[repeat_ix]
        synchro_block = relev_data_block.timing_synchro
        snippet_frames, snippet_transitions = synchro_block.get_snippet_frames(
            stim_ix, stim_ix + 1, crop_h=self.crop_h_ix, crop_w=self.crop_w_ix
        )

        snippet_transitions = snippet_transitions.astype(np.float32)  # weird but necessary
        # since the sample times get large enough that the step size between float32
        # might be more than 1

        history_frames = snippet_frames[:NS_BROWNIAN_N_FRAMES_PER_IMAGE, ...]
        target_frames = snippet_frames[NS_BROWNIAN_N_FRAMES_PER_IMAGE:, ...]

        start_sample, end_sample = synchro_block.get_snippet_sample_times(stim_ix, stim_ix + 1)

        bin_start = max(start_sample, int(np.ceil(snippet_transitions[0])))
        bin_end = min(end_sample, int(np.floor(snippet_transitions[-1])))

        spike_bins = np.r_[bin_start:bin_end:self.bin_width]

        # bin spikes for this
        binned_spikes = movie_bin_spikes_multiple_cells2(
            relev_data_block.vision_dataset,
            self.cell_ids_to_bin[relev_data_block.vision_name],
            spike_bins
        )

        if self.image_rescale_lambda is not None:
            history_frames = self.image_rescale_lambda(history_frames)
            target_frames = self.image_rescale_lambda(target_frames)

        return history_frames, target_frames, snippet_transitions, spike_bins, binned_spikes


def construct_repeat_training_blocks_from_data_repeats_dataloader(
        data_repeats_dataloader: RepeatsJitteredMovieDataloader,
        stim_indices_to_include: List[int]) -> List[RepeatBrownianTrainingBlock]:
    '''
    Constructs a series of repeat training blocks from real data repeats

    Note that we randomly choose which particular repeat for each stimulus we use. This is
        because it is possible that the retina drifts in time during the recording, and hence
        we don't want to bias the model by training only on a particular subset of tiem in the recording

    :param data_repeats_dataloader:
    :return:
    '''

    num_repeats = data_repeats_dataloader.num_repeats

    training_blocks = []
    for stim_ix in stim_indices_to_include:
        which_repeat = np.random.choice(np.r_[0:num_repeats])

        history_frames, target_frames, frame_transitions, spike_bin_times, binned_spikes = data_repeats_dataloader[
            stim_ix, which_repeat]
        all_frames = np.concatenate([history_frames, target_frames], axis=0)

        training_blocks.append(RepeatBrownianTrainingBlock(
            all_frames, frame_transitions, binned_spikes, spike_bin_times))

    return training_blocks


class ShuffledRepeatsJitteredMovieDataloader:
    '''
    Dataloader for doing repeat shuffling analysis for noise correlations
        (i.e. shuffling between repeats of the same stimulus)

    Reaches into the repeat dataset, optionally does the shuffling if specified
        by the user

    Subtleties here:
        (1) If the frame rate is unstable between repeats, then any result here
            is hard to interpret, since the repeats would not be "true" repeats
        (2) Because the frame transition times are different between each repeat,
            we have to pick a "master" set of frame transition times to use when
            doing the reconstruction. The fairest thing to do here is to randomly
            take the frame transition times from one of the repeat trials. However,
            this introduces a tricky point: how do we bin spikes from trial A into
            the frame transition times from trial B?

            There are multiple ways to do this. The easiest way, which we do here,
            is to assume that the frame rate is very stable between repeats of the
            same stimulus, and align the repeats based on the time at which the first
            frame of the repeat trials occurs. We then bin spikes at the specified bin
            rate for each trial.

    '''

    def __init__(self,
                 loaded_brownian_movies: List[LoadedBrownianMovies],
                 cell_matching: OrderedMatchedCellsStruct,
                 bin_width: int,
                 crop_h_ix: Optional[Tuple[int, int]] = None,
                 crop_w_ix: Optional[Tuple[int, int]] = None,
                 image_rescale_lambda: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 n_shuffle_at_a_time: int = 1):

        self.n_shuffle_at_a_time = n_shuffle_at_a_time
        self.cell_matching = cell_matching
        self.n_cells = sum(list(self.cell_matching.get_n_cells_by_type().values()))  # type: int

        self.bin_width = bin_width
        self.crop_h_ix = crop_h_ix
        self.crop_w_ix = crop_w_ix

        self.image_rescale_lambda = image_rescale_lambda

        self.data_blocks = []  # type: List[_JitterReconstructionBrownianMovieBlock]
        self.cell_ids_to_bin = {}  # type: Dict[str, List[List[int]]]
        self.n_stimuli = 0

        for loaded_brownian_movie in loaded_brownian_movies:

            # this is guaranteed to be the repeat partition
            synchro_section_list = loaded_brownian_movie.repeat_blocks  # type: List[SynchronizedNSBrownianSection]
            for synchro_block in synchro_section_list:
                self.data_blocks.append(_JitterReconstructionBrownianMovieBlock(
                    loaded_brownian_movie.name,
                    loaded_brownian_movie.dataset,
                    synchro_block
                ))

                self.n_stimuli = synchro_block.n_stimuli - 1

            matched_cell_ids = []
            for ct in self.cell_matching.get_cell_types():
                matched_cell_ids.extend(self.cell_matching.get_cell_order_for_ds_name(
                    ct, loaded_brownian_movie.name))
            self.cell_ids_to_bin[loaded_brownian_movie.name] = matched_cell_ids

        self.n_repeats = len(self.data_blocks)

    def __len__(self):
        return self.n_stimuli

    def __getitem__(self, to_get) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        assert isinstance(to_get, int) or isinstance(to_get, np.int64), 'can only get one stimulus image at a time'
        assert 0 <= to_get < self.n_stimuli, 'index out-of-bounds'

        # we need to bin spikes for every repeat trial
        # since a priori we don't know which ones we will use
        # and it's too complicated to do it in a more efficient way
        history_frames, target_frames = None, None  # set these in the loop

        # note that the entries here will have variable number of time bins
        # we need to trim based on the minimum number of time bins
        binned_spikes_all_repeats = []  # List[np.ndarray]
        spike_bins_times_all_repeats = []  # type: List[np.ndarray]

        snippet_transitions_all_repeats = []  # type: List[np.ndarray]

        # we retime the contents of spike_bins_times_all_repeats and snippet_transitions_all_repeats
        # so that they all of the same clock...

        for repeat_ix, data_block in enumerate(self.data_blocks):

            synchro_block = data_block.timing_synchro
            snippet_frames, snippet_transitions = synchro_block.get_snippet_frames(
                to_get, to_get + 1, crop_h=self.crop_h_ix, crop_w=self.crop_w_ix
            )

            snippet_transitions = snippet_transitions.astype(np.float32)  # weird but necessary
            # since the sample times get large enough that the step size between float32
            # might be more than 1

            if repeat_ix == 0:
                history_frames = snippet_frames[:NS_BROWNIAN_N_FRAMES_PER_IMAGE, ...]
                target_frames = snippet_frames[NS_BROWNIAN_N_FRAMES_PER_IMAGE:, ...]

            start_sample, end_sample = synchro_block.get_snippet_sample_times(to_get, to_get + 1)

            bin_start = max(start_sample, int(np.ceil(snippet_transitions[0])))
            bin_end = min(end_sample, int(np.floor(snippet_transitions[-1])))

            spike_bins = np.r_[bin_start:bin_end:self.bin_width]

            # bin spikes for this
            binned_spikes = movie_bin_spikes_multiple_cells2(
                data_block.vision_dataset,
                self.cell_ids_to_bin[data_block.vision_name],
                spike_bins
            )

            binned_spikes_all_repeats.append(binned_spikes)

            snippet_transitions_all_repeats.append(snippet_transitions)
            spike_bins_times_all_repeats.append(spike_bins)

        # now trim the binned spikes based on the minimum length
        min_n_bins = np.min([x.shape[1] for x in binned_spikes_all_repeats])

        # pick random repeat trials to get each cell's spike trains from
        repeat_selector = np.random.choice(self.n_repeats, size=(self.n_shuffle_at_a_time, self.n_cells),
                                           replace=True)

        # this is a number
        frame_transition_selector = np.random.choice(self.n_repeats, (self.n_shuffle_at_a_time,))

        # assemble the synthetic trial(s)
        n_frame_transistion_times = snippet_transitions_all_repeats[0].shape[0]
        synthetic_frame_transitions = np.zeros((self.n_shuffle_at_a_time, n_frame_transistion_times),
                                               dtype=np.float32)
        synthetic_bin_times = np.zeros((self.n_shuffle_at_a_time, min_n_bins + 1),
                                       dtype=np.float32)
        synthetic_binned_spikes = np.zeros((self.n_shuffle_at_a_time, self.n_cells, min_n_bins))
        for shuffle_ix in range(self.n_shuffle_at_a_time):

            frame_transition_pick = frame_transition_selector[shuffle_ix]
            synthetic_frame_transitions[shuffle_ix, :] = snippet_transitions_all_repeats[frame_transition_pick]
            synthetic_bin_times[shuffle_ix, :] = spike_bins_times_all_repeats[frame_transition_pick][:min_n_bins + 1]
            for cell_ix in range(self.n_cells):
                cell_binned_spikes_pick = repeat_selector[shuffle_ix, cell_ix]
                synthetic_binned_spikes[shuffle_ix, cell_ix, :] = binned_spikes_all_repeats[cell_binned_spikes_pick][
                                                                  cell_ix, :min_n_bins]

        if self.image_rescale_lambda is not None:
            history_frames = self.image_rescale_lambda(history_frames)
            target_frames = self.image_rescale_lambda(target_frames)

        return history_frames, target_frames, synthetic_frame_transitions, synthetic_bin_times, synthetic_binned_spikes


def construct_repeat_training_blocks_from_shuffled_repeats_dataloader(
        shuffled_repeats_dataloader: ShuffledRepeatsJitteredMovieDataloader,
        stim_indices_to_include: List[int]) -> List[RepeatBrownianTrainingBlock]:
    '''
    Constructs a series of repeat training blocks from shuffled repeats

    :param shuffled_repeats_dataloader:
    :return:
    '''

    n_stimuli = len(shuffled_repeats_dataloader)
    training_blocks = []
    for stim_ix in stim_indices_to_include:
        history_frames, target_frames, frame_transitions, spike_bin_times, binned_spikes = shuffled_repeats_dataloader[
            stim_ix]
        all_frames = np.concatenate([history_frames, target_frames], axis=0)

        training_blocks.append(RepeatBrownianTrainingBlock(
            all_frames, frame_transitions, binned_spikes, spike_bin_times))

    return training_blocks


def extract_center_spike_train_from_repeat_training_block(
        training_block: RepeatBrownianTrainingBlock,
        center_cell_wn: Tuple[str, int],
        cells_ordered: OrderedMatchedCellsStruct) -> np.ndarray:
    '''

    :param training_block:
    :param center_cell_wn:
    :param cells_ordered:
    :return:
    '''

    center_cell_type, center_cell_id = center_cell_wn
    center_cell_ix = cells_ordered.get_concat_idx_for_cell_id(center_cell_id)
    return training_block.spike_times[center_cell_ix, :]


def extract_coupled_spike_trains_from_repeat_training_block(
        training_block: RepeatBrownianTrainingBlock,
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct) -> np.ndarray:
    '''

    :param training_block:
    :param coupled_cells:
    :param cells_ordered:
    :return:
    '''

    _sel_ix = []
    for ct in cells_ordered.get_cell_types():
        _sel_ix.extend([cells_ordered.get_concat_idx_for_cell_id(cell_id)
                        for cell_id in coupled_cells[ct]])
    return training_block.spike_times[_sel_ix, :]


def construct_brownian_movie_framerate_timebins(jittered_movie_section: SynchronizedNSBrownianSection):
    num_stimuli_in_block = jittered_movie_section.n_stimuli
    frame_transition_times = jittered_movie_section.get_snippet_transition_times(0, num_stimuli_in_block - 1)
    return frame_transition_times


def upsample_wn_movie_patches(frames: np.ndarray,
                              frame_transition_times: np.ndarray,
                              upsample_factor: int) -> Tuple[np.ndarray, np.ndarray]:
    '''

    :param frames: (n_frames, height, width)
    :param frame_transition_times: (n_frames + 1, ), start and end times for each frame transition
    :param upsample_factor: int, factor to upsample by
    :return: upsampled frames, shape (n_frames * upsample_factor, height, width),
            AND
            upsampled frame transition times, shape (n_frames * upsample_factor + 1, )
    '''

    n_frames, height, width = frames.shape
    frames_flat = frames.reshape(n_frames, -1)
    upsampled_frames_flat = integer_upsample_movie(
        frames_flat,
        upsample_factor
    )

    n_upsampled_frames = n_frames * upsample_factor
    upsampled_transition_times = np.zeros((n_upsampled_frames + 1,),
                                          dtype=frame_transition_times.dtype)
    for read_low_ix in range(0, n_frames):
        read_low = frame_transition_times[read_low_ix]
        read_high = frame_transition_times[read_low_ix + 1]

        upsampled_times = np.linspace(read_low, read_high, upsample_factor, endpoint=False)
        write_low = read_low_ix * upsample_factor
        write_high = write_low + upsample_factor

        upsampled_transition_times[write_low:write_high] = upsampled_times

    upsampled_transition_times[-1] = frame_transition_times[-1]

    return upsampled_frames_flat.reshape(-1, height, width), upsampled_transition_times


def load_jittered_nscenes_dataset_and_timebin(nscenes_info_list: List[dcp.NScenesDatasetInfo],
                                              test_movie_blocks: List[dcp.MovieBlockSectionDescriptor],
                                              heldout_movie_blocks: List[dcp.MovieBlockSectionDescriptor]) \
        -> List[LoadedBrownianMovies]:
    '''

    :param nscenes_info_list:
    :param samples_per_bin:
    :param test_movie_blocks:
    :param heldout_movie_blocks:
    :return:
    '''

    test_blocks_by_stim_and_block = split_movie_block_descriptors_by_stimulus_and_block(test_movie_blocks)
    heldout_blocks_by_stim_and_block = split_movie_block_descriptors_by_stimulus_and_block(heldout_movie_blocks)

    loaded_dataset_list = []
    for i, nscenes_info in enumerate(nscenes_info_list):
        print("Loading natural scenes dataset {0}/{1}".format(i + 1, len(nscenes_info_list)),
              file=sys.stderr)

        nscenes_dataset = vl.load_vision_data(nscenes_info.path,
                                              nscenes_info.name,
                                              include_params=True,
                                              include_neurons=True)

        experiment_structure_and_params = dispatch_ns_brownian_experimental_structure_and_params(
            dcp.generate_lookup_key_from_dataset_info(nscenes_info))

        dataset_trial_structure = experiment_structure_and_params.experiment_structure

        data_movie_path, repeat_movie_path = nscenes_info.movie_path, nscenes_info.repeat_path

        data_blocks, repeat_blocks = parse_structured_ns_brownian_triggers_and_assign_frames(
            nscenes_dataset.get_ttl_times(),
            dataset_trial_structure,
            NS_BROWNIAN_N_FRAMES_PER_TRIGGER,
            NS_BROWNIAN_FRAME_RATE,
            SAMPLE_RATE,
            data_movie_path,
            repeat_movie_path,
            raw_mode=experiment_structure_and_params.raw_mode,
            do_trigger_interpolation=experiment_structure_and_params.do_trigger_interpolation,
            tolerance_interval=experiment_structure_and_params.tolerance_interval,
            interpolation_tolerance=experiment_structure_and_params.interpolation_tolerance
        )

        # now we need a function to chop up the data blocks into smaller chunks if necessary
        # to deal with the test and heldout partitions
        train_part, test_part, heldout_part = split_train_test_heldout_synchro_blocks(
            data_blocks,
            test_blocks_by_stim_and_block[data_movie_path],
            heldout_blocks_by_stim_and_block[data_movie_path]
        )

        loaded_movie_dataset = LoadedBrownianMovies(
            nscenes_info.path,
            nscenes_info.name,
            nscenes_dataset,
            train_part,
            test_part,
            heldout_part,
            repeat_blocks
        )

        loaded_dataset_list.append(loaded_movie_dataset)

    return loaded_dataset_list


def framerate_timebin_natural_movies_subset_cells(nscenes_dataset: vl.VisionCellDataTable,
                                                  nscenes_name: str,
                                                  jittered_movie_section: SynchronizedNSBrownianSection,
                                                  cells_ordered: OrderedMatchedCellsStruct,
                                                  center_cell_wn_id: Tuple[str, int],
                                                  typed_subset_cells_wn_id: Dict[str, List[int]],
                                                  jitter_spike_times: float = 0.0) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Performs spike binning at frame rate for a single SynchronizedNSBrownianSection

    :param nscenes_dataset: Vision dataset, corresponding to the jittered natural movie run
    :param nscenes_name:
    :param jittered_movie_section: SynchronizedNSBrownianSection corresponding to the relevant
        section of the jittered natural movie
    :param cells_ordered: OrderedMatchedCellsStruct, data structure containing information
        about white noise/natural scenes cell matching/identification, as well as cell ordering
    :param center_cell_wn_id: cell type and WN id of the center cell that we should bin
    :param typed_subset_cells_wn_id: Dict[str, List[int]], WN ids of the cells whose spikes we should bin, keyed
        by string cell type
    :param jitter_spike_times: float, default=0.0, standard deviation of Gaussian (in units of electrical samples)
        to jitter the recorded spike times by
    :return:
    '''

    bin_times = construct_brownian_movie_framerate_timebins(jittered_movie_section)
    reduced_cells_with_ordering = []  # type: List[List[int]]
    for cell_type, wn_id_list in typed_subset_cells_wn_id.items():
        reduced_cells_with_ordering.extend([cells_ordered.get_match_for_ds(cell_type, ref_id, nscenes_name) \
                                            for ref_id in wn_id_list])

    multicell_spikes, cell_ordering = merge_multicell_spike_vectors(nscenes_dataset,
                                                                    reduced_cells_with_ordering)
    if jitter_spike_times != 0.0:
        multicell_spikes = jitter_spikes_dict(multicell_spikes, jitter_spike_times)

    binned_spikes = spikebinning.bin_spikes_movie(multicell_spikes,
                                                  cell_ordering,
                                                  bin_times)

    center_cell_spikes, center_cell_ordering = merge_multicell_spike_vectors(
        nscenes_dataset,
        [cells_ordered.get_match_for_ds(center_cell_wn_id[0], center_cell_wn_id[1], nscenes_name), ])
    if jitter_spike_times != 0.0:
        center_cell_spikes = jitter_spikes_dict(center_cell_spikes, jitter_spike_times)

    binned_center_cell_spikes = spikebinning.bin_spikes_movie(center_cell_spikes,
                                                              center_cell_ordering,
                                                              bin_times)

    return bin_times, binned_center_cell_spikes, binned_spikes


def construct_natural_movies_timebins(
        jittered_movie_section: SynchronizedNSBrownianSection,
        samples_per_bin: int) -> np.ndarray:
    '''
    Constructs the time bins edges for a given SynchronizedNSBrownianSection, where the number of
        samples per bin is also provided
    :param jittered_movie_section:
    :param samples_per_bin: int, number of electrical samples that should occur for each bin
    :return:
    '''

    start_sample = jittered_movie_section.section_begin_sample_num
    end_sample = jittered_movie_section.section_end_sample_num
    bin_times = np.r_[start_sample:end_sample:samples_per_bin]

    return bin_times


def multimovie_construct_natural_movies_timebins(
        jittered_movie_blocks: List[LoadedBrownianMovieBlock],
        samples_per_bin: int) -> List[np.ndarray]:
    return [construct_natural_movies_timebins(a.timing_synchro, samples_per_bin)
            for a in jittered_movie_blocks]


def timebin_natural_movies_center_cell_spikes(nscenes_dataset: vl.VisionCellDataTable,
                                              nscenes_name: str,
                                              cells_ordered: OrderedMatchedCellsStruct,
                                              center_cell_wn_id: Tuple[str, int],
                                              time_bin_edges: np.ndarray,
                                              jitter_spike_times: float = 0.0) -> np.ndarray:
    '''
    Bins spikes for the cell being fit for the natural movie GLM models

    :param nscenes_dataset: Vision dataset, corresponding to the jittered natural movie run
    :param nscenes_name:
    :param cells_ordered: OrderedMatchedCellsStruct, data structure containing information
        about white noise/natural scenes cell matching/identification, as well as cell ordering
    :param center_cell_wn_id: cell type and WN id of the center cell that we should bin
    :param time_bin_edges: np.ndarray, time bin edges, shape (n_time_bins + 1, )
    :param jitter_spike_times: float, default=0.0, standard deviation of Gaussian (in units of electrical samples)
        to jitter the recorded spike times by
    :return:
    '''

    center_cell_type, center_id = center_cell_wn_id
    center_cell_matches = cells_ordered.get_match_for_ds(center_cell_type, center_id, nscenes_name)
    return movie_bin_spikes_multiple_cells2(nscenes_dataset,
                                            [center_cell_matches, ],
                                            time_bin_edges,
                                            jitter_time_amount=jitter_spike_times)


def timebin_natural_movies_coupled_cell_spikes(nscenes_dataset: vl.VisionCellDataTable,
                                               nscenes_name: str,
                                               cells_ordered: OrderedMatchedCellsStruct,
                                               typed_subset_cells_wn_id: Dict[str, List[int]],
                                               time_bin_edges: np.ndarray,
                                               jitter_spike_times: float = 0.0) -> np.ndarray:
    '''
    Performs spike binning for the coupled cells

    :param nscenes_dataset: nscenes_dataset: Vision dataset, corresponding to the jittered natural movie run
    :param nscenes_name:
    :param cells_ordered: OrderedMatchedCellsStruct, data structure containing information
        about white noise/natural scenes cell matching/identification, as well as cell ordering
    :param typed_subset_cells_wn_id: Dict[str, List[int]], WN ids of the cells whose spikes we should bin, keyed
        by string cell type
    :param time_bin_edges: np.ndarray, time bin edges, shape (n_time_bins + 1, )
    :param jitter_spike_times: float, default=0.0, standard deviation of Gaussian (in units of electrical samples)
        to jitter the recorded spike times by
    :return:
    '''

    ct_order = cells_ordered.get_cell_types()

    reduced_cells_with_ordering = []  # type: List[List[int]]
    for cell_type in ct_order:
        if cell_type in typed_subset_cells_wn_id:
            wn_id_list = typed_subset_cells_wn_id[cell_type]
            reduced_cells_with_ordering.extend([cells_ordered.get_match_for_ds(cell_type, ref_id, nscenes_name) \
                                                for ref_id in wn_id_list])

    return movie_bin_spikes_multiple_cells2(nscenes_dataset,
                                            reduced_cells_with_ordering,
                                            time_bin_edges,
                                            jitter_time_amount=jitter_spike_times)


def construct_wn_movie_timebins(wn_movie_block: LoadedWNMovieBlock,
                                bin_width_samples: int) -> np.ndarray:
    wn_frame_transition_times = wn_movie_block.timing_synchro
    wn_bin_edges = np.r_[
                   wn_frame_transition_times[0]:wn_frame_transition_times[-1] - bin_width_samples:bin_width_samples]
    return wn_bin_edges


def multimovie_construct_wn_timebins(wn_movie_blocks: List[LoadedWNMovieBlock],
                                     bin_width_samples: int) -> List[np.ndarray]:
    return [construct_wn_movie_timebins(a, bin_width_samples)
            for a in wn_movie_blocks]


def timebin_wn_movie_coupled_cell_spikes(
        wn_movie_block: LoadedWNMovieBlock,
        spike_time_bins: np.ndarray,
        coupled_cells: Dict[str, List[int]],
        cells_ordered: OrderedMatchedCellsStruct,
        jitter_time_amount: float = 0.0) -> np.ndarray:
    '''

    :param wn_movie_block:
    :param spike_time_bins:
    :param coupled_cells:
    :param cells_ordered:
    :param jitter_time_amount:
    :return:
    '''
    ct_order = cells_ordered.get_cell_types()
    all_coupled_cell_types_ordered = []
    for coupled_cell_type in ct_order:
        all_coupled_cell_types_ordered.extend(coupled_cells[coupled_cell_type])
    coupled_cells_listlist = [[x, ] for x in all_coupled_cell_types_ordered]

    wn_coupled_spikes = movie_bin_spikes_multiple_cells2(wn_movie_block.vision_dataset,
                                                         coupled_cells_listlist,
                                                         spike_time_bins,
                                                         jitter_time_amount=jitter_time_amount)

    return wn_coupled_spikes


def timebin_natural_movies_subset_cells(nscenes_dataset: vl.VisionCellDataTable,
                                        nscenes_name: str,
                                        cells_ordered: OrderedMatchedCellsStruct,
                                        center_cell_wn_id: Tuple[str, int],
                                        typed_subset_cells_wn_id: Dict[str, List[int]],
                                        time_bin_edges: np.ndarray,
                                        jitter_spike_times: float = 0.0) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''
    Performs spike binning for a single SynchronizedNSBrownianSection

    :param nscenes_dataset: Vision dataset, corresponding to the jittered natural movie run
    :param cells_ordered: OrderedMatchedCellsStruct, data structure containing information
        about white noise/natural scenes cell matching/identification, as well as cell ordering
    :param center_cell_wn_id: cell type and WN id of the center cell that we should bin
    :param typed_subset_cells_wn_id: Dict[str, List[int]], WN ids of the cells whose spikes we should bin, keyed
        by string cell type
    :param time_bin_edges: np.ndarray, time bin edges, shape (n_time_bins + 1, )
    :param jitter_spike_times: float, default=0.0, standard deviation of Gaussian (in units of electrical samples)
        to jitter the recorded spike times by
    :return: bin time cutoffs (np.ndarray), binned center cell spikes (np.ndarray), AND binned spikes (np.ndarray)
    '''

    reduced_cells_with_ordering = []  # type: List[List[int]]
    for cell_type, wn_id_list in typed_subset_cells_wn_id.items():
        reduced_cells_with_ordering.extend([cells_ordered.get_match_for_ds(cell_type, ref_id, nscenes_name) \
                                            for ref_id in wn_id_list])

    multicell_spikes, cell_ordering = merge_multicell_spike_vectors(nscenes_dataset,
                                                                    reduced_cells_with_ordering)
    if jitter_spike_times != 0.0:
        multicell_spikes = jitter_spikes_dict(multicell_spikes, jitter_spike_times)

    binned_spikes = spikebinning.bin_spikes_movie(multicell_spikes,
                                                  cell_ordering,
                                                  time_bin_edges)

    center_cell_spikes, center_cell_ordering = merge_multicell_spike_vectors(
        nscenes_dataset,
        [cells_ordered.get_match_for_ds(center_cell_wn_id[0], center_cell_wn_id[1], nscenes_name), ])
    if jitter_spike_times != 0.0:
        center_cell_spikes = jitter_spikes_dict(center_cell_spikes, jitter_spike_times)

    binned_center_cell_spikes = spikebinning.bin_spikes_movie(center_cell_spikes,
                                                              center_cell_ordering,
                                                              time_bin_edges)

    return binned_center_cell_spikes, binned_spikes


def extract_singledata_movie_and_spikes(nscenes_dataset: vl.VisionCellDataTable,
                                        nscenes_name: str,
                                        synchro_block: SynchronizedNSBrownianSection,
                                        bin_width_samples: int,
                                        stimulus_window: Tuple[int, int],
                                        center_cell_iden: Tuple[str, int],
                                        coupled_cells_by_type: Dict[str, List[int]],
                                        cell_matching: OrderedMatchedCellsStruct) \
        -> Tuple[Tuple[int, int], Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    '''

    :param synchro_block: data structure describing the stimulus and TTL structure
        of the section of data
    :param bin_width_samples: width of a single bin, in units of recording array samples
    :param stimulus_window: start (inclusive) and end (exclusive) indices of the stimulus
        image (note, not frame) to show. Note that the indices refer to the start of the
        synchro_block, i.e. 0 is the first stimulus image presented in the block, no matter
        when in time the block occurs in the experimental recording
    :return: in order,

        indices of the first (inclusive) and last (exclusive) frames of the movie. Note
            that we don't actually fetch the movie under question here, since the caller
            will have to be responsible for doing crops, downsampling etc. and we want
            to limit the number of arguments to this function

        upsample selection indices, integer-valued shape (n_bins, 2)
            AND
        upsample frame weights, floating-point valued shape (n_bins, 2)

        center cell spikes, shape (n_bins, )

        coupled cell spikes, shape (n_coupled_cells, n_bins)
    '''
    start_stim, end_stim = stimulus_window

    frame_start_ix = synchro_block.compute_start_frame_idx_for_stim(start_stim, since_block_start=True)
    frame_end_ix = synchro_block.compute_end_frame_idx_for_stim(end_stim - 1, since_block_start=True)

    frame_transition_times = synchro_block.get_frame_transition_times(frame_start_ix, frame_end_ix)

    start_sample_num, next_ttl, frames_between_start_and_ttl = synchro_block.compute_stimulus_start(
        start_stim, since_block_start=True)

    last_sample_num, before_end_ttl, frames_between_ttl_and_end = synchro_block.compute_stimulus_end(
        end_stim - 1, since_block_start=True)

    bin_times = np.r_[start_sample_num:last_sample_num + 1:bin_width_samples]

    overlap_sel_ix, overlap_weights = compute_interval_overlaps(frame_transition_times,
                                                                bin_times)

    reduced_cells_with_ordering = []  # type: List[List[int]]
    for cell_type, wn_id_list in coupled_cells_by_type.items():
        reduced_cells_with_ordering.extend([cell_matching.get_match_for_ds(cell_type, ref_id, nscenes_name) \
                                            for ref_id in wn_id_list])

    multicell_spikes, cell_ordering = merge_multicell_spike_vectors(nscenes_dataset,
                                                                    reduced_cells_with_ordering)

    binned_spikes = spikebinning.bin_spikes_movie(multicell_spikes,
                                                  cell_ordering,
                                                  bin_times)

    center_cell_spikes, center_cell_ordering = merge_multicell_spike_vectors(
        nscenes_dataset,
        [cell_matching.get_match_for_ds(center_cell_iden[0], center_cell_iden[1], nscenes_name), ])

    binned_center_cell_spikes = spikebinning.bin_spikes_movie(center_cell_spikes,
                                                              center_cell_ordering,
                                                              bin_times)

    return (frame_start_ix, frame_end_ix), (overlap_sel_ix, overlap_weights), binned_center_cell_spikes, binned_spikes


def preload_bind_jittered_movie_patches_to_synchro(
        loaded_brownian_movies: List[LoadedBrownianMovies],
        included_blocks_by_name: Dict[str, List[int]],
        crop_low_high: Tuple[Tuple[int, int], Tuple[int, int]],
        dataset_partition: PartitionType,
        verbose: bool = True) \
        -> List[LoadedBrownianMovieBlock]:
    '''

    :param loaded_brownian_movies:
    :param included_blocks_by_name:
    :param crop_low_high:
    :param dataset_partition:
    :return:
    '''

    output_blocks = []  # type: List[LoadedBrownianMovieBlock]
    for loaded_brownian_movie in loaded_brownian_movies:
        ds_key = loaded_brownian_movie.name
        if verbose:
            print(f'Loading frames for {ds_key} {dataset_partition}', file=sys.stderr)

        rel_block_nums = included_blocks_by_name[ds_key]
        if dataset_partition == PartitionType.TRAIN_PARTITION:

            if verbose:
                pbar = tqdm(total=len(rel_block_nums))

            # we may have multiple synchronized sections corresponding to a single
            # block trial of the training stimulus, since if we crop a section of data
            # out for one of the other partitions we may split the training part into
            # multiple pieces
            for rel_block_num in rel_block_nums:
                synchro_section_list = loaded_brownian_movie.train_blocks[rel_block_num]
                for synchro_section in synchro_section_list:
                    output_blocks.append(_load_patches_and_bind_to_dataset(synchro_section,
                                                                           loaded_brownian_movie.name,
                                                                           loaded_brownian_movie.dataset,
                                                                           crop_low_high))
                if verbose:
                    pbar.update(1)

            if verbose:
                pbar.close()

        elif dataset_partition == PartitionType.TEST_PARTITION or \
                dataset_partition == PartitionType.HELDOUT_PARTITION:

            synchro_section_dict = loaded_brownian_movie.test_blocks if dataset_partition == PartitionType.TEST_PARTITION \
                else loaded_brownian_movie.heldout_blocks
            pbar = tqdm(total=len(rel_block_nums))
            for rel_block_num in rel_block_nums:
                output_blocks.append(_load_patches_and_bind_to_dataset(synchro_section_dict[rel_block_num],
                                                                       loaded_brownian_movie.name,
                                                                       loaded_brownian_movie.dataset,
                                                                       crop_low_high))

                if verbose:
                    pbar.update(1)
            if verbose:
                pbar.close()

    return output_blocks


def preload_bind_wn_movie_patches(loaded_wn_movie: LoadedWNMovies,
                                  start_end_wn_frame: List[Tuple[int, int]],
                                  crop_slices: Tuple[slice, slice]) \
        -> List[LoadedWNMovieBlock]:
    ret_list = []
    for (start_frame, end_frame) in start_end_wn_frame:
        synchro_section = loaded_wn_movie.wn_synchro_block
        bw_cropped_frames, transition_times = synchro_section.fetch_frames(start_frame,
                                                                           end_frame,
                                                                           crop_slices=crop_slices,
                                                                           is_bw=True)

        ret_list.append(LoadedWNMovieBlock(loaded_wn_movie.name, loaded_wn_movie.dataset,
                                           transition_times, bw_cropped_frames))

    return ret_list


def preload_bind_wn_movie_patches_at_framerate(loaded_wn_movie: LoadedWNMovies,
                                               start_end_wn_frame: List[Tuple[int, int]],
                                               crop_slices: Tuple[slice, slice]) \
        -> List[LoadedWNMovieBlock]:
    ret_list = []
    for (start_frame, end_frame) in start_end_wn_frame:
        synchro_section = loaded_wn_movie.wn_synchro_block
        bw_cropped_frames, transition_times = synchro_section.fetch_frames(start_frame,
                                                                           end_frame,
                                                                           crop_slices=crop_slices,
                                                                           is_bw=True)

        upsampled_frames, upsampled_transition_times = upsample_wn_movie_patches(
            bw_cropped_frames,
            transition_times,
            synchro_section.frame_interval
        )

        ret_list.append(LoadedWNMovieBlock(
            loaded_wn_movie.name,
            loaded_wn_movie.dataset,
            upsampled_transition_times,
            upsampled_frames
        ))

    return ret_list


def _load_patches_and_bind_to_dataset(synchro_block: SynchronizedNSBrownianSection,
                                      vision_name: str,
                                      vision_dataset: vl.VisionCellDataTable,
                                      crop_low_high: Tuple[Tuple[int, int], Tuple[int, int]]):
    h_low_high, w_low_high = crop_low_high
    cropped_section = synchro_block.fetch_frames_bw(h_low_high=h_low_high, w_low_high=w_low_high)
    return LoadedBrownianMovieBlock(vision_name, vision_dataset, synchro_block, cropped_section)


def interpolate_frame_transition_times2(synchro_section: SynchronizedNSBrownianSection) \
        -> np.ndarray:
    frame_start, frame_stop = synchro_section.frame_start_stop
    frames_per_ttl = synchro_section.frames_per_trigger

    n_frames_shown = frame_stop - frame_start
    frame_interpolation_time_buffer = []

    # first deal with the section of time before the arrival of the first trigger, if such
    # a section of time exists
    n_frames_early = synchro_section.first_triggered_frame_offset
    if n_frames_early != 0:
        first_sample = synchro_section.section_begin_sample_num
        last_sample = synchro_section.triggers[0]

        section_frame_transition_times = np.linspace(first_sample, last_sample, endpoint=False, num=n_frames_early,
                                                     dtype=np.float32)
        frame_interpolation_time_buffer.append(section_frame_transition_times)

    # then deal with the regular trigger section
    ttl_times = synchro_section.triggers
    n_ttl_triggers = ttl_times.shape[0]
    for i in range(0, n_ttl_triggers - 1):
        ttl_start, ttl_end = ttl_times[i], ttl_times[i + 1]
        interval_transition_times = np.linspace(ttl_start, ttl_end,
                                                num=frames_per_ttl, endpoint=False)
        frame_interpolation_time_buffer.append(interval_transition_times)

    # finally deal with the section of time after the arrival of the last trigger, if such
    # a section of time exists
    n_frames_late = synchro_section.last_triggered_frame_remaining
    if n_frames_late != 0:
        first_sample = ttl_times[-1]
        last_sample = synchro_section.section_end_sample_num

        section_frame_transition_times = np.linspace(first_sample, last_sample, endpoint=True,
                                                     num=n_frames_late + 1, dtype=np.float32)
        frame_interpolation_time_buffer.append(section_frame_transition_times)
    else:
        frame_interpolation_time_buffer.append(np.array([ttl_times[-1], ], dtype=np.float32))

    frame_transition_times_all = np.concatenate(frame_interpolation_time_buffer)

    assert frame_transition_times_all.shape[0] == n_frames_shown + 1, 'something terrible happened'

    return frame_transition_times_all


def compute_interval_overlaps_from_synchro(synchro_section: SynchronizedNSBrownianSection,
                                           spike_bin_edges: np.ndarray):
    '''

    :param synchro_section:
    :param spike_bin_edges: shape (n_bins + 1, )
    :return:
    '''

    movie_frame_transition_times = interpolate_frame_transition_times2(synchro_section)
    sel_a, weight_a = compute_interval_overlaps(movie_frame_transition_times, spike_bin_edges)
    return sel_a, weight_a


def multimovie_compute_interval_overlaps(
        synchro_sections: List[LoadedBrownianMovieBlock],
        multimovie_spike_bin_edges: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    '''

    :param synchro_sections:
    :param multimovie_spike_bin_edges:
    :return:
    '''
    return [compute_interval_overlaps_from_synchro(a.timing_synchro, b)
            for a, b, in zip(synchro_sections, multimovie_spike_bin_edges)]


def repeat_training_compute_interval_overlaps(
        repeat_blocks: List[RepeatBrownianTrainingBlock]) -> List[Tuple[np.ndarray, np.ndarray]]:
    '''

    :param repeat_blocks:
    :return:
    '''

    return [compute_interval_overlaps(x.frame_transition_times, x.spike_bin_edges)
            for x in repeat_blocks]


def compute_interval_overlaps_for_wn(wn_movie_block: LoadedWNMovieBlock,
                                     spike_bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''

    :param wn_movie_block:
    :param spike_bin_edges:
    :return:
    '''

    wn_frame_transition_times = wn_movie_block.timing_synchro
    sel_a, weight_a = compute_interval_overlaps(wn_frame_transition_times, spike_bin_edges)
    return sel_a, weight_a


def multimovie_compute_interval_overlaps_for_wn(
        wn_movie_blocks: List[LoadedWNMovieBlock],
        multimovie_spike_bin_edges: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [compute_interval_overlaps_for_wn(a, b) for a, b in zip(wn_movie_blocks, multimovie_spike_bin_edges)]


class DemoJitteredMovieDataloader:
    '''
    Dataloader for fetching data for the Jupyter Notebook
        demo version, using a subset of exported data from
        a pickle file

    Can only fetch one trial at a time
    '''

    def __init__(self,
                 exported_data_dict: Dict[int,
                                          Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray]]):

        min_spike_bins = float('inf')
        for _, trial_data in exported_data_dict.items():
            _, _, bin_times, binned_spikes = trial_data
            min_spike_bins = min(binned_spikes.shape[1], min_spike_bins)

        _history_frames_buffer = []
        _target_frames_buffer = []
        _frame_transitions_buffer = []

        self.bin_times_buffer = []
        _eq_size_bin_times_buffer = []

        self.binned_spikes_buffer = []
        _eq_size_binned_spikes_buffer = []
        for trial_ix, trial_data in exported_data_dict.items():

            (history_frames, target_frames), frame_transitions, bin_times, binned_spikes = trial_data
            _history_frames_buffer.append(history_frames)
            _target_frames_buffer.append(target_frames)
            _frame_transitions_buffer.append(frame_transitions)

            self.bin_times_buffer.append(bin_times)
            self.binned_spikes_buffer.append(binned_spikes)

            _eq_size_bin_times_buffer.append(bin_times[:min_spike_bins+1])
            _eq_size_binned_spikes_buffer.append(binned_spikes[:, :min_spike_bins])

        self.history_frames_buffer = np.stack(_history_frames_buffer, axis=0)
        self.target_frames_buffer = np.stack(_target_frames_buffer, axis=0)
        self.frame_transitions_buffer = np.stack(_frame_transitions_buffer, axis=0)

        self.eq_size_bin_times_buffer = np.stack(_eq_size_bin_times_buffer, axis=0)
        self.eq_size_binned_spikes_buffer = np.stack(_eq_size_binned_spikes_buffer, axis=0)

    def __len__(self):
        return self.history_frames_buffer.shape[0]

    def __getitem__(self, item) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if isinstance(item, int) or isinstance(item, np.int64):
            return (
                self.history_frames_buffer[item],
                self.target_frames_buffer[item],
                self.frame_transitions_buffer[item],
                self.bin_times_buffer[item],
                self.binned_spikes_buffer[item]
            )
        else:
            return (
                self.history_frames_buffer[item],
                self.target_frames_buffer[item],
                self.frame_transitions_buffer[item],
                self.eq_size_bin_times_buffer[item],
                self.eq_size_binned_spikes_buffer[item]
            )


@dataclass
class _JitterReconstructionBrownianMovieBlock:
    vision_name: str
    vision_dataset: vl.VisionCellDataTable
    timing_synchro: SynchronizedNSBrownianSection


class JitteredMovieBatchDataloader:

    def __init__(self,
                 loaded_brownian_movies: List[LoadedBrownianMovies],
                 cell_matching: OrderedMatchedCellsStruct,
                 dataset_partition: PartitionType,
                 bin_width: int,
                 crop_h_ix: Optional[Tuple[int, int]] = None,
                 crop_w_ix: Optional[Tuple[int, int]] = None,
                 image_rescale_lambda: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 time_jitter_spikes: Union[float, Dict[int, float]] = 0.0):

        self.cell_matching = cell_matching

        self.bin_width = bin_width
        self.crop_h_ix = crop_h_ix
        self.crop_w_ix = crop_w_ix

        self.image_rescale_lambda = image_rescale_lambda

        self.time_jitter_spikes = time_jitter_spikes

        self.data_blocks = []  # type: List[_JitterReconstructionBrownianMovieBlock]
        self.data_block_sizes = []  # type: List[int]
        self.data_block_cumsum = []  # type: List[int]
        self.cell_ids_to_bin = {}  # type: Dict[str, List[List[int]]]
        self.jitter_spike_times_lookup = {}  # type: Dict[str, Union[float, Dict[int, float]]]
        self.length = 0
        for loaded_brownian_movie in loaded_brownian_movies:
            if dataset_partition == PartitionType.TEST_PARTITION:
                synchro_section_dict = loaded_brownian_movie.test_blocks
            elif dataset_partition == PartitionType.HELDOUT_PARTITION:
                synchro_section_dict = loaded_brownian_movie.heldout_blocks
            else:
                synchro_section_dict = loaded_brownian_movie.train_blocks

            ds_name = loaded_brownian_movie.name

            for block_num, synchro_block in synchro_section_dict.items():
                self.data_blocks.append(_JitterReconstructionBrownianMovieBlock(
                    ds_name,
                    loaded_brownian_movie.dataset,
                    synchro_block
                ))

                n_stimuli = synchro_block.n_stimuli - 1

                self.data_block_cumsum.append(sum(self.data_block_sizes))
                self.data_block_sizes.append(n_stimuli)
                self.length += n_stimuli

            matched_cell_ids = []
            for ct in cell_matching.get_cell_types():
                matched_cell_ids.extend(cell_matching.get_cell_order_for_ds_name(
                    ct, loaded_brownian_movie.name))
            self.cell_ids_to_bin[loaded_brownian_movie.name] = matched_cell_ids

            if loaded_brownian_movie.name not in self.jitter_spike_times_lookup:

                if isinstance(time_jitter_spikes, dict):
                    translated_id_dict = {}  # type: Dict[int, float]

                    for ct in cell_matching.get_cell_types():
                        ref_cell_ids = cell_matching.get_reference_cell_order(ct)
                        for ref_cell_id in ref_cell_ids:
                            ns_cell_id_list = cell_matching.get_match_ids_for_ds(ref_cell_id,
                                                                                 loaded_brownian_movie.name)
                            for ns_cell_id in ns_cell_id_list:
                                translated_id_dict[ns_cell_id] = time_jitter_spikes[ref_cell_id]

                    self.jitter_spike_times_lookup[ds_name] = translated_id_dict
                else:
                    self.jitter_spike_times_lookup[ds_name] = time_jitter_spikes

    def _get_relev_block(self, ix: int) \
            -> Tuple[_JitterReconstructionBrownianMovieBlock, int]:

        block_ix = 0
        while (block_ix + 1) < len(self.data_block_cumsum) and \
                self.data_block_cumsum[block_ix + 1] <= ix:
            block_ix += 1

        from_start_of_block = ix - self.data_block_cumsum[block_ix]

        # +1 is because we have to skip the first stimulus in the block
        # since we need the stimulus history
        return self.data_blocks[block_ix], from_start_of_block + 1

    def __len__(self):
        return self.length

    def __getitem__(self, to_get) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(to_get, int) or isinstance(to_get, np.int64):
            # easy case, we only have to get a single example,
            # and so nothing clever needs to be done to assemble
            # a batch with the same number of spike bins
            block_data, stim_idx = self._get_relev_block(to_get)
            synchro_block = block_data.timing_synchro
            snippet_frames, snippet_transitions = synchro_block.get_snippet_frames(
                stim_idx - 1, stim_idx, crop_h=self.crop_h_ix, crop_w=self.crop_w_ix
            )

            # FIXME may need to change timekeeping to ms to avoid 32-bit FP problems
            snippet_transitions = snippet_transitions.astype(np.float32)  # weird but necessary
            # since the sample times get large enough that the step size between float32
            # might be more than 1

            history_frames = snippet_frames[:NS_BROWNIAN_N_FRAMES_PER_IMAGE, ...]
            target_frames = snippet_frames[NS_BROWNIAN_N_FRAMES_PER_IMAGE:, ...]

            start_sample, end_sample = synchro_block.get_snippet_sample_times(stim_idx - 1, stim_idx)

            bin_start = max(start_sample, int(np.ceil(snippet_transitions[0])))
            bin_end = min(end_sample, int(np.floor(snippet_transitions[-1])))

            spike_bins = np.r_[bin_start:bin_end:self.bin_width]

            # bin spikes for this
            binned_spikes = movie_bin_spikes_multiple_cells2(
                block_data.vision_dataset,
                self.cell_ids_to_bin[block_data.vision_name],
                spike_bins,
                jitter_time_amount=self.jitter_spike_times_lookup[block_data.vision_name]
            )

            if self.image_rescale_lambda is not None:
                history_frames = self.image_rescale_lambda(history_frames)
                target_frames = self.image_rescale_lambda(target_frames)

            return history_frames, target_frames, snippet_transitions, spike_bins, binned_spikes

        else:
            # trickier case; some work needs to be done to assemble
            # a batch with same number of spike bins for each example
            loopover = to_get
            if isinstance(to_get, slice):
                stop = self.length if to_get.stop is None else to_get.stop
                step = 1 if to_get.step is None else to_get.step
                loopover = range(to_get.start, stop, step)

            history_frames_buffer = []  # type: List[np.ndarray]
            target_frames_buffer = []  # type: List[np.ndarray]
            frame_transition_buffer = []  # type: List[np.ndarray]
            spike_bins_buffer = []  # type: List[np.ndarray]

            for ix in loopover:
                block_data, stim_idx = self._get_relev_block(ix)
                synchro_block = block_data.timing_synchro
                snippet_frames, snippet_transitions = synchro_block.get_snippet_frames(
                    stim_idx - 1, stim_idx, crop_h=self.crop_h_ix, crop_w=self.crop_w_ix
                )

                # FIXME may need to change timekeeping to ms to avoid 32-bit FP problems
                snippet_transitions = snippet_transitions.astype(np.float32)

                history_frames_buffer.append(snippet_frames[:NS_BROWNIAN_N_FRAMES_PER_IMAGE, ...])
                target_frames_buffer.append(snippet_frames[NS_BROWNIAN_N_FRAMES_PER_IMAGE:, ...])
                frame_transition_buffer.append(snippet_transitions)

                start_sample, end_sample = synchro_block.get_snippet_sample_times(stim_idx - 1, stim_idx)

                bin_start = max(start_sample, int(np.ceil(snippet_transitions[0])))
                bin_end = min(end_sample, int(np.floor(snippet_transitions[-1])))

                spike_bins_buffer.append(np.r_[bin_start:bin_end:self.bin_width])

            min_spike_bins = np.min([x.shape[0] for x in spike_bins_buffer])
            batched_spike_bin_times = np.zeros((len(loopover), min_spike_bins), dtype=np.int64)
            binned_spikes_buffer = []  # type: List[np.ndarray]
            for buffer_ix, ix in enumerate(loopover):
                block_data, stim_idx = self._get_relev_block(ix)

                orig_spike_bins = spike_bins_buffer[buffer_ix]
                n_bins_to_trim = orig_spike_bins.shape[0] - min_spike_bins
                trimmed_spike_bins = orig_spike_bins[n_bins_to_trim:]

                binned_spikes = movie_bin_spikes_multiple_cells2(
                    block_data.vision_dataset,
                    self.cell_ids_to_bin[block_data.vision_name],
                    trimmed_spike_bins,
                    jitter_time_amount=self.jitter_spike_times_lookup[block_data.vision_name]
                )

                batched_spike_bin_times[buffer_ix, :] = trimmed_spike_bins
                binned_spikes_buffer.append(binned_spikes)

            history_frames = np.array(history_frames_buffer, dtype=np.float32)
            target_frames = np.array(target_frames_buffer, dtype=np.float32)

            if self.image_rescale_lambda is not None:
                history_frames = self.image_rescale_lambda(history_frames)
                target_frames = self.image_rescale_lambda(target_frames)

            return (
                history_frames,
                target_frames,
                np.array(frame_transition_buffer, dtype=np.float32),
                batched_spike_bin_times,
                np.array(binned_spikes_buffer, dtype=np.float32)
            )


class FrameRateJitteredMovieBatchDataloader:

    def __init__(self,
                 loaded_brownian_movies: List[LoadedBrownianMovies],
                 cell_matching: OrderedMatchedCellsStruct,
                 dataset_partition: PartitionType,
                 crop_h_ix: Optional[Tuple[int, int]] = None,
                 crop_w_ix: Optional[Tuple[int, int]] = None,
                 image_rescale_lambda: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 time_jitter_spikes: float = 0.0):

        self.cell_matching = cell_matching

        self.crop_h_ix = crop_h_ix
        self.crop_w_ix = crop_w_ix

        self.image_rescale_lambda = image_rescale_lambda

        self.time_jitter_spikes = time_jitter_spikes

        self.data_blocks = []  # type: List[_JitterReconstructionBrownianMovieBlock]
        self.data_block_sizes = []  # type: List[int]
        self.data_block_cumsum = []  # type: List[int]
        self.cell_ids_to_bin = {}  # type: Dict[str, List[List[int]]]
        self.length = 0

        for loaded_brownian_movie in loaded_brownian_movies:
            if dataset_partition == PartitionType.TEST_PARTITION:
                synchro_section_dict = loaded_brownian_movie.test_blocks
            elif dataset_partition == PartitionType.HELDOUT_PARTITION:
                synchro_section_dict = loaded_brownian_movie.heldout_blocks
            else:
                synchro_section_dict = loaded_brownian_movie.train_blocks

            for block_num, synchro_block in synchro_section_dict.items():
                self.data_blocks.append(_JitterReconstructionBrownianMovieBlock(
                    loaded_brownian_movie.name,
                    loaded_brownian_movie.dataset,
                    synchro_block
                ))

                n_stimuli = synchro_block.n_stimuli - 1

                self.data_block_cumsum.append(sum(self.data_block_sizes))
                self.data_block_sizes.append(n_stimuli)
                self.length += n_stimuli

            matched_cell_ids = []
            for ct in cell_matching.get_cell_types():
                matched_cell_ids.extend(cell_matching.get_cell_order_for_ds_name(
                    ct, loaded_brownian_movie.name))
            self.cell_ids_to_bin[loaded_brownian_movie.name] = matched_cell_ids

    def _get_relev_block(self, ix: int) \
            -> Tuple[_JitterReconstructionBrownianMovieBlock, int]:

        block_ix = 0
        while (block_ix + 1) < len(self.data_block_cumsum) and \
                self.data_block_cumsum[block_ix + 1] <= ix:
            block_ix += 1

        from_start_of_block = ix - self.data_block_cumsum[block_ix]

        # +1 is because we have to skip the first stimulus in the block
        # since we need the stimulus history
        return self.data_blocks[block_ix], from_start_of_block + 1

    def __len__(self):
        return self.length

    def __getitem__(self, to_get) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if isinstance(to_get, int) or isinstance(to_get, np.int64):
            block_data, stim_idx = self._get_relev_block(to_get)
            synchro_block = block_data.timing_synchro

            # snippet_transitions is in units of electrical samples since the beginning of the
            # recording, but is non-integer valued
            snippet_frames, snippet_transitions = synchro_block.get_snippet_frames(
                stim_idx - 1, stim_idx, crop_h=self.crop_h_ix, crop_w=self.crop_w_ix
            )

            snippet_transitions = snippet_transitions.astype(np.float32)  # weird but necessary
            # since the sample times get large enough that the step size between float32
            # might be more than 1

            history_frames = snippet_frames[:NS_BROWNIAN_N_FRAMES_PER_IMAGE, ...]
            target_frames = snippet_frames[NS_BROWNIAN_N_FRAMES_PER_IMAGE:, ...]

            # bin spikes for this
            binned_spikes = movie_bin_spikes_multiple_cells2(
                block_data.vision_dataset,
                self.cell_ids_to_bin[block_data.vision_name],
                snippet_transitions,
                jitter_time_amount=self.time_jitter_spikes
            )

            if self.image_rescale_lambda is not None:
                history_frames = self.image_rescale_lambda(history_frames)
                target_frames = self.image_rescale_lambda(target_frames)

            return history_frames, target_frames, binned_spikes
        else:
            # need to do some work to assemble
            loopover = to_get
            if isinstance(to_get, slice):
                stop = self.length if to_get.stop is None else to_get.stop
                step = 1 if to_get.step is None else to_get.step
                loopover = range(to_get.start, stop, step)

            history_frames_buffer = []  # type: List[np.ndarray]
            target_frames_buffer = []  # type: List[np.ndarray]
            binned_spikes_buffer = []  # type: List[np.ndarray]

            for ix in loopover:
                block_data, stim_idx = self._get_relev_block(ix)
                synchro_block = block_data.timing_synchro
                snippet_frames, snippet_transitions = synchro_block.get_snippet_frames(
                    stim_idx - 1, stim_idx, crop_h=self.crop_h_ix, crop_w=self.crop_w_ix
                )

                # FIXME may need to change timekeeping to ms to avoid 32-bit FP problems
                snippet_transitions = snippet_transitions.astype(np.float32)

                history_frames_buffer.append(snippet_frames[:NS_BROWNIAN_N_FRAMES_PER_IMAGE, ...])
                target_frames_buffer.append(snippet_frames[NS_BROWNIAN_N_FRAMES_PER_IMAGE:, ...])

                binned_spikes = movie_bin_spikes_multiple_cells2(
                    block_data.vision_dataset,
                    self.cell_ids_to_bin[block_data.vision_name],
                    snippet_transitions,
                    jitter_time_amount=self.time_jitter_spikes
                )

                binned_spikes_buffer.append(binned_spikes)

            history_frames = np.array(history_frames_buffer, dtype=np.float32)
            target_frames = np.array(target_frames_buffer, dtype=np.float32)

            if self.image_rescale_lambda is not None:
                history_frames = self.image_rescale_lambda(history_frames)
                target_frames = self.image_rescale_lambda(target_frames)

            return (
                history_frames,
                target_frames,
                np.array(binned_spikes_buffer, dtype=np.float32)
            )
