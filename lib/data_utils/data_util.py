import visionloader as vl
import numpy as np

from typing import List, Union, Dict, Tuple, Callable, Optional, Type, Sequence, Any


import torch

from lib.data_utils.matched_cells_struct import OrderedMatchedCellsStruct
from lib.data_utils.sta_metadata import RGB_CONVERSION

import spikebinning
from fastconv import conv2d

from lib.dataset_config_parser import dataset_config_parser as dcp
from lib.dataset_specific_ttl_corrections.block_structure_ttl_corrections import WithinBlockMatching


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    From stackoverflow

    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def batch_blur_image(batched_images: np.ndarray,
                     blur_kernel: np.ndarray) -> np.ndarray:
    '''
    Uses "reflect padding"
    :param batched_images:
    :param blur_kernel:
    :return:
    '''

    kernel_size = blur_kernel.shape[0]
    half_kernel = int(kernel_size // 2)

    # then pad the images using reflect
    batch, height, width = batched_images.shape

    padded = np.zeros((batch, height + 2 * half_kernel, width + 2 * half_kernel),
                      dtype=np.float32)
    padded[:, half_kernel:half_kernel + height, half_kernel:half_kernel + width] = batched_images

    padded[:, 0:half_kernel, half_kernel:-half_kernel] = batched_images[:, 0:half_kernel, :][:, ::-1, :]
    padded[:, height + half_kernel:, half_kernel:-half_kernel] = batched_images[:, height - half_kernel:height, :][:,
                                                                 ::-1, :]
    padded[:, half_kernel:-half_kernel, 0:half_kernel] = batched_images[:, :, 0:half_kernel][:, :, ::-1]
    padded[:, half_kernel:-half_kernel, width + half_kernel:] = batched_images[:, :, width - half_kernel:width][:, :,
                                                                ::-1]

    return conv2d.batch_parallel_2Dconv_valid(padded, blur_kernel)


def batch_downsample_image(batched_images: np.ndarray,
                           downsample_factor: int) -> np.ndarray:
    '''

    Uses "reflect" padding

    :param batched_images: shape (batch, height, width)
    :param downsample_factor:
    :return: shape (batch, height // downsample_factor, width // downsample_factor)
    '''

    # first set up the antialiasing Gaussian kernel
    antialiasing_sigma = (downsample_factor - 1) / 2.0
    kernel_size_raw = int(np.ceil(antialiasing_sigma * 2 * 4))
    kernel_size = kernel_size_raw if kernel_size_raw % 2 == 1 else kernel_size_raw + 1
    gaussian_kernel = matlab_style_gauss2D((kernel_size, kernel_size), sigma=antialiasing_sigma)
    half_kernel = int(kernel_size // 2)

    # then pad the images using reflect
    batch, height, width = batched_images.shape

    padded = np.zeros((batch, height + 2 * half_kernel, width + 2 * half_kernel),
                      dtype=np.float32)
    padded[:, half_kernel:half_kernel + height, half_kernel:half_kernel + width] = batched_images

    padded[:, 0:half_kernel, half_kernel:-half_kernel] = batched_images[:, 0:half_kernel, :][:, ::-1, :]
    padded[:, height + half_kernel:, half_kernel:-half_kernel] = batched_images[:, height - half_kernel:height, :][:,
                                                                 ::-1, :]
    padded[:, half_kernel:-half_kernel, 0:half_kernel] = batched_images[:, :, 0:half_kernel][:, :, ::-1]
    padded[:, half_kernel:-half_kernel, width + half_kernel:] = batched_images[:, :, width - half_kernel:width][:, :,
                                                                ::-1]

    return conv2d.batch_parallel_2Dconv_valid(padded, gaussian_kernel)[:, ::downsample_factor,
           ::downsample_factor]


def fast_get_spike_count_matrix_cpp(vision_dataset: vl.VisionCellDataTable,
                                    good_cells_list: List[int],
                                    sample_interval_list_increasing: List[Tuple[int, int]]) -> np.ndarray:
    spikes_by_cell_id = {}
    for cell_id in good_cells_list:
        spike_times_ptr = vision_dataset.get_spike_times_for_cell(cell_id)
        spike_times_copy = np.copy(spike_times_ptr)
        spikes_by_cell_id[cell_id] = spike_times_copy

    binned_spikes_per_flash = spikebinning.bin_spikes_intervals(spikes_by_cell_id,
                                                                good_cells_list,
                                                                sample_interval_list_increasing)
    # shape (n_intervals, n_cells)
    return binned_spikes_per_flash.T


def _bin_spikes_into_timebins(multicell_id_to_idx: Dict[int, int],
                              spikes_by_cell_id: Dict[int, np.ndarray],
                              spike_idx_offset: Dict[int, int],
                              n_cells_total: int,
                              bin_edge_cutoffs: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    '''
    Bins spikes for cells into the bins specified by bin_edge_cutoffs, and increments
        the spike offset counters in spike_idx_offset

    HAS SIDE EFFECT: MUTATES spike_idx_offset by possibly incrementing offsets for every cell

    :param multicell_id_to_idx:
    :param spikes_by_cell_id:
    :param spike_idx_offset:
    :return:
    '''

    n_bins = bin_edge_cutoffs.shape[0] - 1
    output_buffer = np.zeros((n_cells_total, n_bins), dtype=np.int32)

    for cell_id, map_to_idx in multicell_id_to_idx.items():

        spikes_for_current_cell = spikes_by_cell_id[cell_id]

        # advance the counter dict
        offset_i = spike_idx_offset[cell_id]

        for bin_edge_idx_low in range(n_bins - 1):

            bin_edge_idx_high = bin_edge_idx_low + 1
            bin_low, bin_high = bin_edge_cutoffs[bin_edge_idx_low], bin_edge_cutoffs[bin_edge_idx_high]

            n_spikes_in_bin = 0

            while offset_i < len(spikes_for_current_cell) and spikes_for_current_cell[offset_i] < bin_low:
                offset_i += 1

            while offset_i < len(spikes_for_current_cell) and spikes_for_current_cell[offset_i] < bin_high:
                n_spikes_in_bin += 1
                offset_i += 1

            output_buffer[map_to_idx, bin_edge_idx_low] += n_spikes_in_bin

        spike_idx_offset[cell_id] = offset_i

    return output_buffer, spike_idx_offset


def _generate_initial_bin_dicts(vision_dataset: vl.VisionCellDataTable,
                                good_cell_ordering: List[List[int]]) \
        -> Tuple[Dict[int, np.ndarray], Dict[int, int], Dict[int, int]]:
    multicell_id_to_idx = {}  # type: Dict[int, int]
    spikes_by_cell_id = {}
    spike_idx_offset = {}
    for idx, cell_id_list in enumerate(good_cell_ordering):
        for cell_id in cell_id_list:
            multicell_id_to_idx[cell_id] = idx

            spike_times_ptr = vision_dataset.get_spike_times_for_cell(cell_id)
            spike_times_copy = np.copy(spike_times_ptr)

            spikes_by_cell_id[cell_id] = spike_times_copy
            spike_idx_offset[cell_id] = 0

    return spikes_by_cell_id, spike_idx_offset, multicell_id_to_idx


def fast_time_domain_trial_struct_bin_spikes_multicell(vision_dataset: vl.VisionCellDataTable,
                                                       good_cell_ordering: List[List[int]],
                                                       trial_bin_list_increasing: np.ndarray) -> np.ndarray:
    '''
    Function for binning spikes into multiple bins (for time-domain model for flashes), if we include
        likely duplicates with nonoverlapping spike times

    Allows for overlapping trials (the beginning of trial i+1 is allowed to overlap with the end of
        of trial i)

    :param vision_dataset: vl.VisionCellDataTable
    :param good_cell_ordering: ordering of cells, with duplicates
    :param trial_bin_list_increasing: shape (n_trials, n_bins_edges = n_bins + 1), integer-valued
        corresponding to bin cutoff sample times
    :return:
    '''

    merged_spike_times_by_cell_id = {}  # type: Dict[int, np.ndarray]
    cell_order = []  # type: List[int]
    for dup_list in good_cell_ordering:
        cell_id = dup_list[0]
        cell_order.append(cell_id)
        if len(dup_list) == 1:
            merged_spike_times_by_cell_id[cell_id] = vision_dataset.get_spike_times_for_cell(cell_id)
        else:
            spike_vector_list = [vision_dataset.get_spike_times_for_cell(x) for x in dup_list]
            merged_spike_vector = spikebinning.merge_multiple_sorted_array(*spike_vector_list)
            merged_spike_times_by_cell_id[cell_id] = merged_spike_vector

    return spikebinning.bin_spikes_consecutive_trials(merged_spike_times_by_cell_id,
                                                      cell_order,
                                                      trial_bin_list_increasing)


def merge_multicell_spike_vectors(vision_dataset: vl.VisionCellDataTable,
                                  good_cell_ordering: List[List[int]]) \
        -> Tuple[Dict[int, np.ndarray], List[int]]:
    '''

    :param vision_dataset:
    :param good_cell_ordering:
    :return:
    '''
    # first we need to get the spike vectors
    # and then merge the oversplit cells
    merged_spike_vector_dict = {}  # type: Dict[int, np.ndarray]
    cell_order = []  # type: List[int]
    for cell_id_list in good_cell_ordering:
        first_cell_id = cell_id_list[0]
        cell_order.append(first_cell_id)
        if len(cell_id_list) > 1:
            to_merge_vectors = [vision_dataset.get_spike_times_for_cell(cell_id) for cell_id in cell_id_list]
            merged_spike_vector_dict[first_cell_id] = spikebinning.merge_multiple_sorted_array(to_merge_vectors)
        else:
            merged_spike_vector_dict[first_cell_id] = vision_dataset.get_spike_times_for_cell(first_cell_id)

    return merged_spike_vector_dict, cell_order


def merge_multicell_spike_vectors2(vision_dataset: vl.VisionCellDataTable,
                                   ds_name: str,
                                   cells_ordered: OrderedMatchedCellsStruct,
                                   wn_cell_ids_in_order: List[int]) \
        -> Tuple[Dict[int, np.ndarray], List[int]]:

    reduced_cells_with_ordering = [cells_ordered.get_match_ids_for_ds(target_cell, ds_name)
                                   for target_cell in wn_cell_ids_in_order]

    merged_spike_vector_dict = {}  # type: Dict[int, np.ndarray]
    for wn_cell_id, ns_cell_id_list in zip(wn_cell_ids_in_order, reduced_cells_with_ordering):

        if len(ns_cell_id_list) > 1:
            to_merge_vectors = [vision_dataset.get_spike_times_for_cell(cell_id) for cell_id in ns_cell_id_list]
            merged_spike_vector_dict[wn_cell_id] = spikebinning.merge_multiple_sorted_array(to_merge_vectors)
        else:
            merged_spike_vector_dict[wn_cell_id] = vision_dataset.get_spike_times_for_cell(ns_cell_id_list[0])

    return merged_spike_vector_dict, wn_cell_ids_in_order


def time_domain_trials_bin_spikes_multiple_cells2(vision_dataset: vl.VisionCellDataTable,
                                                  good_cell_ordering: List[List[int]],
                                                  trial_bin_edges: np.ndarray) -> np.ndarray:
    '''

    :param vision_dataset:
    :param good_cell_ordering:
    :param bin_list: shape (n_trials, n_bin_edges)
    :return: shape (n_trials, n_cells, n_bins)
    '''

    # first we need to get the spike vectors
    # and then merge the oversplit cells
    merged_spike_vector_dict, cell_order = merge_multicell_spike_vectors(vision_dataset,
                                                                         good_cell_ordering)

    # then do spike binning
    output_matrix = spikebinning.bin_spikes_trials_parallel(merged_spike_vector_dict,
                                                            cell_order,
                                                            trial_bin_edges)

    return output_matrix


def time_domain_bin_spikes_multiple_cells(vision_dataset: vl.VisionCellDataTable,
                                          good_cell_ordering: List[List[int]],
                                          sample_bin_list_increasing: List[np.ndarray]) -> np.ndarray:
    '''
    Function for binning spikes into multiple bins (for time-domain model for flashes), if we include
        likely duplicates with nonoverlapping spike times

    :param vision_dataset: vl.VisionCellDataTable
    :param good_cell_ordering: ordering of cells, with duplicates
    :param sample_bin_list_increasing: List of bin edges, each entry has shape (n_bins + 1, )
        Bin edges must be integer valued. Each array must have the same shape
    :return:
    '''

    n_images = len(sample_bin_list_increasing)
    n_cells = len(good_cell_ordering)
    n_bins_per_trial = sample_bin_list_increasing[0].shape[0] - 1

    # generate output matrix
    output_matrix = np.zeros((n_images, n_cells, n_bins_per_trial),
                             dtype=np.int32)

    # generate multiple cell cell_id to idx mapping
    # break the abstraction layer of vl.VisionCellDataTable for performance purposes...
    # keep grab all of the spike times for the cells that we care about
    # and keep a deep copy of that in a Dict

    # spike_idx_offset contains the index of the next spike whose time
    # we haven't looked at yet. This way we can count spikes while
    # only making a single pass through the data for each cell
    spikes_by_cell_id, spike_idx_offset, multicell_id_to_idx = _generate_initial_bin_dicts(vision_dataset,
                                                                                           good_cell_ordering)

    for bin_idx, bin_boundaries in enumerate(sample_bin_list_increasing):
        single_bin_packed, spike_idx_offset = _bin_spikes_into_timebins(multicell_id_to_idx,
                                                                        spikes_by_cell_id,
                                                                        spike_idx_offset,
                                                                        n_cells,
                                                                        bin_boundaries)
        output_matrix[bin_idx, ...] = single_bin_packed

    return output_matrix


def fast_get_spike_count_multiple_cells(vision_dataset: vl.VisionCellDataTable,
                                        good_cell_ordering: List[List[int]],
                                        sample_interval_list_increasing: List[Tuple[int, int]]) -> np.ndarray:
    '''
    Function for binning spikes, if we include likely duplicates with nonoverlapping spike times

    We don't need a sophisticated algorithm for this, because we just care about the onset spikes
        with no sub-bin temporal resolution
    :param vision_dataset:
    :param good_cell_ordering:
    :param sample_interval_list_increasing:
    :return:
    '''
    n_images = len(sample_interval_list_increasing)
    n_cells = len(good_cell_ordering)

    # generate output matrix
    output_matrix = np.zeros((n_images, n_cells),
                             dtype=np.int32)

    # generate multiple cell cell_id to idx mapping
    # break the abstraction layer of vl.VisionCellDataTable for performance purposes...
    # keep grab all of the spike times for the cells that we care about
    # and keep a deep copy of that in a Dict

    # spike_idx_offset contains the index of the next spike whose time
    # we haven't looked at yet. This way we can count spikes while
    # only making a single pass through the data for each cell
    spikes_by_cell_id, spike_idx_offset, multicell_id_to_idx = _generate_initial_bin_dicts(vision_dataset,
                                                                                           good_cell_ordering)

    for interval_idx, interval in enumerate(sample_interval_list_increasing):

        for cell_id, map_to_idx in multicell_id_to_idx.items():

            spikes_for_current_cell = spikes_by_cell_id[cell_id]

            # advance the counter dict
            i = spike_idx_offset[cell_id]
            while i < len(spikes_for_current_cell) and spikes_for_current_cell[i] < interval[0]:
                i += 1

            # now we're at the first sample within the interval
            # count spikes within the interval
            n_spikes_in_interval = 0
            while i < len(spikes_for_current_cell) and spikes_for_current_cell[i] < interval[1]:
                n_spikes_in_interval += 1
                i += 1

            # and now we're outside the interval, do cleanup
            spike_idx_offset[cell_id] = i

            output_matrix[interval_idx, map_to_idx] += n_spikes_in_interval

    return output_matrix  # shape (n_intervals, n_cells)


def fast_get_spike_count_matrix(vision_dataset: vl.VisionCellDataTable,
                                good_cells_list: List[int],
                                sample_interval_list_increasing: List[Tuple[int, int]]) -> np.ndarray:
    # break the abstraction layer of vl.VisionCellDataTable for performance purposes...
    n_images = len(sample_interval_list_increasing)
    n_cells = len(good_cells_list)

    output_matrix = np.zeros((n_images, n_cells),
                             dtype=np.float32)

    # keep grab all of the spike times for the cells that we care about
    # and keep a deep copy of that in a Dict

    # spike_idx_offset contains the index of the next spike whose time
    # we haven't looked at yet. This way we can count spikes while
    # only making a single pass through the data for each cell
    spikes_by_cell_id = {}
    spike_idx_offset = {}
    for cell_id in good_cells_list:
        spike_times_ptr = vision_dataset.get_spike_times_for_cell(cell_id)
        spike_times_copy = np.copy(spike_times_ptr)
        spikes_by_cell_id[cell_id] = spike_times_copy
        spike_idx_offset[cell_id] = 0

    # loop as follows: for each image interval
    spike_counts = np.zeros((n_cells,), dtype=np.double)
    for interval_idx, interval in enumerate(sample_interval_list_increasing):

        spike_counts[:] = 0.0

        for idx, cell_id in enumerate(good_cells_list):

            spikes_for_current_cell = spikes_by_cell_id[cell_id]

            # advance the counter dict
            i = spike_idx_offset[cell_id]
            while i < len(spikes_for_current_cell) and spikes_for_current_cell[i] < interval[0]:
                i += 1

            # now we're at the first sample within the interval
            # count spikes within the interval
            n_spikes_in_interval = 0
            while i < len(spikes_for_current_cell) and spikes_for_current_cell[i] < interval[1]:
                n_spikes_in_interval += 1
                i += 1

            # and now we're outside the interval, do cleanup
            spike_idx_offset[cell_id] = i

            # and store the data
            spike_counts[idx] = n_spikes_in_interval

        output_matrix[interval_idx, :] = spike_counts

    return output_matrix  # shape (n_intervals, n_cells)


def convert_color_movie_to_bw(all_frames_rgb: np.ndarray):
    '''

    :param all_frames_rgb:
    :return:
    '''
    return np.squeeze(all_frames_rgb @ RGB_CONVERSION.T, axis=3).astype(np.uint8)


def make_image_transform_lambda(image_new_low: Union[int, float],
                                image_new_high: Union[int, float],
                                image_type: Type) \
        -> Callable[[np.ndarray], np.ndarray]:
    # assumes that the original image is [0, 255]
    scale_factor = (image_new_high - image_new_low) / 255.0

    def image_transform(img: np.ndarray) -> np.ndarray:
        image_retype = img.astype(image_type)
        image_retype = np.multiply(image_retype, scale_factor, out=image_retype, casting='same_kind')
        image_retype = np.add(image_retype, image_new_low, out=image_retype, casting='same_kind')
        return image_retype

        #return (img * scale_factor + image_new_low).astype(image_type)

    return image_transform


def make_torch_image_transform_lambda(image_new_low: Union[int, float],
                                      image_new_high: Union[int, float]) \
        -> Callable[[torch.Tensor], torch.Tensor]:
    
    scale_factor = (image_new_high - image_new_low) / 255.0
    
    def image_transform(img: torch.Tensor) -> torch.Tensor:
        return img.mul_(scale_factor).add_(image_new_low)

    return image_transform


def make_torch_transform_to_recons_metric_lambda(image_orig_low: float,
                                                 image_orig_high: float) \
        -> Callable[[torch.Tensor], torch.Tensor]:

    delta = image_orig_high - image_orig_low
    def image_transform(img: torch.Tensor) -> torch.Tensor:
        return torch.clamp_((img.add_(-image_orig_low)).div_(delta), min=0.0, max=1.0)
    return image_transform


def partition_movie_block_matches_dataset(ds_movie_path: str,
                                          interv_by_block: Dict[int, WithinBlockMatching],
                                          test_block_section_list: Optional[
                                              List[dcp.MovieBlockSectionDescriptor]] = None,
                                          heldout_block_section_list: Optional[
                                              List[dcp.MovieBlockSectionDescriptor]] = None) \
        -> Tuple[List[WithinBlockMatching], List[WithinBlockMatching], List[
            WithinBlockMatching]]:
    train_blocks_list = []  # type: List[WithinBlockMatching]
    test_blocks_list = []  # type: List[WithinBlockMatching]
    heldout_blocks_list = []  # type: List[WithinBlockMatching]

    for block_num, frames_interval in interv_by_block.items():
        # determine if part of the block needs to go to the test partition
        all_block_offsets_in_block = frames_interval.generate_block_index_set()

        for block_descriptor in test_block_section_list:

            if ds_movie_path == block_descriptor.path and block_descriptor.block_num == block_num:
                # apply the partition
                partition_set = set(range(block_descriptor.block_low, block_descriptor.block_high))
                test_blocks_list.append(frames_interval.copy_by_set(partition_set))

                all_block_offsets_in_block.difference_update(partition_set)

        # determine if part of the block needs to go to the heldout partition
        for block_descriptor in heldout_block_section_list:

            if ds_movie_path == block_descriptor.path and block_descriptor.block_num == block_num:
                # apply the partition
                partition_set = set(range(block_descriptor.block_low, block_descriptor.block_high))
                heldout_blocks_list.append(frames_interval.copy_by_set(partition_set))

                all_block_offsets_in_block.difference_update(partition_set)

        # otherwise everything goes to the training partition
        if len(all_block_offsets_in_block) > 0:
            train_blocks_list.append(frames_interval.copy_by_set(all_block_offsets_in_block))

    return train_blocks_list, test_blocks_list, heldout_blocks_list
