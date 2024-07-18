import numpy as np
from typing import List, Callable, Tuple, Dict, Optional, Any, Set, Union
import visionloader as vl

from lib.dataset_specific_ttl_corrections.ttl_interval_constants import FLASH_TRAINING_BLOCK_SIZE, \
    FLASH_TEST_BLOCK_SIZE, N_BLOCKS

from abc import ABC


class WithinBlockMatching(ABC):

    def __init__(self):
        self.frame_list = [] # type: List[int]
        self.bin_list = [] # type: List[Any]

    def __len__(self):
        raise NotImplementedError()

    def append_frames(self, movie_frame_idx: int, interval_bins: Any) -> None:
        raise NotImplementedError()

    def copy_by_set(self, partition_set: Set[int]) -> 'WithinBlockMatching':
        raise NotImplementedError()

    def generate_block_index_set(self) -> Set[int]:
        raise NotImplementedError()


class WithinBlockMatchedFrameBins(WithinBlockMatching):

    def __init__(self,
                 block_num: int,
                 frame_list: Optional[List[int]] = None,
                 bin_cutoff_list: Optional[List[np.ndarray]] = None):
        self.block_num = block_num
        self.frame_list = [] if frame_list is None else frame_list
        self.bin_list = [] if bin_cutoff_list is None else bin_cutoff_list

    def append_frames(self, movie_frame_idx: int, bin_times: np.ndarray) -> None:
        self.frame_list.append(movie_frame_idx)
        self.bin_list.append(bin_times)

    def __len__(self):
        return len(self.frame_list)

    def copy_by_set(self, partition_set: Set[int]) -> 'WithinBlockMatchedFrameBins':

        within_set = WithinBlockMatchedFrameBins(self.block_num)
        for frame_idx, bin_cutoffs in zip(self.frame_list, self.bin_list):
            if (frame_idx - (self.block_num * FLASH_TRAINING_BLOCK_SIZE)) in partition_set:
                within_set.append_frames(frame_idx, bin_cutoffs)
        return within_set

    def generate_block_index_set(self) -> Set[int]:
        return set([(x - self.block_num * FLASH_TRAINING_BLOCK_SIZE) for x in self.frame_list])

    def __repr__(self):
        return 'WithinBlockMatchedFrameBins({0}, {1}...{2}, {3}...{4})'.format(self.block_num,
                                                                               self.frame_list[0], self.frame_list[-1],
                                                                               self.bin_list[0], self.bin_list[-1])


def _make_timebin_boundaries(frame_transition_ttl_time: Union[int, float],
                             bin_width: Union[int, float],
                             n_bins_before: int,
                             n_bins_after: int) -> np.ndarray:
    '''
    Produces time bin boundaries for the time-domain flash model (i.e. for GLMs)

    Bin width can be non-integer, but this function converts the resulting bin time
        cutoffs into integer sample numbers

    :param frame_transition_ttl_time: sample num. at which the frame transition occured
    :param bin_width: width of the bins, in units of 20 kHz samples
    :param n_bins_before: number of bins to include before the frame transition
    :param n_bins_after: number of bins to include after the frame transition
    :return: array of time bins, must be integer valued
    '''

    bin_count = np.r_[-n_bins_before:(1 + n_bins_after)]
    all_bin_boundaries = bin_width * bin_count

    timebins_unknown_type = frame_transition_ttl_time + all_bin_boundaries
    if np.issubdtype(timebins_unknown_type.dtype, np.integer):
        return timebins_unknown_type
    else:
        return np.around(timebins_unknown_type, decimals=0).astype(np.int64)


def block_structure_intervals_no_change_bin_timebins(vision_dataset: vl.VisionCellDataTable,
                                                     bin_width: int,
                                                     n_bins_before: int,
                                                     n_bins_after: int) \
        -> Dict[int, WithinBlockMatchedFrameBins]:
    '''
    Given the trigger times, generate the appropriate sized spike intervals
        and correct for missing TTLs and the like

    This is the default case, in which case there are no missing TTLs

    :return:
    '''
    non_interpolated_ttls = vision_dataset.get_ttl_times()

    frame_intervals_by_block = {}  # type: Dict[int, WithinBlockMatchedFrameBins]

    start = 0
    frame_num_counter = 0
    for i in range(N_BLOCKS):
        good_block = WithinBlockMatchedFrameBins(i)

        curr_block_max = start + FLASH_TRAINING_BLOCK_SIZE
        while start < min(curr_block_max, len(non_interpolated_ttls)):
            frame_transition_time = non_interpolated_ttls[start]

            bin_edges = _make_timebin_boundaries(frame_transition_time,
                                                 bin_width,
                                                 n_bins_before,
                                                 n_bins_after)

            good_block.append_frames(frame_num_counter, bin_edges)
            frame_num_counter += 1
            start += 1

        frame_intervals_by_block[i] = good_block
        start += FLASH_TEST_BLOCK_SIZE

    return frame_intervals_by_block


def make_block_struct_corrected_timebin_function(bad_blocks_with_correction: Dict[int, int]) \
        -> Callable[[vl.VisionCellDataTable, int, int, int], Dict[int, WithinBlockMatchedFrameBins]]:
    def callable_fn(vision_dataset: vl.VisionCellDataTable,
                    bin_width: int,
                    n_bins_before: int,
                    n_bins_after: int) \
            -> Dict[int, WithinBlockMatchedFrameBins]:

        non_interpolated_ttls = vision_dataset.get_ttl_times()

        frame_intervals_by_block = {}  # type: Dict[int, WithinBlockMatchedFrameBins]

        start = 0
        frame_num_counter = 0
        for i in range(N_BLOCKS):
            if i in bad_blocks_with_correction:
                frame_num_counter += FLASH_TRAINING_BLOCK_SIZE
                start += (FLASH_TRAINING_BLOCK_SIZE + bad_blocks_with_correction[i] + FLASH_TEST_BLOCK_SIZE)
            else:
                good_block = WithinBlockMatchedFrameBins(i)

                curr_block_max = start + FLASH_TRAINING_BLOCK_SIZE
                while start < min(curr_block_max, len(non_interpolated_ttls)):
                    frame_transition_time = non_interpolated_ttls[start]

                    bin_edges = _make_timebin_boundaries(frame_transition_time,
                                                         bin_width,
                                                         n_bins_before,
                                                         n_bins_after)

                    good_block.append_frames(frame_num_counter, bin_edges)
                    frame_num_counter += 1
                    start += 1

                frame_intervals_by_block[i] = good_block
                start += FLASH_TEST_BLOCK_SIZE

        return frame_intervals_by_block

    return callable_fn


class WithinBlockMatchedFrameInterval(WithinBlockMatching):
    '''
    Keeps track of all of the intervals for single-bin flash trials
        for a given datarun

        (i.e. same bin format as Nora's trials)
    '''

    def __init__(self,
                 block_num: int,
                 frame_list: Optional[List[int]] = None,
                 interval_list: Optional[List[Tuple[int, int]]] = None):
        self.block_num = block_num
        self.frame_list = [] if frame_list is None else frame_list
        self.interval_list = [] if interval_list is None else interval_list

    def append_frames(self, movie_frame_idx: int, interval: Tuple[int, int]) -> None:
        self.frame_list.append(movie_frame_idx)
        self.interval_list.append(interval)

    def __len__(self):
        return len(self.frame_list)

    def copy_by_set(self, partition_set: Set[int]) -> 'WithinBlockMatchedFrameInterval':

        within_set = WithinBlockMatchedFrameInterval(self.block_num)
        for frame_idx, interval in zip(self.frame_list, self.interval_list):
            if (frame_idx - (self.block_num * FLASH_TRAINING_BLOCK_SIZE)) in partition_set:
                within_set.append_frames(frame_idx, interval)
        return within_set

    @property
    def bin_list(self) -> np.ndarray:
        return np.array(self.interval_list, dtype=np.int64)

    def generate_block_index_set(self) -> Set[int]:
        return set([(x - self.block_num * FLASH_TRAINING_BLOCK_SIZE) for x in self.frame_list])

    def __repr__(self):
        if len(self.frame_list) == 0:
            return 'WithinBlockMatchedFrameInterval({0},None)'.format(self.block_num)
        return 'WithinBlockMatchedFrameInterval({0},{1}:{2})'.format(self.block_num, np.min(self.frame_list),
                                                                     np.max(self.frame_list))

class RepeatMatchedFrame(ABC):

    def __init__(self):
        self.frame_list = [] # type: List[int]

    def __len__(self):
        raise NotImplementedError

    def append_frames(self, movie_frame_idx: int, interval_bins: Any) -> None:
        raise NotImplementedError


class RepeatMatchedFrameInterval(RepeatMatchedFrame):

    def __init__(self,
                 frame_list: Optional[List[int]] = None,
                 interval_list: Optional[List[Tuple[int, int]]] = None):
        self.frame_list = [] if frame_list is None else frame_list
        self.interval_list = [] if interval_list is None else interval_list

    def append_frames(self, movie_frame_idx: int, interval: Tuple[int, int]) -> None:
        self.frame_list.append(movie_frame_idx)
        self.interval_list.append(interval)

    def __len__(self):
        return len(self.frame_list)

    def __repr__(self):
        return 'RepeatMatchedFrameInterval({0}:{1})'.format(np.min(self.frame_list), np.max(self.frame_list))


class RepeatMatchedFrameTimebins(RepeatMatchedFrame):

    def __init__(self,
                 frame_list: Optional[List[int]] = None,
                 bin_cutoff_list: Optional[List[np.ndarray]] = None):

        if (frame_list is None and bin_cutoff_list is not None) or \
                (frame_list is not None and bin_cutoff_list is None):
            raise ValueError("frame_list and bin_cutoff_list must both either be specified or not specified")

        if frame_list is None:
            self.frame_list = []
            self.bin_cutoff_list = []
        else:
            if len(frame_list) != len(bin_cutoff_list):
                raise ValueError("frame_list and bin_cutoff_list must have the same length")
            self.frame_list = frame_list
            self.bin_cutoff_list = bin_cutoff_list

    def append_frames(self, movie_frame_idx: int, bin_times: np.ndarray) -> None:
        self.frame_list.append(movie_frame_idx)
        self.bin_cutoff_list.append(bin_times)

    def __len__(self):
        return len(self.frame_list)

    def __repr__(self):
        return 'RepeatMatchedFrameTimebins({0}:{1})'.format(np.min(self.frame_list), np.max(self.frame_list))
