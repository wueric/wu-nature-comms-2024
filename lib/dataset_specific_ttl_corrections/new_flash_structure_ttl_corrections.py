import numpy as np

from typing import List, Union, Tuple, Dict
import sys
from abc import ABC

from lib.dataset_config_parser.dataset_config_parser import LookupKey
from lib.dataset_specific_ttl_corrections.block_structure_ttl_corrections import WithinBlockMatchedFrameBins, \
    WithinBlockMatching, _make_timebin_boundaries, WithinBlockMatchedFrameInterval, RepeatMatchedFrame, \
    RepeatMatchedFrameTimebins, RepeatMatchedFrameInterval


class TriggerInterpMethod(ABC):

    def interpolate_parse_triggers(self,
                                   trigger_times: np.ndarray,
                                   start_trigger_idx: int,
                                   n_expected_triggers: int):
        raise NotImplementedError


class NoTriggerInterpMethod(TriggerInterpMethod):

    def __init__(self):
        pass

    def interpolate_parse_triggers(self,
                                   trigger_times: np.ndarray,
                                   start_trigger_idx: int,
                                   n_expected_triggers: int):
        # user asserts that no interpolation is required to handle this section
        relevant_trigger_chunk = trigger_times[start_trigger_idx:start_trigger_idx + n_expected_triggers]
        return relevant_trigger_chunk, start_trigger_idx + n_expected_triggers


class ManualTriggerInterpMethod(TriggerInterpMethod):
    '''
    Manually tell the algorithm which frame transition times to interpolate

    The algorithm will ONLY interpolate the manually identified frame transition
        times, all other frame transition times will be assumed to be correct.

    Only use this function when the trigger timing is really messed up and manual
        processing is the only option.

    Note that the first trigger cannot be interpolated
    '''

    def __init__(self,
                 manual_interpolated_frame_ix: List[int]):
        self.manual_interpolated_frame_ix = manual_interpolated_frame_ix

    def interpolate_parse_triggers(self,
                                   trigger_times: np.ndarray,
                                   start_trigger_idx: int,
                                   n_expected_triggers: int):
        '''

        :param trigger_times:
        :param start_trigger_idx:
        :param n_expected_triggers:
        :param expected_trigger_interval:
        :param manual_interpolated_frame_ix: list of frames (referenced w.r.t. the beginning of the block)
            whose onset time needs to be interpolated. Assumed to be sorted.

        :return:
        '''
        # Announce to stderr that we are doing trigger interpolation
        # Announce to stderr that we're doing trigger interpolation
        if len(self.manual_interpolated_frame_ix) == 0:
            # user asserts that no interpolation is required to handle this section
            relevant_trigger_chunk = trigger_times[start_trigger_idx:start_trigger_idx + n_expected_triggers]
            return relevant_trigger_chunk, start_trigger_idx + n_expected_triggers
        else:
            print("Performing manually-specified trigger interpolation", file=sys.stderr)
            interpolate_together = _get_consecutive_frames(self.manual_interpolated_frame_ix)

            triggers_with_interpolations = []
            block_trigger_count, real_trigger_used_count = 0, 0
            for together in interpolate_together:

                first_to_interpolate = together[0]

                # everything up to but excluding first_to_interpolate is deemed by the
                # user to be correct
                while block_trigger_count < first_to_interpolate:
                    triggers_with_interpolations.append(trigger_times[start_trigger_idx + real_trigger_used_count])
                    block_trigger_count += 1
                    real_trigger_used_count += 1

                # now interpolate the desired number of triggers
                upper_ix = start_trigger_idx + real_trigger_used_count
                lower_ix = upper_ix - 1

                upper_time = trigger_times[upper_ix]
                lower_time = trigger_times[lower_ix]

                extra_trigger_times = np.linspace(lower_time, upper_time,
                                                  num=len(together) + 2, endpoint=True)[1:-1]

                for x in extra_trigger_times:
                    triggers_with_interpolations.append(int(np.rint(x)))
                    block_trigger_count += 1

            while block_trigger_count < n_expected_triggers:
                triggers_with_interpolations.append(trigger_times[start_trigger_idx + real_trigger_used_count])
                block_trigger_count += 1
                real_trigger_used_count += 1

            return np.array(triggers_with_interpolations), start_trigger_idx + real_trigger_used_count


class AutomaticTriggerInterpMethod(TriggerInterpMethod):
    '''
    Assumption: there are only dropped triggers, never extra triggers

    Currently this function is the same as ns_brownian_parse_triggers;
        not clear that they should be the same or different yet

    It turns out that frame rate instability seems to be a much bigger problem
        for the flashed trials than for the continuous jitter stimulus

    This means that we need a more flexible way to parse triggers, where
        we have the option to either automatically interpolate triggers,
        or manually interpolate triggers
    '''

    def __init__(self,
                 expected_trigger_interval: Union[float, int],
                 tolerance_interval: Tuple[float, float] = (0.92, 1.08),
                 interpolation_tolerance: float = 0.08):

        self.expected_trigger_interval = expected_trigger_interval
        self.tolerance_interval = tolerance_interval
        self.interpolation_tolerance = interpolation_tolerance

    def interpolate_parse_triggers(self,
                                   trigger_times: np.ndarray,
                                   start_trigger_idx: int,
                                   n_expected_triggers: int):

        tol_low, tol_high = self.tolerance_interval

        relevant_trigger_chunk = trigger_times[start_trigger_idx:start_trigger_idx + n_expected_triggers]
        delta_triggers = relevant_trigger_chunk[1:] - relevant_trigger_chunk[:-1]

        does_not_need_interpolation = np.logical_and.reduce([
            delta_triggers > (tol_low * self.expected_trigger_interval),
            delta_triggers < (tol_high * self.expected_trigger_interval)
        ])

        if np.all(does_not_need_interpolation):
            return relevant_trigger_chunk, start_trigger_idx + n_expected_triggers

        # Announce to stderr that we are doing trigger interpolation
        # Announce to stderr that we're doing trigger interpolation
        print("Performing trigger interpolation", file=sys.stderr)
        prev_trigger_time = relevant_trigger_chunk[0]
        interpolated_trigger_times = [prev_trigger_time, ]
        interpolated_count, ix = 1, 1
        while interpolated_count < n_expected_triggers:

            trigger_time = relevant_trigger_chunk[ix]

            # figure out how many trigger intervals this corresponds to
            delta = trigger_time - prev_trigger_time
            num_intervals = delta / self.expected_trigger_interval

            is_close_to_one = (tol_low < num_intervals < tol_high)
            if is_close_to_one:
                interpolated_trigger_times.append(trigger_time)
                prev_trigger_time = trigger_time
                ix += 1
                interpolated_count += 1

            else:
                integer_multiple_delta = np.abs(np.rint(num_intervals) - num_intervals)
                if integer_multiple_delta < self.interpolation_tolerance:
                    rounded_int_n_intervals = int(np.rint(num_intervals))
                    extra_trigger_times = np.linspace(prev_trigger_time, trigger_time,
                                                      num=rounded_int_n_intervals + 1, endpoint=True)[1:]
                    for x in extra_trigger_times:
                        interpolated_trigger_times.append(int(np.rint(x)))
                        interpolated_count += 1

                    ix += 1

                    prev_trigger_time = trigger_time

                else:
                    # Announce to stderr that we can't figure out how to do trigger interpolation
                    print(f"Trigger interval was {integer_multiple_delta} away from integer multiple at index {ix};" + \
                          f" ({ix - start_trigger_idx} from the start of the interval)")
                    print("Cannot do trigger interpolation, gaps between trigger times are non-integer" + \
                          "multiple of expected trigger interval",
                          file=sys.stderr)

                    raise ValueError("Trigger interpolation failed")

        return np.array(interpolated_trigger_times), ix + start_trigger_idx


class FlashBlockDescriptor:

    def __init__(self,
                 n_stimuli: int,
                 is_data_block: bool,
                 include_block: bool = True,
                 is_full_block: bool = False,
                 interp_method: TriggerInterpMethod = NoTriggerInterpMethod()):
        self.n_stimuli = n_stimuli
        self.is_data_block = is_data_block
        self.include_block = include_block
        self.is_full_block = is_full_block
        self.interp_method = interp_method

    @property
    def n_triggers_expected(self):
        return self.n_stimuli

    @property
    def n_triggers_to_skip(self):
        return 0

    @property
    def n_data_blocks(self) -> int:
        raise NotImplementedError

    @property
    def n_data_stimuli(self) -> int:
        raise NotImplementedError


class FlashedDataBlockDescriptor(FlashBlockDescriptor):

    def __init__(self,
                 n_stimuli: int,
                 is_full_block: bool = True,
                 interp_method: TriggerInterpMethod = NoTriggerInterpMethod()):
        super().__init__(n_stimuli, True, include_block=True, is_full_block=is_full_block,
                         interp_method=interp_method)

    @property
    def n_data_blocks(self) -> int:
        return 1

    @property
    def n_data_stimuli(self) -> int:
        return self.n_stimuli


class FlashedRepeatBlockDescriptor(FlashBlockDescriptor):
    def __init__(self,
                 n_stimuli: int,
                 is_full_block: bool = True,
                 interp_method: TriggerInterpMethod = NoTriggerInterpMethod()):
        super().__init__(n_stimuli, False, include_block=True, is_full_block=is_full_block,
                         interp_method=interp_method)

    @property
    def n_data_blocks(self) -> int:
        return 0

    @property
    def n_data_stimuli(self) -> int:
        return 0


class FlashedSkipDataBlockDescriptor(FlashBlockDescriptor):

    def __init__(self,
                 n_stimuli: int,
                 n_triggers_in_block: int):
        super().__init__(n_stimuli, True, include_block=False)
        self.n_triggers_in_block = n_triggers_in_block

    @property
    def n_triggers_to_skip(self):
        return self.n_triggers_in_block

    @property
    def n_data_blocks(self) -> int:
        return 1

    @property
    def n_data_stimuli(self) -> int:
        return self.n_stimuli


class FlashedSkipRepeatBlockDescriptor(FlashBlockDescriptor):

    def __init__(self,
                 n_stimuli: int,
                 skip_n_triggers: int):
        super().__init__(n_stimuli, False, include_block=False)
        self.skip_n_triggers = skip_n_triggers

    @property
    def n_triggers_to_skip(self):
        return self.skip_n_triggers

    @property
    def n_data_blocks(self) -> int:
        return 0

    @property
    def n_data_stimuli(self) -> int:
        return 0


def _get_consecutive_frames(frame_list: List[int]) -> List[List[int]]:
    i = 0
    dest = []
    while i < len(frame_list):
        curr = frame_list[i]
        buildup = [curr, ]
        i += 1
        while (i < len(frame_list)) and \
                (frame_list[i] == (curr + 1)):
            curr = frame_list[i]
            buildup.append(curr)
            i += 1
        dest.append(buildup)

    return dest


class FlashTrialTimeStructure:
    @property
    def is_single_bin(self) -> bool:
        raise NotImplementedError


class FlashBinTimeStructure(FlashTrialTimeStructure):

    def __init__(self,
                 bin_width: Union[int, float],
                 n_before_bins: int,
                 n_after_bins: int):
        super().__init__()

        self.bin_width = bin_width
        self.n_before_bins = n_before_bins
        self.n_after_bins = n_after_bins

    @property
    def is_single_bin(self) -> bool:
        return False


class FlashIntervalTimeStructure(FlashTrialTimeStructure):

    def __init__(self,
                 interval_width: int):
        super().__init__()

        self.interval_width = interval_width

    @property
    def is_single_bin(self) -> bool:
        return True


def parse_flash_trial_triggers_and_assign_frames(
        trigger_times: np.ndarray,
        trial_structure: List[FlashBlockDescriptor],
        time_bin_info: FlashTrialTimeStructure,
        start_trigger_idx: int = 0) \
        -> Tuple[Dict[int, WithinBlockMatching], List[RepeatMatchedFrame]]:
    '''
    Function to parse flashed stimulus block structure, and assign experimentally-recorded
        trigger times to each flashed stimulus presentation

    There are two types of model timing designs that we want to support here:
        (1) Single big bin, i.e. Nora's timing structure for linear reconstruction
        (2) Many bins, for GLMs or other models that attempt to model spike timing structure

    Optionally interpolates dropped triggers if those are obvious

    :param trigger_times:
    :param trial_structure:
    :param start_trigger_idx:
    :param tolerance_interval:
    :param do_trigger_interpolation:
    :param interpolation_tolerance:
    :return:
    '''

    data_block_count, data_stimuli_count = 0, 0
    data_blocks, repeat_blocks = {}, []
    for trial_block in trial_structure:

        if not trial_block.include_block:

            skip_stimuli = trial_block.n_data_stimuli
            skip_data_blocks = trial_block.n_data_blocks

            data_stimuli_count += skip_stimuli
            data_block_count += skip_data_blocks

            start_trigger_idx += trial_block.n_triggers_to_skip

        else:
            bin_times, start_trigger_idx = trial_block.interp_method.interpolate_parse_triggers(
                trigger_times, start_trigger_idx, trial_block.n_triggers_expected
            )

            # there are two kinds of blocks
            # (1) Data blocks, which are associated with frames from the data movie
            #       which is non-repeating. In order to correctly associate recording time
            #       bins with frames, we have to advance counters that point into the data movie
            # (2) Repeat blocks, which are associated with frames from the repeats movie
            #       The entire repeat movie is used for the repeat block, and so no state
            #       needs to be kept/updated here

            # counts the number of images that this block consumes from the DATA movie
            # will be 0 if from a REPEAT block
            n_data_stimulus_images = trial_block.n_data_stimuli

            # counts the number of images from any movie;
            # should match the above for a DATA block
            n_images = trial_block.n_stimuli

            if trial_block.is_data_block:

                if isinstance(time_bin_info, FlashBinTimeStructure):
                    # we need to construct time bins here
                    block_frame = WithinBlockMatchedFrameBins(data_block_count)
                    for block_rel_stim_count, trigger_time in enumerate(bin_times):
                        stimulus_ix = data_stimuli_count + block_rel_stim_count

                        bin_edges = _make_timebin_boundaries(trigger_time,
                                                             time_bin_info.bin_width,
                                                             time_bin_info.n_before_bins,
                                                             time_bin_info.n_after_bins)

                        block_frame.append_frames(stimulus_ix, bin_edges)

                elif isinstance(time_bin_info, FlashIntervalTimeStructure):

                    block_frame = WithinBlockMatchedFrameInterval(data_block_count)
                    for block_rel_stim_count, trigger_time in enumerate(bin_times):
                        stimulus_ix = data_stimuli_count + block_rel_stim_count
                        end_interval_sample = trigger_time + time_bin_info.interval_width

                        block_frame.append_frames(stimulus_ix, (trigger_time, end_interval_sample))

                else:
                    assert False, 'time_bin_info not a known data type'

                data_blocks[data_block_count] = block_frame
                data_block_count += 1

                data_stimuli_count += n_data_stimulus_images

            else:

                if isinstance(time_bin_info, FlashBinTimeStructure):

                    block_frame = RepeatMatchedFrameTimebins()
                    for repeat_ix, trigger_time in enumerate(bin_times):
                        bin_edges = _make_timebin_boundaries(trigger_time,
                                                             time_bin_info.bin_width,
                                                             time_bin_info.n_before_bins,
                                                             time_bin_info.n_after_bins)

                        block_frame.append_frames(repeat_ix, bin_edges)

                elif isinstance(time_bin_info, FlashIntervalTimeStructure):

                    block_frame = RepeatMatchedFrameInterval()
                    for repeat_ix, trigger_time in enumerate(bin_times):
                        end_interval_sample = trigger_time + time_bin_info.interval_width
                        block_frame.append_frames(repeat_ix, (trigger_time, end_interval_sample))

                else:
                    assert False, 'time_bin_info not a known data type'

                repeat_blocks.append(block_frame)

    return data_blocks, repeat_blocks


def _construct_no_trigger_correction_experiment() -> List[Union[FlashedDataBlockDescriptor,
                                                                FlashedRepeatBlockDescriptor]]:

    return [
        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 0
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 1
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 2
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 3
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 4
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 5
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 6
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 7
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 8
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 9
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),
    ]


def get_experiment_block_structure(path_lookup: LookupKey) -> List[FlashBlockDescriptor]:
    return FLASHED_EXPERIMENT_STRUCTURE2[path_lookup]


FLASHED_EXPERIMENT_STRUCTURE2 = {

    # FIXME path was /Volumes/Lab/Users/ericwu/yass-reconstruction/2019-11-07-0/data001/data001
    ('2019-11-07-0', 'data001') : _construct_no_trigger_correction_experiment(),

    # FIXME path was /Volumes/Lab/Users/ericwu/yass-reconstruction/2019-11-07-0/data002/data002
    ('2019-11-07-0', 'data002') : [
        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 0
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 1
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 2
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        # This one has a dropped trigger
        FlashedDataBlockDescriptor(1000, interp_method=ManualTriggerInterpMethod([803, ])),  # 3
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 4
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 5
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 6
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 7
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        # This one has an interval that's too
        # long, but it looks like a late frame transition rather than a dropped trigger
        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 8
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 9
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),
    ],
    ('2018-08-07-5', 'data001') : [
        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 0
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 1
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 2
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 3
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 4
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 5
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 6
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        # this block requires interpolation
        FlashedDataBlockDescriptor(1000, interp_method=ManualTriggerInterpMethod([908, ])),  # 7
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 8
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 9
        # stimulus ended early
        FlashedSkipRepeatBlockDescriptor(150, 10),  # made up numbers since this is the end
    ],
    ('2018-08-07-5', 'data002') : _construct_no_trigger_correction_experiment(),
    ('2018-11-12-5', 'data009') : _construct_no_trigger_correction_experiment(), # need to check
    ('2017-11-29-0', 'data004') : _construct_no_trigger_correction_experiment(), # need to check
    ('2017-12-04-5', 'data006') : _construct_no_trigger_correction_experiment(),
    ('2017-12-04-5', 'data007') : [
        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 0
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 1
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 2
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        # This one has a dropped trigger
        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 3
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=ManualTriggerInterpMethod([113, ])),  # 4
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 5
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 6
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 7
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        # This one has an interval that's too
        # long, but it looks like a late frame transition rather than a dropped trigger
        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 8
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 9
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),
    ],
    ('2018-03-01-0', 'data011') : _construct_no_trigger_correction_experiment(),
    ('2018-03-01-0', 'data012') : _construct_no_trigger_correction_experiment(),

    # FIXME path was /Volumes/Lab/Users/ericwu/yass-reconstruction/2018-03-01-0/data009/data009
    ('2018-03-01-0', 'data009') : [
        # recording started 20 frames late
        # for now just skip the first block, since skipping stimulus frames
        # is actually hard with our current setup
        FlashedSkipDataBlockDescriptor(1000, 980),
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 1
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 2
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 3
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 4
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 5
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 6
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        # this block requires interpolation
        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 7
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 8
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),

        FlashedDataBlockDescriptor(1000, interp_method=NoTriggerInterpMethod()),  # 9
        FlashedRepeatBlockDescriptor(150, interp_method=NoTriggerInterpMethod()),
    ]
}
