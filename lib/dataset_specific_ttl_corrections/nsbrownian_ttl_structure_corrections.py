import sys
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

import numpy as np
from rawmovie import RawMovieReader2

from lib.dataset_config_parser.dataset_config_parser import LookupKey
from lib.dataset_specific_ttl_corrections.ttl_interval_constants import NS_BROWNIAN_N_FRAMES_PER_TRIGGER, \
    NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE, NS_BROWNIAN_SAMPLES_PER_TRIGGER

'''
Trial structure for the INBrownian movies (jittered movies, no gray between different images)

Each image is shown (jittered around) for 60 frames at 120 Hz
We expect 100 frames to occur for every TTL trigger received

10 separate blocks, each containing
    * 500 data images (jittered static images), each shown for 60 frames
        (300 expected triggers)
    * 150 repeat images (jittered static images), each shown for 60 frames
        (90 expected triggers)
    
Because the GLM model uses the past to predict future spikes (we typically
    use around 250 ms of history, for both stimulus and spikes, corresponding
    approximately to the integration time of an RGC), each image presentation
    is NOT a decoupled trial, and hence we have to treat each block as a 
    continuous movie. We can break up each movie block into sub-blocks, but
    we have to trim the first image in each sub-block as that includes data
    from the previous block.
    
We will partition the data into four distinct partitions:

1. Training partition, approximately 90% of the total data
2. Test partition, consisting of 1/2 of one of the data image blocks
3. Heldout partition, consisting of the other half of the image block
    that was used for the test partition

======= AND ===============
4. Repeats, 150 sets of jittered images, each repeated 10 times (one repeat
    per block, as described above). These will be used solely for model
    evaluation and study purposes

'''


class NSBrownianBlockDescriptor:

    def __init__(self,
                 n_stimuli: int,
                 n_frames_per_stimulus: int,
                 is_data_block: bool,
                 include_block: bool = True,
                 is_full_block: bool = False,
                 override_block_trigger_interpolation: Optional[bool] = None,
                 override_block_raw_mode: Optional[bool] = None):
        self.n_stimuli = n_stimuli
        self.n_frames_per_stimulus = n_frames_per_stimulus
        self.is_data_block = is_data_block
        self.include_block = include_block
        self.is_full_block = is_full_block
        self.override_block_trigger_interpolation = override_block_trigger_interpolation
        self.override_block_raw_mode = override_block_raw_mode

    @property
    def n_frames_total(self):
        return self.n_stimuli * self.n_frames_per_stimulus

    @property
    def n_triggers_expected(self):
        return int(np.floor(self.n_frames_total / NS_BROWNIAN_N_FRAMES_PER_TRIGGER))

    @property
    def n_triggers_to_skip(self):
        return 0

    @property
    def n_stimulus_images_and_frames(self) -> Tuple[int, int]:
        raise NotImplementedError

    @property
    def n_data_blocks(self) -> int:
        raise NotImplementedError


class NSBrownianDataBlockDescriptor(NSBrownianBlockDescriptor):

    def __init__(self,
                 n_stimuli: int,
                 n_frames_per_stimulus: int,
                 is_full_block: bool = True,
                 override_block_trigger_interpolation: Optional[bool] = None,
                 override_block_raw_mode: Optional[bool] = None):
        super().__init__(n_stimuli, n_frames_per_stimulus, True, include_block=True, is_full_block=is_full_block,
                         override_block_trigger_interpolation=override_block_trigger_interpolation,
                         override_block_raw_mode=override_block_raw_mode)

    @property
    def n_data_blocks(self) -> int:
        return 1

    @property
    def n_stimulus_images_and_frames(self) -> Tuple[int, int]:
        return self.n_stimuli, self.n_frames_per_stimulus * self.n_stimuli


class NSBrownianRepeatBlockDescriptor(NSBrownianBlockDescriptor):

    def __init__(self,
                 n_stimuli: int,
                 n_frames_per_stimulus: int,
                 is_full_block: bool = True,
                 override_block_trigger_interpolation: Optional[bool] = None,
                 override_block_raw_mode: Optional[bool] = None):
        super().__init__(n_stimuli, n_frames_per_stimulus, False, include_block=True, is_full_block=is_full_block,
                         override_block_trigger_interpolation=override_block_trigger_interpolation,
                         override_block_raw_mode=override_block_raw_mode)

    @property
    def n_data_blocks(self) -> int:
        return 0

    @property
    def n_stimulus_images_and_frames(self) -> Tuple[int, int]:
        return 0, 0


def _merge_override_block_trigger_interpolation(global_interpolate: bool,
                                                data_block: NSBrownianBlockDescriptor) -> bool:
    if data_block.override_block_trigger_interpolation is None:
        return global_interpolate
    return data_block.override_block_trigger_interpolation


def _merge_override_raw_mode(global_raw_mode: bool,
                             data_block: NSBrownianBlockDescriptor) -> bool:
    if data_block.override_block_raw_mode is None:
        return global_raw_mode
    return data_block.override_block_raw_mode


class NSBrownianSkipDataBlockDescriptor(NSBrownianBlockDescriptor):

    def __init__(self,
                 n_stimuli: int,
                 n_frames_per_stimulus: int,
                 n_triggers_in_block: int):
        super().__init__(n_stimuli, n_frames_per_stimulus, True, include_block=False)
        self.n_triggers_in_block = n_triggers_in_block

    @property
    def n_stimulus_images_and_frames(self) -> Tuple[int, int]:
        return self.n_stimuli, self.n_frames_per_stimulus * self.n_stimuli

    @property
    def n_triggers_to_skip(self):
        return self.n_triggers_in_block

    @property
    def n_data_blocks(self) -> int:
        return 1


class NSBrownianSkipRepeatBlockDescriptor(NSBrownianBlockDescriptor):

    def __init__(self,
                 n_stimuli: int,
                 n_frames_per_stimulus: int,
                 n_triggers_in_block: int):
        super().__init__(n_stimuli, n_frames_per_stimulus, False, include_block=False)
        self.n_triggers_in_block = n_triggers_in_block

    @property
    def n_stimulus_images_and_frames(self) -> Tuple[int, int]:
        return 0, 0

    @property
    def n_triggers_to_skip(self):
        return self.n_triggers_in_block

    @property
    def n_data_blocks(self) -> int:
        return 0


class SynchronizedNSBrownianSection:
    '''
    This data structure is designed to represent

    There are a few subtleties with this class that are meant to deal with splitting
    recorded data blocks into multiple chunks (i.e. for the test and heldout splits).
    In particular:
    (1) The first TTL in self.triggers does not necessarily coincide with the
        appearance of the first frame.

        In the typical case, where SynchronizedNSBrownianBlock begins on the TTL trigger, the first
        TTL by definition must coincide with the first frame. However, in the case
        that the SynchronizedNSBrownianBlock refers to a chunk of data within a block, it is
        highly unlikely that the first frame in SynchronizedNSBrownianBlock corresponds
        to the first TTL.

        We use the following two instance variables to account for this possibility

        self.first_triggered_frame_offset: int, the number of frames that are presented
            before the first trigger occurs
        self.block_begin_sample_num: int, the sample number corresponding to the (approximate)
            occurrence of the first frame in the block being presented to the retina

    (2) Each SynchronizedNSBrownianBlock can only start on the presentation of a new
        stimulus image

    (3) Each SynchronizedNSBrownianBlock can only end at the end of the presentation of
        a stimulus image

    This class is designed to do lazy-loading of the frames, since there are way
        too many frames to fit into memory at the same time without
        cropping or some other method of reducing the total number of pixels
    '''

    def __init__(self,
                 stimulus_image_start_stop: Tuple[int, int],
                 n_frames_per_stimulus: int,
                 frame_movie_path: str,
                 trigger_times: np.ndarray,
                 frames_per_trigger: int,
                 electrical_sample_rate: int,
                 display_frame_rate: float,
                 n_stimuli_since_trial_block_start: int = 0,
                 first_triggered_frame_offset: int = 0,
                 section_begin_sample_num: Optional[int] = None,
                 last_triggered_frame_remaining: int = 0,
                 section_end_sample_num: Optional[int] = None):
        '''

        :param n_stimuli: int, number of distinct stimuli (different images)
            included in the block
        :param n_frames_per_stimulus: int, number of frames that each stimulus (image)
            is shown and jittered about for
        :param frame_movie_path: str, path to the .rawMovie file
        :param frame_start_stop: Tuple[int, int], first and last frame (indexed w.r.t.
            the beginning of the .rawMovie) included in the block, so that the relevant
            chunk of stimulus can be lazy-loaded
        :param stimulus_image_start_stop: first and last stimulus image (indexed w.r.t.
            the first image of the .rawMovie) included in the block
        :param trigger_times:
        :param n_stimuli_since_trial_block_start: number of stimulus images (not frames)
            between the start of the experimental block and the beginning of this object.
            If the beginning of this object corresponds to the beginning of the experimental
            block, this parameter should be 0.
        :param first_triggered_frame_offset: int, the number of frames that are presented
            before the first trigger occurs. Default value is 0, corresponding to the first
            frame of the movie being presented at the same time as the arrival of the first
            trigger tracked in this object, i.e. trigger_times[0]
        :param block_begin_sample_num: int, the sample number (in units of recording array
            samples since the beginning of the recording) that corresponds to the time at which
            the first frame belonging to this object is shown
        :param last_triggered_frame_remaining: int, the number of frames that occur after the
            last trigger, i.e. after trigger_times[-1]
        :param section_end_sample_num: int, the sample number (in units of recording array
            samples since the beginning of the recording) that corresponds to the time at which
            the last frame belonging ot this object is taken off the display
        '''
        self.n_frames_per_stimulus = n_frames_per_stimulus
        self.triggers = trigger_times  # type: np.ndarray
        self.frames_per_trigger = frames_per_trigger
        self.electrical_sample_rate = electrical_sample_rate
        self.display_frame_rate = display_frame_rate

        self.frame_fetch_path = frame_movie_path  # type: str
        self.stimulus_start_stop = stimulus_image_start_stop  # type: Tuple[int, int]

        self.n_stimuli_since_trial_block_start = n_stimuli_since_trial_block_start
        self.first_triggered_frame_offset = first_triggered_frame_offset
        self.section_begin_sample_num = trigger_times[
            0] if section_begin_sample_num is None else section_begin_sample_num

        self.last_triggered_frame_remaining = last_triggered_frame_remaining
        self.section_end_sample_num = trigger_times[-1] if section_end_sample_num is None else section_end_sample_num

        if self.section_begin_sample_num > trigger_times[0]:
            raise ValueError(
                f'first sample of section, {self.section_begin_sample_num}, occurred after the first trigger {self.triggers[0]}')

        if self.section_end_sample_num < trigger_times[-1]:
            raise ValueError(
                f'last sample of section {self.section_end_sample_num} occurred before the last trigger {self.triggers[-1]}')

        # finally, compute/interpolate the frame transition times for all of the frames
        # that way when we need a time for a specific frame or window of frames, we can
        # just look it up from the precomputed data structure

        # shape (n_frames + 1, )
        self.interpolated_frame_times = self._interpolate_frame_transition_times()

    def __str__(self) -> str:
        stim_low, stim_high = self.stimulus_start_stop
        return f'SynchronizedNSBrownianSection([stimulus{stim_low}:{stim_high}], [sample{self.section_begin_sample_num}:{self.section_end_sample_num}])'

    @property
    def expected_trigger_interval(self) -> float:
        return (self.electrical_sample_rate / self.display_frame_rate) * self.frames_per_trigger

    @property
    def n_stimuli(self) -> int:
        return self.stimulus_start_stop[1] - self.stimulus_start_stop[0]

    @property
    def frame_start_stop(self) -> Tuple[int, int]:
        return self.stimulus_start_stop[0] * self.n_frames_per_stimulus, self.stimulus_start_stop[
            1] * self.n_frames_per_stimulus

    def stimuli_start_stop_rel_to_exp_block_start(self) -> Tuple[int, int]:
        stim_start, stim_stop = self.stimulus_start_stop
        exp_block_start = stim_start - self.n_stimuli_since_trial_block_start
        return stim_start - exp_block_start, stim_stop - exp_block_start

    def fetch_frames_bw(self,
                        start_frame_ix: Optional[int] = None,
                        stop_frame_ix: Optional[int] = None,
                        h_low_high: Optional[Tuple[int, int]] = None,
                        w_low_high: Optional[Tuple[int, int]] = None) -> np.ndarray:
        # avoids leaking the file pointer, overhead for hitting disk
        # to read the header not that bad compared to reading the entire block
        # from disk
        with RawMovieReader2(self.frame_fetch_path,
                             chunk_n_frames=500) as rmr:
            start, stop = self.frame_start_stop
            if start_frame_ix is not None:
                start = start_frame_ix
            if start_frame_ix is not None:
                stop = stop_frame_ix

            ret_val = rmr.get_frame_sequence_bw(start, stop,
                                                h_low_high=h_low_high,
                                                w_low_high=w_low_high)[0]
            return ret_val

    def compute_start_frame_idx_for_stim(self, stim_idx: int, since_block_start: bool = True) -> int:
        '''
        Computes the frame index (defined w.r.t. the start of the .rawMovie) that the stimulus
            begins at
        :param stim_idx: int, index of the stimulus image (not frame)
        :param since_block_start: whether stim_idx counts from the beginning of the .rawMovie
            or from the beginning of this instance of SynchronizedNSBrownianSection. By default, we count
            from the beginning of this instance of SynchronizedNSBrownianSection block since that is more
            intuitive to refer to
        :return:
        '''
        # need to compute a raw frame number
        if since_block_start:
            block_frame_start = self.frame_start_stop[0]
            frame_number = block_frame_start + (stim_idx * self.n_frames_per_stimulus)
        else:
            frame_number = stim_idx * self.n_frames_per_stimulus
        return frame_number

    def compute_end_frame_idx_for_stim(self, stim_idx: int, since_block_start: bool = True) -> int:
        return self.compute_start_frame_idx_for_stim(stim_idx + 1, since_block_start=since_block_start)

    @property
    def stim_idx_of_exp_block_start(self) -> int:
        return self.stimulus_start_stop[0] - self.n_stimuli_since_trial_block_start

    @property
    def frame_idx_of_exp_block_start(self) -> int:
        return self.stim_idx_of_exp_block_start * self.n_frames_per_stimulus

    def compute_stimulus_start(self, stim_idx: int,
                               since_block_start: bool = True) \
            -> Tuple[int, int, int]:
        '''
        Computes the sample number (in units of recording array samples since the
            beginning of the recording) that the stimulus image corresponding to stim_idx
            first appears.

        Also computes the index of the next TTL trigger that occurs at or after the
            returned sample number

        Also computes the number of frames that occurs between the onset of the specific
            frame and the TTL returned above

        Together, these three numbers should be sufficient for splitting a SynchronizedNSBrownianSection
            into multiple SynchronizedNSBrownianSection

        stim_idx can be specified w.r.t. either the start of block in the trial structure
            or w.r.t. the beginning of the data movie, depending on whether the flag
            since_block_start is set
        :param stim_idx: int, index of the stimulus image (not frame) in question
                :param since_block_start: whether stim_idx counts from the beginning of the .rawMovie
            or from the beginning of this instance of SynchronizedNSBrownianSection. By default, we count
            from the beginning of this instance of SynchronizedNSBrownianSection block since that is more
            intuitive to refer to
        :return:
            int, sample number since the beginning of the recording corresponding to
                the approximate time that the stimulus image first appears

            AND

            int, index of the TTL that occurs at or immediately after the sample
                number returned above

            AND

            int, the number of frames that are shown between the returned first sample
                and the TTL corresponding to the returned index

        '''

        # need to compute a raw frame number
        if since_block_start:
            stim_number = self.stimulus_start_stop[0] + stim_idx
            frame_number = stim_number * self.n_frames_per_stimulus

        else:
            frame_number = stim_idx * self.n_frames_per_stimulus

        frames_since_first_frame = frame_number - self.frame_start_stop[0]
        if frames_since_first_frame < 0:
            raise ValueError('Specified stimulus does not overlap with this section')

        count_frames_since_first_trigger = frames_since_first_frame - self.first_triggered_frame_offset
        if count_frames_since_first_trigger < 0:

            first_trigger_time = self.triggers[0]
            time_per_frame = (first_trigger_time - self.section_begin_sample_num) / self.first_triggered_frame_offset

            sample_number = self.section_begin_sample_num + time_per_frame * frames_since_first_frame
            return sample_number, 0, int(-count_frames_since_first_trigger)

        else:
            before_ttl_idx = count_frames_since_first_trigger // NS_BROWNIAN_N_FRAMES_PER_TRIGGER
            frames_after_trigger = count_frames_since_first_trigger % NS_BROWNIAN_N_FRAMES_PER_TRIGGER

            # case (a) exact hit, i.e. the stimulus image first appears at the exact sample as the TTL
            if frames_after_trigger == 0:
                return self.triggers[before_ttl_idx], before_ttl_idx, 0
            # case (b) we have to do some time interpolation to determine when exactly the stimulus image
            # first appears
            else:
                # if we hit this cse, before_ttl_idx should be at least 0
                assert before_ttl_idx >= 0, 'something has gone terribly wrong'

                next_ttl_idx = before_ttl_idx + 1
                ttl_low_time, ttl_high_time = self.triggers[before_ttl_idx], self.triggers[next_ttl_idx]

                time_per_frame = (ttl_high_time - ttl_low_time) / NS_BROWNIAN_N_FRAMES_PER_TRIGGER
                frame_sample_num = int(np.floor(ttl_low_time + time_per_frame * frames_after_trigger))

                return frame_sample_num, next_ttl_idx, (NS_BROWNIAN_N_FRAMES_PER_TRIGGER - frames_after_trigger)

    def compute_stimulus_end(self, stim_idx: int,
                             since_block_start: bool = True) -> Tuple[int, int, int]:
        '''
        Computes the sample number (in units of recording array samples since the beginning of the recording)
            that the stimulus image corresponding stim_idx disappears (i.e. either the next stimulus
            image appears, or the movie ends)

        Also computes the index of the TTL trigger that occurs at or immediately before the returned sample number

        Also computes the number of frames that occurs after the TTL trigger occurs above and the end
            of the specified stimulus

        :param stim_idx: int, index of the stimulus image (not frame) in question
        :param since_block_start: bool, whether stim_idx refers to the start of the block
            in the trial structure or to the beginning of the entire data movie; default True
        :return:
        '''
        stimulus_number = self.stimulus_start_stop[0] + stim_idx if since_block_start else stim_idx

        if stimulus_number < self.stimulus_start_stop[1]:
            # If we have additional stimuli after the current stimulus,
            # the end of the current stimulus also corresponds to the start of the next stimulus
            # so we can reuse the start calculation using the next stimulus and then apply corrections

            # plus_one_ttl_idx is the index of the TTL that occurs at or after the onset of the next stimulus
            # We may have to correct it by subtracting in certain cases

            # frames_between is the number of frames that occur between end_sample_num and the time specified
            # by the TTL at plus_one_ttl_idx
            end_sample_num, plus_one_ttl_idx, frames_between = self.compute_stimulus_start(stim_idx + 1,  # FIXME test
                                                                                           since_block_start=since_block_start)

            # FIXME make sure we get the TTL indexing correct according to the spec in the documentation
            ttl_ix_to_return = plus_one_ttl_idx
            frames_after_last_ttl = 0
            if frames_between != 0:
                # a nonzero number of frames were shown between end_sample_num and the arrival of the first
                # TTL in the next section specified by plus_one_ttl_idx
                ttl_ix_to_return = plus_one_ttl_idx - 1
                frames_after_last_ttl = NS_BROWNIAN_N_FRAMES_PER_TRIGGER - frames_between

            return end_sample_num, ttl_ix_to_return, frames_after_last_ttl
        elif stimulus_number == self.stimulus_start_stop[1]:
            return self.triggers[-1], self.triggers.shape[0] - 1, 0
        else:
            raise ValueError(
                f'last stimulus in this object is {self.stimulus_start_stop[1]}, requested endpoint to stimulus {stimulus_number}')

    def get_snippet_transition_times(self,
                                     start_stim_idx: int,
                                     end_stim_idx: int,
                                     since_block_start: bool = True) -> np.ndarray:
        start_frame_idx = self.compute_start_frame_idx_for_stim(start_stim_idx, since_block_start=since_block_start)
        end_frame_idx = self.compute_end_frame_idx_for_stim(end_stim_idx, since_block_start=since_block_start)
        frame_transition_times = self.get_frame_transition_times(start_frame_idx,
                                                                 end_frame_idx)

        return frame_transition_times

    def get_snippet_frames(self,
                           start_stim_idx: int,
                           end_stim_idx: int,
                           since_block_start: bool = True,
                           crop_h: Optional[Tuple[int, int]] = None,
                           crop_w: Optional[Tuple[int, int]] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        '''

        :param start_stim_idx:
        :param end_stim_idx:
        :param crop_h:
        :param crop_w:
        :return:
        '''

        start_frame_idx = self.compute_start_frame_idx_for_stim(start_stim_idx, since_block_start=since_block_start)
        end_frame_idx = self.compute_end_frame_idx_for_stim(end_stim_idx, since_block_start=since_block_start)

        frames_stimulus = self.fetch_frames_bw(start_frame_idx,
                                               end_frame_idx,
                                               h_low_high=crop_h,
                                               w_low_high=crop_w)

        frame_transition_times = self.get_frame_transition_times(start_frame_idx,
                                                                 end_frame_idx)

        return frames_stimulus, frame_transition_times

    def get_snippet_sample_times(self,
                                 start_stim_idx: int,
                                 end_stim_idx: int, since_block_start: bool = True) \
            -> Tuple[int, int]:
        '''
        Gets the sample numbers corresponding to the first appearance of stimulus image start_stim_ix
            (note image, not frame) , and the sample number corresponding to the end of stimulus image
            end_stim_idx (again, note image, not frame)

        :param start_stim_idx:
        :param end_stim_idx:
        :param since_block_start:
        :return:
        '''
        start_time, _, _ = self.compute_stimulus_start(start_stim_idx, since_block_start=since_block_start)
        end_time, _, _ = self.compute_stimulus_end(end_stim_idx, since_block_start=since_block_start)
        return start_time, end_time

    def verify_valid(self,
                     tolerance_interval_mul: Tuple[float, float] = (0.92, 1.08)) -> bool:
        '''
        Verifies whether the data structure is well-formed, and that the
            we don't have problems with missing frames or TTL triggers
        :return:
        '''
        expected_samples_per_frame = self.expected_trigger_interval / NS_BROWNIAN_N_FRAMES_PER_TRIGGER

        tol_too_short, tol_too_long = tolerance_interval_mul

        too_long = expected_samples_per_frame * tol_too_long
        too_short = expected_samples_per_frame * tol_too_short

        frame_start, frame_end = self.frame_start_stop
        n_frames_total = frame_end - frame_start

        # first verify that the total number of frames is reasonable
        # total_sample_time = self.section_end_sample_num - self.section_begin_sample_num
        # measured_frame_time = total_sample_time / n_frames_total
        # if total_sample_time > too_long or measured_frame_time < too_short:
        #    raise ValueError(f'Average frame time incorrect. Expected approx {expected_samples_per_frame}, received {measured_frame_time}')

        # then verify that the section before the first trigger is valid
        # if such a section exists
        if self.first_triggered_frame_offset != 0:
            time_before_first_trigger = self.triggers[0] - self.section_begin_sample_num
            measured_frame_time = time_before_first_trigger / self.first_triggered_frame_offset

            if measured_frame_time > too_long or measured_frame_time < too_short:
                raise ValueError(
                    f'Average frame time before first trigger incorrect. Expected approx {expected_samples_per_frame}, received {measured_frame_time}')

        # make sure we have an appropriate number of frames for the trigger section
        n_frames_trigger_section = n_frames_total - self.first_triggered_frame_offset - self.last_triggered_frame_remaining
        expected_frames_trigger_section = (self.triggers.shape[0] - 1) * NS_BROWNIAN_N_FRAMES_PER_TRIGGER
        if n_frames_trigger_section != expected_frames_trigger_section:
            raise ValueError(
                f'Triggered section has incorrect number of frames; had {n_frames_trigger_section}, expected {expected_frames_trigger_section}')

        delta_trigger = self.triggers[1:] - self.triggers[:-1]
        if np.any(delta_trigger > (tol_too_long * self.expected_trigger_interval)) or \
                np.any(delta_trigger < (tol_too_short * self.expected_trigger_interval)):
            print(delta_trigger, self.expected_trigger_interval)
            raise ValueError('Triggered section trigger interval incorrect')

        # finally, make sure that the section after the last trigger is valid
        # if such a section exists
        if self.last_triggered_frame_remaining != 0:
            time_after_last_trigger = self.section_end_sample_num - self.triggers[-1]
            measured_frame_time = time_after_last_trigger / self.last_triggered_frame_remaining

            if measured_frame_time > too_long or measured_frame_time < too_short:
                raise ValueError(
                    f'Average frame time after last trigger incorrect. Expected approx {expected_samples_per_frame}, received {measured_frame_time}')

        return True

    def _interpolate_frame_transition_times(self) -> np.ndarray:
        '''
        Computes estimated frame transition times
        :param start_frame_ix: optional start frame number, inclusive,
            refers to the first frame of the block if not specified
        :param end_frame_ix: optional end frame number, exclusive,
            refers to the last+1 frame of the block if not specified
        :return:
        '''
        frame_start, frame_stop = self.frame_start_stop
        frames_per_ttl = self.frames_per_trigger

        n_frames_shown = frame_stop - frame_start
        frame_interpolation_time_buffer = []

        # first deal with the section of time before the arrival of the first trigger, if such
        # a section of time exists
        n_frames_early = self.first_triggered_frame_offset
        if n_frames_early != 0:
            first_sample = self.section_begin_sample_num
            last_sample = self.triggers[0]

            section_frame_transition_times = np.linspace(first_sample, last_sample, endpoint=False, num=n_frames_early,
                                                         dtype=np.float32)
            frame_interpolation_time_buffer.append(section_frame_transition_times)

        # then deal with the regular trigger section
        ttl_times = self.triggers
        n_ttl_triggers = ttl_times.shape[0]
        for i in range(0, n_ttl_triggers - 1):
            ttl_start, ttl_end = ttl_times[i], ttl_times[i + 1]
            interval_transition_times = np.linspace(ttl_start, ttl_end,
                                                    num=frames_per_ttl, endpoint=False)
            frame_interpolation_time_buffer.append(interval_transition_times)

        # finally deal with the section of time after the arrival of the last trigger, if such
        # a section of time exists
        n_frames_late = self.last_triggered_frame_remaining
        if n_frames_late != 0:
            first_sample = ttl_times[-1]
            last_sample = self.section_end_sample_num

            section_frame_transition_times = np.linspace(first_sample, last_sample, endpoint=True,
                                                         num=n_frames_late + 1, dtype=np.float32)
            frame_interpolation_time_buffer.append(section_frame_transition_times)
        else:
            frame_interpolation_time_buffer.append(np.array([ttl_times[-1], ], dtype=np.float32))

        frame_transition_times_all = np.concatenate(frame_interpolation_time_buffer)

        assert frame_transition_times_all.shape[0] == n_frames_shown + 1, 'something terrible happened'

        return frame_transition_times_all

    def get_frame_transition_times(self, frame_start_ix: int, frame_end_ix: int) -> np.ndarray:

        block_first_frame, block_last_frame = self.frame_start_stop
        frame_rel_start = frame_start_ix - block_first_frame
        frame_rel_end = frame_end_ix - block_first_frame

        relevant_times = self.interpolated_frame_times[frame_rel_start:frame_rel_end + 1]
        assert relevant_times.shape[0] == (frame_end_ix - frame_start_ix + 1)
        return relevant_times


def ns_brownian_parse_triggers(trigger_times: np.ndarray,
                               start_trigger_idx: int,
                               n_expected_triggers: int,
                               expected_trigger_interval: float,
                               raw_mode: bool = False,
                               tolerance_interval: Tuple[float, float] = (0.92, 1.08),
                               do_trigger_interpolation: bool = True,
                               interpolation_tolerance: float = 0.08) \
        -> Tuple[np.ndarray, int]:
    '''

    We return the start time and the end time, so that linspace can be used easily
        without doing extra arithmetic

    Note that sometimes the last trigger in a stimulus block is also dropped; this isn't
        "incorrect" but needs to be dealt with correctly

    Assumptions:
        1. There are only dropped triggers, never extra triggers

    :param trigger_times: shape (n_trigger_times, ), the recorded triggers for
        the dataset
    :param start_trigger_idx: int, the index of the first trigger to be processed
        (so that this function can be used anywhere)
    :param n_expected_triggers: int, the number of triggers that are expected
        in the current block, if no triggers are dropped
    :param expected_trigger_interval: float
    :param raw_mode: bool, True if user manually specifies that the triggers in the
        dataset are correct (i.e. the timing is screwed up, but the user knows a
        a priori that the triggers are correct). Default False, since we want to
        warn the caller if the triggers are potentially screwed up.
    :param do_trigger_interpolation: bool, whether or not to interpolate the trigger
        times. Will print a warning to stderr if trigger interpolation is required;
        Will raise an Exception if trigger interpolation is required but cannot
        be figured out algorithmically
    :return:
    '''
    tol_low, tol_high = tolerance_interval

    relevant_trigger_chunk = trigger_times[start_trigger_idx:start_trigger_idx + n_expected_triggers]
    delta_triggers = relevant_trigger_chunk[1:] - relevant_trigger_chunk[:-1]

    does_not_need_interpolation = np.logical_and.reduce([
        delta_triggers > (tol_low * expected_trigger_interval),
        delta_triggers < (tol_high * expected_trigger_interval)
    ])

    if raw_mode or np.all(does_not_need_interpolation):
        return relevant_trigger_chunk, start_trigger_idx + n_expected_triggers

    # there's a subtlety here since we are interpreting trigger blocks rather than
    # over the whole recording: if triggers are indeed dropped, then relevant_trigger_chunk
    # includes triggers that don't belong to the current block, and we have to
    # figure out how to ignore those

    # rather than pre-determine whether or not and how to do the trigger interpolation,
    # we simply loop through and count triggers; if at any point we can't figure out
    # how to do the interpolation, spit out a warning and crash
    if not do_trigger_interpolation:
        raise ValueError("Trigger interpolation required, but user specified no trigger interpolation")

    # Announce to stderr that we're doing trigger interpolation
    print("Performing trigger interpolation", file=sys.stderr)
    prev_trigger_time = relevant_trigger_chunk[0]
    interpolated_trigger_times = [prev_trigger_time, ]
    interpolated_count, ix = 1, 1
    while interpolated_count < n_expected_triggers:

        trigger_time = relevant_trigger_chunk[ix]

        # figure out how many trigger intervals this corresponds to
        delta = trigger_time - prev_trigger_time
        num_intervals = delta / expected_trigger_interval

        is_close_to_one = (tol_low < num_intervals < tol_high)
        if is_close_to_one:
            interpolated_trigger_times.append(trigger_time)
            prev_trigger_time = trigger_time
            ix += 1
            interpolated_count += 1

        else:
            integer_multiple_delta = np.abs(np.rint(num_intervals) - num_intervals)
            if integer_multiple_delta < interpolation_tolerance:
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


def parse_structured_ns_brownian_triggers_and_assign_frames(
        trigger_times: np.ndarray,
        trial_structure: List[NSBrownianBlockDescriptor],
        frames_per_trigger: int,
        frame_rate: Union[float, int],
        electrical_sample_rate: Union[float, int],
        data_movie_path: str,
        repeat_movie_path: str,
        start_trigger_idx: int = 0,
        raw_mode: bool = False,
        tolerance_interval: Tuple[float, float] = (0.92, 1.08),
        do_trigger_interpolation: bool = True,
        interpolation_tolerance: float = 0.08) \
        -> Tuple[Dict[int, SynchronizedNSBrownianSection], List[SynchronizedNSBrownianSection]]:
    '''
    Function to parse the stimulus block structure, and to assign experimentally recorded
        trigger times and stimulus frames to each block.

    :param trigger_times: vector of integer trigger times, shape (n_ttls, ), in units
        of recording array samples
    :param trial_structure: describes the structure of the experimental design, including
        which blocks to skip
    :param expected_trigger_interval: float, expected amount of time between received
        triggers, in units of recording array samples
    :param data_movie_path: path to data .rawMovie file
    :param repeat_movie_path: path to repeats .rawMovie file
    :param start_trigger_idx:
    :param tolerance_interval: scalar multiple of expected_trigger_interval in which a
        received trigger must arrive, so that we're sure that we don't unknowingly lose
        synchronization with the stimulus
    :param do_trigger_interpolation: bool, whether to interpolate missing triggers where
        possible, or to give up and throw an error if triggers are missing
    :param interpolation_tolerance: scalar tolerance within which an interpolated trigger
        interval must be if the interpolation is done; if this cannot be met then throw
        an error
    :return:
    '''

    expected_trigger_interval = (electrical_sample_rate / frame_rate) * frames_per_trigger

    data_block_descriptor_ret = {}  # type: Dict[int, SynchronizedNSBrownianSection]
    repeat_block_descriptor_ret = []  # type: List[SynchronizedNSBrownianSection]
    data_block_count, data_image_count = 0, 0
    for trial_block in trial_structure:

        # if skip triggers
        if not trial_block.include_block:

            skip_images, skip_frames = trial_block.n_stimulus_images_and_frames
            advance_data_blocks = trial_block.n_data_blocks

            data_block_count += advance_data_blocks
            data_image_count += skip_images

            start_trigger_idx = start_trigger_idx + trial_block.n_triggers_to_skip

        else:

            block_do_trigger_interpolate = _merge_override_block_trigger_interpolation(
                do_trigger_interpolation,
                trial_block
            )

            block_do_raw_mode = _merge_override_raw_mode(
                raw_mode,
                trial_block
            )

            bin_times, start_trigger_idx = ns_brownian_parse_triggers(
                trigger_times,
                start_trigger_idx,
                trial_block.n_triggers_expected,
                expected_trigger_interval,
                raw_mode=block_do_raw_mode,
                tolerance_interval=tolerance_interval,
                do_trigger_interpolation=block_do_trigger_interpolate,
                interpolation_tolerance=interpolation_tolerance)

            # there are two kinds of blocks
            # (1) Data blocks, which are associated with frames from the data movie
            #       which is non-repeating. In order to correctly associate recording time
            #       bins with frames, we have to advance counters that point into the data movie
            # (2) Repeat blocks, which are associated with frames from the repeats movie
            #       The entire repeat movie is used for the repeat block, and so no state
            #       needst to be kept/updated here

            # counts the number of images and frames that this block consumes from the DATA movie
            # will both be 0 if from a REPEAT block
            n_stimulus_images, n_stimulus_frames = trial_block.n_stimulus_images_and_frames

            # counts the number of images and frames from any movie;
            # should match the above for a DATA block
            n_images, n_frames = trial_block.n_stimuli, trial_block.n_frames_total

            # if the number of frames is a multiple of the number of frames received per trigger
            # then we have to "hallucinate" the last trigger time otherwise we have no timing
            # information for the last group of frames
            n_frames_remainder = n_frames % frames_per_trigger
            end_sample_after_last_trigger = None
            median_bin_time = np.median(bin_times[1:] - bin_times[:-1])
            if n_frames_remainder == 0:
                hallucinated_bin_time = int(np.round(median_bin_time + bin_times[-1]))
                bin_times = np.concatenate([bin_times, np.array([hallucinated_bin_time, ])], axis=0)

            # otherwise, we have to count the number of end frames, and guess what sample
            # the last frame corresponds to
            else:
                last_chunk_size = median_bin_time * n_frames_remainder / frames_per_trigger
                end_sample_after_last_trigger = int(np.round(last_chunk_size + bin_times[-1]))

            if trial_block.is_data_block:
                image_start, image_stop = data_image_count, data_image_count + n_images
                synchro_block = SynchronizedNSBrownianSection((image_start, image_stop),
                                                              trial_block.n_frames_per_stimulus,
                                                              data_movie_path,
                                                              bin_times,
                                                              frames_per_trigger,
                                                              electrical_sample_rate,
                                                              frame_rate,
                                                              last_triggered_frame_remaining=n_frames_remainder,
                                                              section_end_sample_num=end_sample_after_last_trigger)

                data_block_descriptor_ret[data_block_count] = synchro_block

            else:
                image_start, image_stop = 0, n_images
                synchro_block = SynchronizedNSBrownianSection((image_start, image_stop),
                                                              trial_block.n_frames_per_stimulus,
                                                              repeat_movie_path,
                                                              bin_times,
                                                              frames_per_trigger,
                                                              electrical_sample_rate,
                                                              frame_rate,
                                                              last_triggered_frame_remaining=n_frames_remainder,
                                                              section_end_sample_num=end_sample_after_last_trigger)
                repeat_block_descriptor_ret.append(synchro_block)

            # update the state variables
            data_image_count += n_stimulus_images
            data_block_count += trial_block.n_data_blocks

    return data_block_descriptor_ret, repeat_block_descriptor_ret


@dataclass
class BrownianExperimentStructureParams:
    experiment_structure: List[NSBrownianBlockDescriptor]
    raw_mode: bool = False
    do_trigger_interpolation: bool = True
    tolerance_interval: Tuple[float, float] = (0.92, 1.08)
    interpolation_tolerance: float = 0.08


NS_BROWNIAN_EXPERIMENT_STRUCTURE = {
    ('2018-08-07-5', 'data010'):
        BrownianExperimentStructureParams(
            [
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 1
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 2
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 3
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 4
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 5
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 6
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 7
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 8
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 9
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 10
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE)
            ],
            do_trigger_interpolation=True, raw_mode=False),

    # this one has screwed up trigger times, but I think there were not any dropped triggers
    # so raw_mode=True it is
    ('2018-08-07-5', 'data009'):
        BrownianExperimentStructureParams(
            [
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 1
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 2
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 3
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 4
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 5
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 6
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 7
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 8
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 9
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 10
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
            ],
            do_trigger_interpolation=True, raw_mode=False),

    # this one has screwed up trigger times, but I think there were not any dropped triggers
    # so raw_mode=True it is
    ('2019-11-07-0', 'data005'):
        BrownianExperimentStructureParams(
            [
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 1
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 2
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 3
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 4
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 5
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 6
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 7
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 8
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 9
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 10
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
            ],
            do_trigger_interpolation=False, raw_mode=True),
    ('2019-11-07-0', 'data006'):
        BrownianExperimentStructureParams(
            [
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 1
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 2
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 3
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 4
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 5
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 6
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 7
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 8
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 9
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 10
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
            ],
            do_trigger_interpolation=False, raw_mode=True),

    ('2018-11-12-5', 'data004'):
        BrownianExperimentStructureParams(
            [
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 1
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 2
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 3
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 4
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 5
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 6
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 7
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 8
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 9
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE,
                                              override_block_trigger_interpolation=True, override_block_raw_mode=False),  # 10
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
            ],
            do_trigger_interpolation=False, raw_mode=True),

    # I think there were not any dropped triggers
    # so raw_mode=True it is
    ('2018-11-12-5', 'data005'):
        BrownianExperimentStructureParams(
            [
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 1
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 2
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 3
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 4
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 5
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 6
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 7
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 8
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 9
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
                NSBrownianDataBlockDescriptor(NS_BROWNIAN_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),  # 10
                NSBrownianRepeatBlockDescriptor(NS_BROWNIAN_TEST_BLOCK_SIZE, NS_BROWNIAN_N_FRAMES_PER_IMAGE),
            ],
            do_trigger_interpolation=False, raw_mode=True),

}


def dispatch_ns_brownian_experimental_structure_and_params(lookup_key: LookupKey):
    return NS_BROWNIAN_EXPERIMENT_STRUCTURE[lookup_key]
