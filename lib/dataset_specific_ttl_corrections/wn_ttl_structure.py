import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from whitenoise import RandomNoiseFrameGenerator
from lib.data_utils.sta_metadata import RGB_CONVERSION


class WhiteNoiseSynchroSection:

    def __init__(self,
                 frame_gen: RandomNoiseFrameGenerator,
                 trigger_times_interpolated: np.ndarray,
                 frames_per_trigger: int,
                 electrical_sample_rate: int):
        '''

        :param frame_gen: white noise frame generator object
        :param trigger_times_interpolated: trigger times, first
            trigger time corresponds to presentation of the first frame

            Interpolation must be done already, so the trigger intervals are
            guaranteed to be correct
        :param frames_per_trigger: Number of unique frames displayed per trigger
            (not the frame rate, i.e. different values for interval 2 and interval 1)
        :param electrical_sample_rate:
        '''

        self.frame_gen = frame_gen
        self.trigger_times = trigger_times_interpolated
        self.frames_per_trigger = frames_per_trigger
        self.electrical_sample_rate = electrical_sample_rate

    @property
    def frame_interval(self) -> int:
        return self.frame_gen.refresh_interval

    def fetch_frames(self,
                     start_frame_ix: int,
                     stop_frame_ix: int,
                     crop_slices: Optional[Tuple[slice, slice]] = None,
                     is_bw: bool = False) -> Tuple[np.ndarray, np.ndarray]:

        # resets the frame generator pointer on every call
        self.frame_gen.reset_seed_to_beginning()
        self.frame_gen.advance_seed_n_frames(start_frame_ix)

        frames_fetched = self.frame_gen.generate_block_of_frames(stop_frame_ix - start_frame_ix)
        if crop_slices is not None:
            crop_h_slice, crop_w_slice = crop_slices
            frames_fetched = frames_fetched[:, crop_h_slice, crop_w_slice, :]
        if is_bw:
            frames_fetched = (frames_fetched @ RGB_CONVERSION[None, :, :, None]).squeeze(-1)

        start_trigger_ix = start_frame_ix // self.frames_per_trigger
        start_frame_offset = start_frame_ix % self.frames_per_trigger

        end_trigger_ix = stop_frame_ix // self.frames_per_trigger
        end_frame_offset = stop_frame_ix % self.frames_per_trigger

        frame_time_acc = []  # type: List[np.ndarray]

        next_trigger_ix = start_trigger_ix
        next_frame_start_offset = start_frame_offset
        next_frame_end_offset = end_frame_offset if next_trigger_ix == end_trigger_ix else 0
        while next_trigger_ix <= end_trigger_ix:

            ttime1, ttime2 = self.trigger_times[next_trigger_ix], self.trigger_times[next_trigger_ix + 1]

            all_frame_times = np.linspace(ttime1, ttime2, endpoint=False,
                                          num=self.frames_per_trigger, dtype=np.float32)

            if next_trigger_ix == end_trigger_ix:
                frame_time_acc.append(all_frame_times[next_frame_start_offset:next_frame_end_offset + 1])
                next_trigger_ix += 1

            else:

                frame_time_acc.append(all_frame_times)
                next_trigger_ix += 1
                next_frame_start_offset = 0
                next_frame_end_offset = end_frame_offset if next_trigger_ix == end_trigger_ix else 0

        return frames_fetched, np.concatenate(frame_time_acc, axis=0)
