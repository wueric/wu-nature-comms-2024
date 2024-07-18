import numpy as np
from typing import List, Callable, Tuple, Union, Dict

NSCENES_STIXEL_SIZE = 2


def make_upsample_wn_to_nscenes_noflip(upsample_factor: int) -> Callable:
    '''
    Makes a callable function that upsamples a filter matrix from STA
        coordinates into nscenes stimulus coordinates

    :param upsample_factor: factor that we need to upsample the white noise receptive fields by
        to match the raw stimulus frames from the natural scenes flashes
    :return:
    '''

    def upsample_fn(wn_filter_matrices: np.ndarray,
                    nscenes_crop_width_low: int = 0,
                    nscenes_crop_width_high: int = 0,
                    nscenes_crop_height_low: int = 0,
                    nscenes_crop_height_high: int = 0,
                    downsample_nscenes_factor: int = 1) -> np.ndarray:
        '''
        To get from STA matrix dimensions to the nscenes stimulus image dimension
            we need to upsample

        No flipping is required to get the STA orientation to match up with the nscenes image

        :param wn_filter_matrices: shape (n_cells, sta_height, sta_width)
        :param nscenes_crop_width_low: number of nscenes width pixels we crop from the low side
        :param nscenes_crop_width_high: number of nscenes width pixels we crop from the high side
        :param nscenes_crop_height_low: number of nscenes height pixels we crop from the low side
        :param nscenes_crop_height_high: number of nscense height pixels we crop from the high side
        :param downsample_nscenes_factor : factor that we want to downsample the nscenes stimulus by
        :return:
        '''

        n_cells, orig_height, orig_width = wn_filter_matrices.shape
        corrected_upsample_factor = upsample_factor // downsample_nscenes_factor

        wn_crop_width_low = nscenes_crop_width_low // upsample_factor
        wn_crop_width_high = nscenes_crop_width_high // upsample_factor
        wn_crop_height_low = nscenes_crop_height_low // upsample_factor
        wn_crop_height_high = nscenes_crop_height_high // upsample_factor

        frame_width = orig_width - (wn_crop_width_low + wn_crop_width_high)
        frame_height = orig_height - (wn_crop_height_high + wn_crop_height_low)

        low_width, high_width = wn_crop_width_low, frame_width + wn_crop_width_low
        low_height, high_height = wn_crop_height_low, wn_crop_height_low + frame_height

        cropped_wn_filter_matrices = wn_filter_matrices[:, low_height:high_height, low_width:high_width]

        return cropped_wn_filter_matrices.repeat(corrected_upsample_factor,
                                                 axis=1).repeat(corrected_upsample_factor, axis=2)

    return upsample_fn


WN_STIXEL_SIZE_DISPATCH_DICT = {
    ('2018-08-07-5', 'data000'): 8,
    ('2018-08-07-5', 'data011'): 8,

    ('2017-12-04-5', 'data005'): 10,

    ('2017-11-29-0', 'data001'): 16,

    ('2018-03-01-0', 'data010'): 16,

    ('2019-11-07-0', 'data000'): 8,
    ('2019-11-07-0', 'data003'): 8,

    ('2018-11-12-5', 'data002'): 16,

    ('2018-11-12-5', 'data008'): 8,
}

dispatch_dict = {

    ('2018-08-07-5', 'data000'): make_upsample_wn_to_nscenes_noflip(8 // 2),
    ('2018-08-07-5', 'data011'): make_upsample_wn_to_nscenes_noflip(8 // 2),

    ('2017-12-04-5', 'data005'): make_upsample_wn_to_nscenes_noflip(10 // 2),

    ('2017-11-29-0', 'data001'): make_upsample_wn_to_nscenes_noflip(16 // 2),

    ('2018-03-01-0', 'data010'): make_upsample_wn_to_nscenes_noflip(16 // 2),

    ('2019-11-07-0', 'data000'): make_upsample_wn_to_nscenes_noflip(8 // 2),
    ('2019-11-07-0', 'data003'): make_upsample_wn_to_nscenes_noflip(8 // 2),

    ('2018-11-12-5', 'data002'): make_upsample_wn_to_nscenes_noflip(16 // 2),
    ('2018-11-12-5', 'data008'): make_upsample_wn_to_nscenes_noflip(8 // 2),
}


def dispatch_stimulus_scale(ref_lookup_key: Tuple[str, str]) -> Callable:
    return dispatch_dict[ref_lookup_key]


def get_stixel_size_wn(ref_lookup_key: Tuple[str, str]) -> int:
    return WN_STIXEL_SIZE_DISPATCH_DICT[ref_lookup_key]
