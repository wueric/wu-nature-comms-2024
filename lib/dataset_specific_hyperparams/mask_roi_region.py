import numpy as np

from lib.data_utils.sta_metadata import compute_sig_stixel_mask

from dataclasses import dataclass

from collections import namedtuple

from lib.dataset_config_parser.dataset_config_parser import LookupKey


@dataclass
class MaskHyperparameters:
    threshold: float
    n_coverage: int
    use_cc_alg: bool = False


ValidMaskKey = namedtuple('ValidMaskKey', ['path', 'downsample_factor'])

MASK_GENERATION_LUT = {
    ('2018-08-07-5', 'data000'): {
        1: MaskHyperparameters(2.5e-2, 3),
    },
    ('2018-08-07-5', 'data011'): {
        1: MaskHyperparameters(2.5e-2, 3),
    },
    ('2017-11-29-0', 'data001'): {
        1: MaskHyperparameters(2e-2, 3),
    },
    ('2018-03-01-0', 'data010'): {
        1: MaskHyperparameters(2e-2, 3),
    },
    ('2019-11-07-0', 'data003'): {
        1: MaskHyperparameters(1e-2, 3, use_cc_alg=True),
    },
    ('2017-12-04-5', 'data005'): {
        1: MaskHyperparameters(2e-2, 2, use_cc_alg=True),
    },
    ('2018-11-12-5', 'data002'): {
        1: MaskHyperparameters(1e-2, 3, use_cc_alg=True),
    },
    ('2018-11-12-5', 'data008'): {
        1: MaskHyperparameters(2.5e-2, 2, use_cc_alg=True),
    }
}


def make_sig_stixel_loss_mask(piece_lookup_key: LookupKey,
                              blurred_stas_by_type,
                              crop_hlow: int = 0,
                              crop_hhigh: int = 0,
                              crop_wlow: int = 0,
                              crop_whigh: int = 0,
                              downsample_factor: int = 1) -> np.ndarray:
    lut_lookup = MASK_GENERATION_LUT[piece_lookup_key][downsample_factor]

    # compute the sig stixel mask
    sig_stixel_loss_mask = compute_sig_stixel_mask(blurred_stas_by_type,
                                                   crop_hlow=crop_hlow,
                                                   crop_hhigh=crop_hhigh,
                                                   crop_wlow=crop_wlow,
                                                   crop_whigh=crop_whigh,
                                                   downsample_factor=downsample_factor,
                                                   threshold=lut_lookup.threshold,
                                                   min_coverage=lut_lookup.n_coverage,
                                                   use_cc_alg=lut_lookup.use_cc_alg)
    return sig_stixel_loss_mask
