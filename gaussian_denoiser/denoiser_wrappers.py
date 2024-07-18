import torch
import torch.nn as nn

from typing import Callable, Tuple, Union

from gaussian_denoiser.dpir_models.models.network_unet import UNetRes

import numpy as np

def load_zhang_drunet_unblind_denoiser(device: torch.device) -> nn.Module:
    '''
    Loads the grayscale unblind denoiser model from
        https://github.com/cszn/DPIR
    :return:
    '''
    N_CHANNELS = 1
    MODEL_PATH = 'gaussian_denoiser/unblind_denoisers/drunet_gray.pth'

    model = UNetRes(in_nc=N_CHANNELS + 1, out_nc=N_CHANNELS, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                    downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False

    return model.to(device)


def load_masked_drunet_unblind_denoiser(device: torch.device,
                                        requires_grad: bool = False) -> nn.Module:
    '''
    Loads the grayscale unblind denoiser with an additional channel for the
        valid region mask. Underlying model is the DRUnet model from
        Zhang et al, but with an extra channel that I (wueric) added
        to mask regions of the image that are outside of the region
        of the recorded retina

    :param device:
    :return:
    '''
    N_CHANNELS = 1  # number of image channels, 1 for grayscale
    MODEL_PATH = 'gaussian_denoiser/masked_unblind_denoisers/drunet_mask_l1_epoch7.pth'

    model = UNetRes(in_nc=N_CHANNELS + 2, out_nc=N_CHANNELS, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                    downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['denoiser'], strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = requires_grad
    return model.to(device)


def make_retina_bfcnn_rescaler_fns(retina_range: Tuple[float, float],
                                   bfcnn_range: Tuple[float, float]) \
        -> Tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor]
        ]:
    ret_low, ret_high = retina_range
    ret_scale = ret_high - ret_low

    bfcnn_low, bfcnn_high = bfcnn_range
    bfcnn_scale = bfcnn_high - bfcnn_low

    ret_to_bfcnn_scale = bfcnn_scale / ret_scale
    bfcnn_to_ret_scale = ret_scale / bfcnn_scale

    def retina_to_bfcnn_fn(retina_image: torch.Tensor) -> torch.Tensor:
        return (retina_image - ret_low) * ret_to_bfcnn_scale + bfcnn_low

    def bfcnn_to_retina_fn(bfcnn_image: torch.Tensor) -> torch.Tensor:
        return (bfcnn_image - bfcnn_low) * bfcnn_to_ret_scale + ret_low

    def bfcnn_resid_to_retina_fn(bfcnn_residual: torch.Tensor) -> torch.Tensor:
        return bfcnn_residual * bfcnn_to_ret_scale

    return retina_to_bfcnn_fn, bfcnn_to_retina_fn, bfcnn_resid_to_retina_fn


def make_retina_bfcnn_noise_rescaler_fns(retina_range: Tuple[float, float],
                                         bfcnn_range: Tuple[float, float]) \
        -> Tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor]
        ]:
    ret_low, ret_high = retina_range
    ret_scale = ret_high - ret_low
    ret_scale2 = ret_scale * ret_scale

    bfcnn_low, bfcnn_high = bfcnn_range
    bfcnn_scale = bfcnn_high - bfcnn_low
    bfcnn_scale2 = bfcnn_scale * bfcnn_scale

    ret_to_bfcnn_scale = bfcnn_scale / ret_scale
    bfcnn_to_ret_scale = ret_scale / bfcnn_scale

    def retina_to_bfcnn_fn(retina_image: torch.Tensor) -> torch.Tensor:
        return (retina_image - ret_low) * ret_to_bfcnn_scale + bfcnn_low

    def bfcnn_to_retina_fn(bfcnn_image: torch.Tensor) -> torch.Tensor:
        return (bfcnn_image - bfcnn_low) * bfcnn_to_ret_scale + ret_low

    def retina_noise_var_to_bfcnn_noise_fn(retina_noise_variance: torch.Tensor) -> torch.Tensor:
        return (bfcnn_scale2 * retina_noise_variance) / ret_scale2

    return retina_to_bfcnn_fn, bfcnn_to_retina_fn, retina_noise_var_to_bfcnn_noise_fn


def make_unblind_apply_zhang_dpir_denoiser(denoiser: nn.Module,
                                           retina_range: Tuple[float, float],
                                           bfcnn_range: Tuple[float, float]) \
        -> Callable[[torch.Tensor, Union[float, torch.Tensor]], torch.Tensor]:
    '''
    Denoises an image by applying the Zhang et al DPIR BFCNN unblind denoiser

    IMPORTANT: THIS COMPUTES A DENOISED IMAGE, NOT A RESIDUAL

    :param denoiser: blind denoiser from Zhang et al
    :param retina_range: (low, high) corresponding to the range of pixel values
        for the target image
    :param bfcnn_range: (low, high) corresponding to the range of pixel values
        that the NN was trained with
    :return:
    '''
    retina_to_bfcnn_fn, bfcnn_to_retina_fn, retina_noise_var_to_bfcnn_noise_fn = make_retina_bfcnn_noise_rescaler_fns(
        retina_range,
        bfcnn_range)

    def unblind_apply(batched_images: torch.Tensor,
                      batched_sig2_levels: Union[float, torch.Tensor]) \
            -> torch.Tensor:
        '''

        :param batched_images: shape (batch, height, width)
        :param batched_sig2_levels: either a number of shape (batch, )
        :return: (batch, height, width)
        '''
        batch, height, width = batched_images.shape
        with torch.no_grad():

            batched_images_nn_range = retina_to_bfcnn_fn(batched_images)
            if isinstance(batched_sig2_levels, torch.Tensor):
                rescaled_noise_sig2 = retina_noise_var_to_bfcnn_noise_fn(batched_sig2_levels)

                nn_input = torch.cat((batched_images_nn_range[:, None, :, :],
                                      torch.sqrt(rescaled_noise_sig2).repeat(1, 1, height, width)),
                                     dim=1)
            else:
                rescaled_noise_sig2 = np.sqrt(retina_noise_var_to_bfcnn_noise_fn(batched_sig2_levels)) \
                                      * torch.ones_like(batched_images_nn_range[:, None, :, :])
                nn_input = torch.cat((batched_images_nn_range[:, None, :, :],
                                      rescaled_noise_sig2),
                                     dim=1)

            denoised_output = denoiser(nn_input)

            return bfcnn_to_retina_fn(denoised_output).squeeze(1)

    return unblind_apply


def make_unblind_apply_dpir_denoiser_with_mask(masked_denoiser: nn.Module,
                                               retina_range: Tuple[float, float],
                                               bfcnn_range: Tuple[float, float]) \
        -> Callable[[torch.Tensor, Union[float, torch.Tensor], torch.Tensor], torch.Tensor]:
    '''
    Denoises an image by applying the modified DPIR BF UNet (DRUNet) denoiser
        with an additional valid mask for the target image
    :param masked_denoiser: 3 channel input DRUnet denoiser (channels are image, noise std,
        valid mask) already on compute device
    :param retina_range: (low, high), corresponding to the range of pixel values for the
        target reconstructed image
    :param bfcnn_range: (low, high) corresponding to the range of pixel values that
        the NN was trained with

    :return:
    '''

    retina_to_bfcnn_fn, bfcnn_to_retina_fn, retina_noise_var_to_bfcnn_noise_fn = make_retina_bfcnn_noise_rescaler_fns(
        retina_range,
        bfcnn_range)

    bfcnn_diff = bfcnn_range[1] - bfcnn_range[0]

    def unblind_apply_with_mask(batched_images: torch.Tensor,
                                batched_sig2_levels: Union[float, torch.Tensor],
                                fixed_mask: torch.Tensor) \
            -> torch.Tensor:
        '''

        :param batched_images: shape (batch, height, width)
        :param batched_sig2_levels: either a number or shape (batch, )
        :param fixed_mask: shape (height, width), valid mask for the target image, where
            1 corresponds to the valid region, and 0 is the invalid region. Must be a torch.Tensor
            of type torch.float32 already placed on the correct device


            Note to self: the valid mask must be rescaled to (bfcnn_low, bfcnn_high) since that
            was what the network was trained with. This can be done once ahead of time
            and kept in scope
        :return: (batch, height, width)
        '''
        batch, height, width = batched_images.shape

        with torch.no_grad():
            rescaled_mask = bfcnn_diff * fixed_mask + bfcnn_range[0]
            batched_images_nn_range = retina_to_bfcnn_fn(batched_images)
            mask_repeated = rescaled_mask[None, :, :].expand(batch, -1, -1)
            if isinstance(batched_sig2_levels, torch.Tensor):
                rescaled_noise_sig2 = retina_noise_var_to_bfcnn_noise_fn(batched_sig2_levels)
                # shape (batch, 3, height, width)
                nn_input = torch.cat([batched_images_nn_range[:, None, :, :],
                                      torch.sqrt(rescaled_noise_sig2).repeat(1, 1, height, width),
                                      mask_repeated[:, None, :, :]],
                                     dim=1)
            else:
                rescaled_noise_sig2 = np.sqrt(retina_noise_var_to_bfcnn_noise_fn(batched_sig2_levels)) \
                                      * torch.ones_like(batched_images_nn_range[:, None, :, :])
                nn_input = torch.cat([batched_images_nn_range[:, None, :, :],
                                      rescaled_noise_sig2,
                                      mask_repeated[:, None, :, :]],
                                     dim=1)

            denoised_output = masked_denoiser(nn_input)

            return bfcnn_to_retina_fn(denoised_output).squeeze(1)

    return unblind_apply_with_mask

def make_unblind_apply_dpir_denoiser_with_mask_with_grad(masked_denoiser: nn.Module,
                                               retina_range: Tuple[float, float],
                                               bfcnn_range: Tuple[float, float]) \
        -> Callable[[torch.Tensor, Union[float, torch.Tensor], torch.Tensor], torch.Tensor]:
    '''
    Denoises an image by applying the modified DPIR BF UNet (DRUNet) denoiser
        with an additional valid mask for the target image
    :param masked_denoiser: 3 channel input DRUnet denoiser (channels are image, noise std,
        valid mask) already on compute device
    :param retina_range: (low, high), corresponding to the range of pixel values for the
        target reconstructed image
    :param bfcnn_range: (low, high) corresponding to the range of pixel values that
        the NN was trained with

    :return:
    '''

    retina_to_bfcnn_fn, bfcnn_to_retina_fn, retina_noise_var_to_bfcnn_noise_fn = make_retina_bfcnn_noise_rescaler_fns(
        retina_range,
        bfcnn_range)

    bfcnn_diff = bfcnn_range[1] - bfcnn_range[0]

    def unblind_apply_with_mask(batched_images: torch.Tensor,
                                batched_sig2_levels: Union[float, torch.Tensor],
                                fixed_mask: torch.Tensor) \
            -> torch.Tensor:
        '''

        :param batched_images: shape (batch, height, width)
        :param batched_sig2_levels: either a number or shape (batch, )
        :param fixed_mask: shape (height, width), valid mask for the target image, where
            1 corresponds to the valid region, and 0 is the invalid region. Must be a torch.Tensor
            of type torch.float32 already placed on the correct device


            Note to self: the valid mask must be rescaled to (bfcnn_low, bfcnn_high) since that
            was what the network was trained with. This can be done once ahead of time
            and kept in scope
        :return: (batch, height, width)
        '''
        batch, height, width = batched_images.shape

        rescaled_mask = bfcnn_diff * fixed_mask + bfcnn_range[0]
        batched_images_nn_range = retina_to_bfcnn_fn(batched_images)
        mask_repeated = rescaled_mask[None, :, :].expand(batch, -1, -1)
        if isinstance(batched_sig2_levels, torch.Tensor):
            rescaled_noise_sig2 = retina_noise_var_to_bfcnn_noise_fn(batched_sig2_levels)
            # shape (batch, 3, height, width)
            nn_input = torch.cat([batched_images_nn_range[:, None, :, :],
                                  torch.sqrt(rescaled_noise_sig2).repeat(1, 1, height, width),
                                  mask_repeated[:, None, :, :]],
                                 dim=1)
        else:
            rescaled_noise_sig2 = np.sqrt(retina_noise_var_to_bfcnn_noise_fn(batched_sig2_levels)) \
                                  * torch.ones_like(batched_images_nn_range[:, None, :, :])
            nn_input = torch.cat([batched_images_nn_range[:, None, :, :],
                                  rescaled_noise_sig2,
                                  mask_repeated[:, None, :, :]],
                                 dim=1)

        denoised_output = masked_denoiser(nn_input)

        return bfcnn_to_retina_fn(denoised_output).squeeze(1)

    return unblind_apply_with_mask
