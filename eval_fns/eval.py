import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Union, Tuple

import tqdm


def apply_masked_mse(masked_mse_module,
                     gt_images: np.ndarray,
                     recons_images: np.ndarray,
                     batch_size: int,
                     device) -> np.ndarray:

    n_images, hh, ww = gt_images.shape
    output = np.zeros((n_images, ), dtype=np.float32)

    with torch.no_grad():
        pbar = tqdm.tqdm(total=((n_images // batch_size) + 1))

        for low in range(0, n_images, batch_size):
            high = min(low + batch_size, n_images)

            batch_gt_torch = torch.tensor(gt_images[low:high, ...],
                                          dtype=torch.float32, device=device)
            batch_recons_torch = torch.tensor(recons_images[low:high, ...],
                                              dtype=torch.float32, device=device)

            batched_mse = masked_mse_module(
                batch_gt_torch,
                batch_recons_torch
            ).detach().cpu().numpy()

            output[low:high] = batched_mse
            pbar.update(1)

        pbar.close()

    return output


def apply_masked_ms_ssim(masked_ms_ssim_module,
                         gt_images: np.ndarray,
                         recons_images: np.ndarray,
                         batch_size: int,
                         device) -> np.ndarray:

    n_images, hh, ww = gt_images.shape
    output = np.zeros((n_images, ), dtype=np.float32)

    with torch.no_grad():
        pbar = tqdm.tqdm(total=((n_images // batch_size) + 1))
        for low in range(0, n_images, batch_size):
            high = min(low + batch_size, n_images)

            batch_gt_torch = torch.tensor(gt_images[low:high, ...],
                                          dtype=torch.float32, device=device)
            batch_recons_torch = torch.tensor(recons_images[low:high, ...],
                                              dtype=torch.float32, device=device)

            ms_ssim = masked_ms_ssim_module(
                batch_gt_torch[:, None, :, :],
                batch_recons_torch[:, None, :, :]
            ).detach().cpu().numpy()

            output[low:high] = ms_ssim
            pbar.update(1)
        pbar.close()

    return output


def apply_simple_masked_lpips(lpips_loss_fn,
                              gt_images: np.ndarray,
                              target_images: np.ndarray,
                              mask: np.ndarray,
                              batch_size: int,
                              device: torch.device) -> np.ndarray:
    n_images, hh, ww = target_images.shape

    masked_gt_images = gt_images * mask[None, :, :]
    masked_target_images = target_images * mask[None, :, :]

    output = np.zeros((n_images,), dtype=np.float32)

    with torch.no_grad():
        mask_torch = torch.tensor(mask, dtype=torch.float32, device=device)
        pbar = tqdm.tqdm(total=((n_images // batch_size) + 1))
        for low in range(0, n_images, batch_size):
            high = min(low + batch_size, n_images)

            batch1_gt = masked_gt_images[low:high, ...]
            batch1_target = masked_target_images[low:high, ...]

            batch1_gt_torch = torch.tensor(batch1_gt, dtype=torch.float32,
                                           device=device)[:, None, :, :].expand(-1, 3, -1, -1)
            batch1_target_torch = torch.tensor(batch1_target, dtype=torch.float32,
                                               device=device)[:, None, :, :].expand(-1, 3, -1, -1)

            lpips_spat = lpips_loss_fn(batch1_gt_torch, batch1_target_torch)
            masked_lpips_spat = mask_torch[None, None, :, :] * lpips_spat

            lpips_result = (torch.sum(masked_lpips_spat, dim=(2, 3)) / torch.sum(mask_torch)).squeeze(1)

            output[low:high] = lpips_result.detach().cpu().numpy()

            pbar.update(1)
        pbar.close()

    return output


def batched_minimize_mse_and_return_image(
        ground_truth: np.ndarray,
        experimental_images: np.ndarray,
        mse_module,
        max_conv_distance: int,
        device: torch.device,
        batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:

    n_images, height, width = ground_truth.shape
    mse_scores = np.zeros((n_images, ), dtype=np.float32)
    best_shifted_images = np.zeros((n_images, height, width))

    pbar = tqdm.tqdm(total=(n_images // batch_size)+1)
    for low in range(0, n_images, batch_size):

        high = min(n_images, low + batch_size)
        gt_np = ground_truth[low:high, ...]
        exp_np = experimental_images[low:high, ...]

        with torch.no_grad():
            gt_torch = torch.tensor(gt_np, dtype=torch.float32, device=device)
            exp_torch = torch.tensor(exp_np, dtype=torch.float32, device=device)

            gt_torch = torch.clamp((gt_torch + 1.0) / 2.0, min=0.0, max=1.0)
            exp_torch = torch.clamp((exp_torch + 1.0) / 2.0, min=0.0, max=1.0)

        mse, best_ims = compute_convolutional_min_metric(
            mse_module,
            gt_torch,
            exp_torch,
            max_conv_distance,
            device,
            batch_size=batch_size,
            return_shifted_images=True
        )

        mse_scores[low:high] = mse.detach().cpu().numpy()
        best_shifted_images[low:high, ...] =(2.0 * best_ims.detach().cpu().numpy()) - 1

        pbar.update(1)

    pbar.close()

    return mse_scores, best_shifted_images


def batched_compute_convolutional_masked_ms_ssim(
        ground_truth: np.ndarray,
        experimental_images: np.ndarray,
        ms_ssim_module,
        max_conv_distance: int,
        device,
        batch_size: int = 16) -> np.ndarray:
    n_images = ground_truth.shape[0]
    ms_ssim_scores = np.zeros((n_images,), dtype=np.float32)
    for low in range(0, n_images, batch_size):
        high = min(n_images, low + batch_size)
        gt_np = ground_truth[low:high, ...]
        exp_np = experimental_images[low:high, ...]

        with torch.no_grad():
            gt_torch = torch.tensor(gt_np, dtype=torch.float32, device=device)
            exp_torch = torch.tensor(exp_np, dtype=torch.float32, device=device)

            gt_torch = torch.clamp((gt_torch + 1.0) / 2.0, min=0.0, max=1.0)
            exp_torch = torch.clamp((exp_torch + 1.0) / 2.0, min=0.0, max=1.0)

            computed_similarity = compute_convolutional_max_metric(
                ms_ssim_module,
                gt_torch,
                exp_torch,
                max_conv_distance,
                device,
                batch_size=batch_size).detach().cpu().numpy()

        ms_ssim_scores[low:high] = computed_similarity

    return ms_ssim_scores


def batched_computed_convolutional_masked_LPIPS(
        ground_truth: np.ndarray,
        experimental_images: np.ndarray,
        valid_mask: np.ndarray,
        lpips_fn,
        max_conv_distance: int,
        device,
        batch_size: int = 2) -> np.ndarray:
    n_images = ground_truth.shape[0]
    lpips_scores = np.zeros((n_images,), dtype=np.float32)

    valid_mask_torch = torch.tensor(valid_mask, dtype=torch.float32,
                                    device=device)

    pbar = tqdm.tqdm(total=n_images // batch_size)
    for low in range(0, n_images, batch_size):
        high = min(n_images, low + batch_size)
        gt_np = ground_truth[low:high, ...]
        exp_np = experimental_images[low:high, ...]

        with torch.no_grad():
            gt_torch = torch.tensor(gt_np, dtype=torch.float32, device=device)
            exp_torch = torch.tensor(exp_np, dtype=torch.float32, device=device)

            gt_torch = torch.clamp((gt_torch + 1.0) / 2.0, min=0.0, max=1.0)
            exp_torch = torch.clamp((exp_torch + 1.0) / 2.0, min=0.0, max=1.0)

            lpips_batch = compute_convolutional_min_LPIPS(
                lpips_fn,
                gt_torch,
                exp_torch,
                valid_mask_torch,
                max_conv_distance,
                device,
                batch_size=1
            ).detach().cpu().numpy()

        lpips_scores[low:high] = lpips_batch

        pbar.update(1)

    pbar.close()
    return lpips_scores


class MaskedPSNR(nn.Module):

    def __init__(self, mask: np.ndarray,
                 max_val: float = 1.0,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer('mask', torch.tensor(mask, dtype=dtype))
        self.n_pix = np.sum(mask)
        self.max_val = max_val

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

        diff = a - b
        square_diff = torch.square(diff)
        if a.ndim == self.mask.ndim:
            masked_diff = self.mask * square_diff
            mse = torch.sum(masked_diff) / self.n_pix
            return 20 * np.log10(self.max_val) - 10 * torch.log10(mse)

        elif a.ndim == self.mask.ndim + 1:

            masked_diff = self.mask[None, ...] * square_diff
            summed_diff = torch.sum(masked_diff, dim=(1, 2))
            mse = summed_diff / self.n_pix
            return 20 * np.log10(self.max_val) - 10 * torch.log10(mse)


class MaskedPSNR2(nn.Module):

    def __init__(self, mask: np.ndarray,
                 max_val: float = 1.0,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer('mask', torch.tensor(mask, dtype=dtype))
        self.n_pix = np.sum(mask)
        self.max_val = max_val

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        '''

        :param a: shape (..., height, width)
        :param b: shape (..., height, width)
        :return:
        '''

        ndim = a.ndim

        square_diff = torch.square(a - b)

        mask_expansion_list = [1 for _ in range(ndim - 2)]
        mask_expansion_list.extend([-1, -1])
        masked_diff = self.mask.expand(*mask_expansion_list) * square_diff

        summed_diff = torch.sum(masked_diff, dim=(-1, -2))

        mse = summed_diff / self.n_pix
        return 20 * np.log10(self.max_val) - 10 * torch.log10(mse)


class MaskedMSELoss(nn.Module):

    def __init__(self, mask: np.ndarray,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer('mask', torch.tensor(mask, dtype=dtype))
        self.n_pix = np.sum(mask)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

        square_diff = torch.square(a - b)
        ndim = square_diff.ndim

        mask_expansion_list = [1 for _ in range(ndim - 2)]
        mask_expansion_list.extend([-1, -1])
        masked_diff = self.mask.expand(*mask_expansion_list) * square_diff

        summed_diff = torch.sum(masked_diff, dim=(-1, -2))
        return summed_diff / self.n_pix


# Everything below this line was originally written by Gongfan Fang
# Modifications were made by Eric Wu to mask invalid regions of the image
# in the SSIM calculation
# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input_tensors, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input_tensors.shape) == 4:
        conv = F.conv2d
    elif len(input_tensors.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input_tensors.shape)

    C = input_tensors.shape[1]
    out = input_tensors
    for i, s in enumerate(input_tensors.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input_tensors.shape} and win size: {win.shape[-1]}"
            )

    return out


def _precomputed_masked_ssim(X: torch.Tensor,
                             Y: torch.Tensor,
                             precomputed_valid_mask: torch.Tensor,
                             data_range: Union[float, int],
                             win: torch.Tensor,
                             size_average=True,
                             K=(0.01, 0.03)):
    """ Calculate ssim index for X and Y

        Args:
            X (torch.Tensor): images
            Y (torch.Tensor): images
            win (torch.Tensor): 1-D gauss kernel
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            precomputed_valid_mask (torch.Tensor): shape of the output of conv(X, win), floating point valued,
                either 0.0 if the pixel in invalid or 1.0 if the pixel is valid

        Returns:
            torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if ssim_map.ndim == 4 or ssim_map.ndim == 5:
        masked_ssim_map = precomputed_valid_mask[None, None, ...] * ssim_map
        masked_cs_map = precomputed_valid_mask[None, None, ...] * cs_map

        n_valid = torch.sum(precomputed_valid_mask)

        ssim_per_channel = torch.flatten(masked_ssim_map, 2).sum(-1) / n_valid
        cs = torch.flatten(masked_cs_map, 2).sum(-1) / n_valid

        return ssim_per_channel, cs

    else:
        assert False, 'something really bad happened'


def masked_ssim(
        X: torch.Tensor,
        Y: torch.Tensor,
        precomputed_valid_mask: torch.Tensor,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        win=None,
        K=(0.01, 0.03),
        nonnegative_ssim=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _precomputed_masked_ssim(X, Y, precomputed_valid_mask, data_range=data_range, win=win,
                                                    size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def masked_ms_ssim(
        X, Y, weights, level_valid_masks, data_range=255, size_average=True, win_size=11,
        win_sigma=1.5, win=None, K=(0.01, 0.03)):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        weights (torch.Tensor): weights for different levels
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    if weights.shape[0] != len(level_valid_masks):
        raise ValueError("Must have the same number of MS-SSIM weights as valid masks")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
            2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i, valid_mask in enumerate(level_valid_masks):

        # plt.figure()
        # plt.imshow(not_valid_iter.detach().cpu().numpy(), cmap='gray')
        # plt.show()
        ssim_per_channel, cs = _precomputed_masked_ssim(X, Y, valid_mask,
                                                        win=win, data_range=data_range, size_average=False, K=K)
        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class MaskedSSIM(nn.Module):
    def __init__(self,
                 not_valid_mask: np.ndarray,
                 device: torch.device,
                 data_range=255,
                 size_average=True,
                 win_size: int = 11,
                 win_sigma=1.5,
                 channel=3,
                 spatial_dims=2,
                 K=(0.01, 0.03),
                 nonnegative_ssim=False):
        '''
        Valid region modification by Eric Wu

        Note that we have to precompute the convolutionally-valid region from not_valid_mask,
            and keep that quantity around as a non-differentiable parameter in the model
        :param not_valid_mask: 
        :param data_range: 
        :param size_average: 
        :param win_size: 
        :param win_sigma: 
        :param channel: 
        :param spatial_dims: 
        :param K: 
        :param nonnegative_ssim: 
        '''

        super().__init__()

        self.win_size = win_size

        self.register_buffer('win',
                             _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims).to(
                                 device))

        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

        with torch.no_grad():
            not_valid_torch = torch.tensor(not_valid_mask, dtype=self.win.dtype, device=device)

            blurred_not_valid = gaussian_filter(not_valid_torch[None, None, :, :], self.win).squeeze(1).squeeze(0)
            is_valid_mask = (blurred_not_valid == 0.0).to(torch.float32)

            del not_valid_torch, blurred_not_valid

        self.register_buffer('precomputed_valid_mask', is_valid_mask)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        return masked_ssim(
            X,
            Y,
            self.precomputed_valid_mask,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class Masked_MS_SSIM(nn.Module):
    def __init__(
            self,
            not_valid_mask: np.ndarray,
            weights: np.ndarray,
            device: torch.device,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            K=(0.01, 0.03)):
        super().__init__()

        self.win_size = win_size

        self.register_buffer('win',
                             _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims).to(
                                 device))

        self.size_average = size_average
        self.data_range = data_range
        self.K = K

        self.register_buffer('weights', torch.tensor(weights, dtype=self.win.dtype, device=device))

        self.stacked_valid_mask = nn.ParameterList()

        shapes = [a for a in not_valid_mask.shape]

        with torch.no_grad():
            not_valid_iter = torch.tensor(not_valid_mask, dtype=self.win.dtype, device=device)

            # need to invert and add
            # now we have to do the convolutions
            blurred_not_valid = gaussian_filter(not_valid_iter[None, None, :, :], self.win).squeeze(1).squeeze(0)
            is_valid_mask = (blurred_not_valid == 0.0).to(torch.float32)

            self.stacked_valid_mask.append(nn.Parameter(is_valid_mask, requires_grad=False))

            for i in range(weights.shape[0] - 1):
                padding = [s % 2 for s in shapes]
                temp_not_valid = F.avg_pool2d(not_valid_iter[None, None, ...],
                                              kernel_size=2, padding=padding).squeeze(1).squeeze(0)
                shapes = [a for a in temp_not_valid.shape]
                not_valid_iter = (temp_not_valid != 0.0).to(torch.float32)

                # now we have to do the convolutions
                blurred_not_valid = gaussian_filter(not_valid_iter[None, None, :, :], self.win).squeeze(1).squeeze(0)
                is_valid_mask = (blurred_not_valid == 0.0).to(torch.float32)

                self.stacked_valid_mask.append(nn.Parameter(is_valid_mask, requires_grad=False))

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        return masked_ms_ssim(X, Y, self.weights, self.stacked_valid_mask,
                              data_range=self.data_range,
                              size_average=self.size_average,
                              win=self.win,
                              K=self.K)


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03),
          not_valid=None):
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        not_valid (torch.Tensor): same shape as X and Y, valued 1.0 corresponding pixel is not valid,
            valued 0.0 if corresponding pixel is valid; note that this must be float-valued

    Returns:
        torch.Tensor: ssim results.
    """

    uses_mask = (not_valid is not None)

    if uses_mask:
        assert not_valid.shape == X.shape[2:], f'not_valid.shape {not_valid.shape} must match X.shape {X.shape[2:]}'
        assert not_valid.shape == Y.shape[2:], f'not_valid.shape {not_valid.shape} must match Y.shape {Y.shape[2:]}'

    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if uses_mask:
        blurred_not_valid = gaussian_filter(not_valid[None, None, :, :], win).squeeze(1).squeeze(0)
        is_valid_mask = (blurred_not_valid == 0.0).to(torch.float32)

        if ssim_map.ndim == 4 or ssim_map.ndim == 5:
            masked_ssim_map = is_valid_mask[None, None, ...] * ssim_map
            masked_cs_map = is_valid_mask[None, None, ...] * cs_map

            n_valid = torch.sum(is_valid_mask)

            ssim_per_channel = torch.flatten(masked_ssim_map, 2).sum(-1) / n_valid
            cs = torch.flatten(masked_cs_map, 2).sum(-1) / n_valid

            return ssim_per_channel, cs

        else:
            assert False, 'something really bad happened'

    else:

        ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
        cs = torch.flatten(cs_map, 2).mean(-1)

        return ssim_per_channel, cs


def ssim(
        X,
        Y,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        win=None,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
        not_valid=None):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K, not_valid=not_valid)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
        X, Y, data_range=255, size_average=True, win_size=11,
        win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03),
        not_valid=None):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
            2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    not_valid_iter = not_valid
    for i in range(levels):

        # plt.figure()
        # plt.imshow(not_valid_iter.detach().cpu().numpy(), cmap='gray')
        # plt.show()
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K,
                                     not_valid=not_valid_iter)
        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

            if not_valid_iter is not None:
                temp_not_valid = avg_pool(not_valid_iter[None, None, ...],
                                          kernel_size=2, padding=padding).squeeze(1).squeeze(0)
                not_valid_iter = (temp_not_valid != 0.0).to(torch.float32)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            K=(0.01, 0.03),
            nonnegative_ssim=False,
            not_valid_mask=None
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

        self.not_valid_mask = not_valid_mask

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
            not_valid=self.not_valid_mask
        )


class MS_SSIM(nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            weights=None,
            K=(0.01, 0.03),
            not_valid_mask=None
    ):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K
        self.not_valid_mask = not_valid_mask

    def forward(self, X, Y):
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
            not_valid=self.not_valid_mask
        )


def compute_convolutional_min_LPIPS(
        lpips_fn: nn.Module,
        ground_truth: torch.Tensor,
        maybe_shifted: torch.Tensor,
        valid_mask: torch.Tensor,
        max_shifts: int,
        device: torch.device,
        batch_size: int = 1) -> torch.Tensor:
    '''
    Function for computing the minimum possible LPIPS while
        shifting the reconstructed image, since the notion of
        pixel-wise matching is a little weaker when you have
        to account for eye movements

    '''

    n_images, height, width = ground_truth.shape

    conv_dim = (2 * max_shifts + 1)
    n_possible_shifts = conv_dim * conv_dim

    # shape (conv_dim, conv_dim)
    conv_shifter = np.eye(n_possible_shifts).reshape(n_possible_shifts, conv_dim, conv_dim)

    pbar = tqdm.tqdm(total=n_images // batch_size)

    with torch.no_grad():
        conv_filters = torch.tensor(conv_shifter, dtype=ground_truth.dtype, device=device)
        outputs = torch.zeros((n_images,), dtype=ground_truth.dtype, device=device)

        for low in range(0, n_images, batch_size):
            high = min(low + batch_size, n_images)

            # shape (batch_size, height, width)
            orig_images = ground_truth[low:high, ...] * valid_mask[None, :, :]

            # shape (batch_size, height, width)
            recons_images = maybe_shifted[low:high, ...] * valid_mask[None, :, :]

            # now we have to generate the shifted versions of the
            # reconstructed images

            # shape (batch_size, n_possible_shifts, height, width)
            # -> (batch_size * n_possible_shifts, height, width)
            shifted_recons_images = F.conv2d(recons_images[:, None, :, :],
                                             conv_filters[:, None, :, :],
                                             padding='same').reshape(-1, height, width)

            orig_images_expanded = orig_images[:, None, :, :].expand(
                -1, n_possible_shifts, -1, -1).reshape(-1, height, width)

            lpips_spat = lpips_fn(shifted_recons_images[:, None, :, :].expand(-1, 3, -1, -1),
                                  orig_images_expanded[:, None, :, :].expand(-1, 3, -1, -1))
            # shape (batch_size * n_possible_shifts, height, width)
            masked_lpips_spat = valid_mask[None, None, :, :] * lpips_spat

            # shape (batch_size, n_possible_shifts)
            lpips_result = torch.sum(masked_lpips_spat, dim=(2, 3)).reshape(batch_size, n_possible_shifts)
            min_lpips, _ = torch.min(lpips_result, dim=1)

            min_lpips_normalized = min_lpips / torch.sum(valid_mask)

            outputs[low:high] = min_lpips_normalized

            pbar.update(1)

    pbar.close()

    return outputs


def compute_convolutional_min_metric(
        metric_fn: nn.Module,
        ground_truth: torch.Tensor,
        maybe_shifted: torch.Tensor,
        max_shifts: int,
        device: torch.device,
        batch_size: int = 1,
        return_shifted_images: bool = False) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''

    :param metric_fn:
    :param ground_truth: shape (n_images, height, width)
    :param maybe_shifted: shape (n_images, height, width)
    :param max_shifts:
    :param device:
    :param batch_size:
    :return:
    '''

    n_images, height, width = ground_truth.shape

    conv_dim = (2 * max_shifts + 1)
    n_possible_shifts = conv_dim * conv_dim

    # shape (conv_dim, conv_dim)
    conv_shifter = np.eye(n_possible_shifts).reshape(n_possible_shifts, conv_dim, conv_dim)

    with torch.no_grad():
        conv_filters = torch.tensor(conv_shifter, dtype=ground_truth.dtype,
                                    device=device)

        outputs = torch.zeros((n_images,), dtype=ground_truth.dtype, device=device)
        if return_shifted_images:
            output_images = torch.zeros((n_images, height, width), dtype=ground_truth.dtype, device=device)

        for low in range(0, n_images, batch_size):
            high = min(low + batch_size, n_images)

            # shape (batch_size, height, width)
            orig_images = ground_truth[low:high, ...]

            # shape (batch_size, height, width)
            recons_images = maybe_shifted[low:high, ...]

            # now we have to generate the shifted versions of the
            # reconstructed images

            # shape (batch_size, n_possible_shifts, height, width)
            conv_shifted_images = F.conv2d(recons_images[:, None, :, :],
                                             conv_filters[:, None, :, :],
                                             padding='same')

            orig_images_expanded = orig_images[:, None, :, :].expand(
                -1, n_possible_shifts, -1, -1)

            # shape (batch_size, n_possible_shifts)
            similarity = metric_fn(orig_images_expanded,
                                   conv_shifted_images)

            # shape (batch_size, ) and shape (batch_size, )
            max_similarity, indices = torch.min(similarity, dim=1)

            if return_shifted_images:
                output_images[low:high] = torch.gather(
                        conv_shifted_images,
                        1,
                        indices[:, None, None, None].expand(-1, -1, height, width)).squeeze(1)

            outputs[low:high] = max_similarity

    if return_shifted_images:
        return outputs, output_images
    return outputs


def compute_convolutional_max_metric(
        metric_fn: nn.Module,
        ground_truth: torch.Tensor,
        maybe_shifted: torch.Tensor,
        max_shifts: int,
        device: torch.device,
        batch_size: int = 1) -> torch.Tensor:
    '''

    :param metric_fn:
    :param ground_truth: shape (n_images, height, width)
    :param maybe_shifted: shape (n_images, height, width)
    :param max_shifts:
    :param device:
    :param batch_size:
    :return:
    '''

    n_images, height, width = ground_truth.shape

    conv_dim = (2 * max_shifts + 1)
    n_possible_shifts = conv_dim * conv_dim

    # shape (conv_dim, conv_dim)
    conv_shifter = np.eye(n_possible_shifts).reshape(n_possible_shifts, conv_dim, conv_dim)

    with torch.no_grad():
        conv_filters = torch.tensor(conv_shifter, dtype=ground_truth.dtype,
                                    device=device)

        outputs = torch.zeros((n_images,), dtype=ground_truth.dtype, device=device)

        for low in range(0, n_images, batch_size):
            high = min(low + batch_size, n_images)

            # shape (batch_size, height, width)
            orig_images = ground_truth[low:high, ...]

            # shape (batch_size, height, width)
            recons_images = maybe_shifted[low:high, ...]

            # now we have to generate the shifted versions of the
            # reconstructed images

            # shape (batch_size, n_possible_shifts, height, width)
            shifted_recons_images = F.conv2d(recons_images[:, None, :, :],
                                             conv_filters[:, None, :, :],
                                             padding='same').reshape(-1, height, width)

            orig_images_expanded = orig_images[:, None, :, :].expand(
                -1, n_possible_shifts, -1, -1).reshape(-1, height, width)

            # shape (batch_size, n_possible_shifts)
            similarity = metric_fn(orig_images_expanded[:, None, :, :],
                                   shifted_recons_images[:, None, :, :])

            similarity = similarity.reshape((high - low), n_possible_shifts)

            # shape (batch_size, )
            max_similarity, _ = torch.max(similarity, dim=1)

            outputs[low:high] = max_similarity

    return outputs
