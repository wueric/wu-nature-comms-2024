import torch
import torch.nn.functional as F


def multiresolution_spatial_basis_application(raw_frames: torch.Tensor,
                                              full_scale_basis: torch.Tensor) -> torch.Tensor:
    '''

    * big_height must be an integer multiple of small_height
    * big_width must be an integer multiple of small_width

    :param raw_frames: shape (n_frames, small_height, small_width)
    :param full_scale_basis: (n_basis, big_height, big_width)
    :return: (n_frames, n_basis)
    '''

    n_frames, small_height, small_width = raw_frames.shape
    n_basis, big_height, big_width = full_scale_basis.shape

    assert big_height % small_height == 0, \
        f'height must be integer multiples, big_height {big_height}, small_height {small_height}'
    assert big_width % small_width == 0, \
        f'width must be integer multiples, big_width {big_width}, small_width {small_width}'

    scale_multiple = big_height // small_height
    assert scale_multiple == (big_width // small_width), 'width and height scale multiples must be the same'

    with torch.no_grad():

        conv_kernel = torch.ones((scale_multiple, scale_multiple), dtype=raw_frames.dtype, device=raw_frames.device)

        # shape (n_basis, small_height, small_width)
        reduced_basis = F.conv2d(full_scale_basis[:, None, :, :],
                                 conv_kernel[None, None, :, :],
                                 padding=0,
                                 stride=scale_multiple).squeeze(1)

        reduced_basis_flat = reduced_basis.reshape(reduced_basis.shape[0], -1)
        raw_frames_flat = raw_frames.reshape(raw_frames.shape[0], -1)

        # shape (n_frames, n_pix_small) @ (n_pix_small, n_basis)
        # -> (n_frames, n_basis)
        basis_applied = raw_frames_flat @ reduced_basis_flat.T

        del conv_kernel

    return basis_applied
