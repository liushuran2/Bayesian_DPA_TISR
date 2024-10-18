# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, d, h, w).
        flow (Tensor): Tensor with size (n, d, h, w, 3). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    # if x.size()[-2:] != flow.size()[1:3]:
    #     raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
    #                      f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, d, h, w = x.size()
    # create mesh grid
    grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(0,d), torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_z, grid_x, grid_y), 3).type_as(x)  # (d, h, w, 3)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, :, 1] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, :, 2] / max(h - 1, 1) - 1.0
    grid_flow_z = 2.0 * grid_flow[:, :, :, :, 0] / max(d - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_z, grid_flow_x, grid_flow_y), dim=4)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output
