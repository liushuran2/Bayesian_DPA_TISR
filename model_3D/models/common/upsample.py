# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from .sr_backbone_utils import default_init_weights


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv3d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        _,c,d,h,w = x.size()
        x_ = x.permute(0,2,1,3,4).view(-1,c,h,w)
        x_ = F.pixel_shuffle(x_, self.scale_factor)

        x = x_.view(_,d,c//(self.scale_factor*self.scale_factor),h*self.scale_factor,w*self.scale_factor).permute(0,2,1,3,4)
        # x = F.pixel_shuffle(x, self.scale_factor)
        return x
