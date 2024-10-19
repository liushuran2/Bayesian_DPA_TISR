# Copyright (c) OpenMMLab. All rights reserved.
# By Shuran Liu, 2023
import torch
import torch.nn as nn
from mmcv.cnn import constant_init
# from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

from model_3D.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, SPyNet)
from model_3D.models.common import PixelShufflePack, flow_warp
from model_3D.models.registry import BACKBONES
from dcn.modules.deform_conv import DeformConv

@BACKBONES.register_module()
class DPATISR_3D(nn.Module):
    """bayesian-DPA-TISR network structure.

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue. Default: 10.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self,
                 mid_channels=64,
                 extraction_nblocks=5,
                 propagation_nblocks=5,
                 reconstruction_nblocks=5,
                 factor=2,
                 max_residue_magnitude=10,
                 bayesian=False):

        super().__init__()
        self.mid_channels = mid_channels
        self.bayesian = bayesian

        # optical flow
        self.spynet = SPyNet(pretrained=None)
        # feature extraction module
        self.feat_extract = ResidualBlocksWithInputConv(1, mid_channels, extraction_nblocks)

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.phase_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels)
            self.phase_align[module] = RFFTAlignment(mid_channels)   
            self.backbone[module] = ResidualBlocksWithInputConv(
                (1 + i + 1) * mid_channels, mid_channels, propagation_nblocks)

        # reconstruction module
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, reconstruction_nblocks)
        self.upsamplemodule = PixelShufflePack(
            mid_channels, 64, factor, upsample_kernel=3)
        self.conv_hr = nn.Conv3d(64, 64, 3, 1, 1)
        if bayesian:
            self.conv_last = nn.Conv3d(64, 2, 3, 1, 1)
        else:
            self.conv_last = nn.Conv3d(64, 1, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=factor, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.


        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, d, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :, :].reshape(-1, c, d, h, w)
        lqs_2 = lqs[:, 1:, :, :, :, :].reshape(-1, c, d, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 3, d, h, w)
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 3, d, h, w)

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, d, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, d, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            # second-order deformable phase-space alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :, :]

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 4, 1))
                #
                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :, :]

                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                  flow_n1.permute(0, 2, 3, 4, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 4, 1))

                # phase-space convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = self.phase_align[module_name](feat_prop, feat_current)
                feat_n2 = self.phase_align[module_name](feat_n2, feat_current)
                # flow-guided deformable convolution
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                        flow_n1, flow_n2)
            # feat_prop = feat_current
            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
                ] + [feat_prop]

            feat = torch.cat(feat, dim=1)

            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output SR and datauncertainty sequence with shape (n, t, 2c, scale*h, scale*w).

        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            _,c,d,h,w =hr.size()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsamplemodule(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.bayesian:
                mu = hr[:,0:1,::]
                sigma = hr[:,1:2,::]
                sigma = self.sigmoid(sigma)
            else:
                mu = hr
            for depth in range(lqs.shape[3]):
                mu[:,:,depth,:,:] = mu[:,:,depth,:,:] + self.img_upsample(lqs[:, i, :, depth, :, :])
            # mu = mu + self.img_upsample(lqs[:, i, :, :, :])
            if self.bayesian:
                hr = torch.cat((mu, sigma), dim = 1)
            else:
                hr = mu

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output SR and datauncertainty sequence with shape (n, t, 2c, scale*h, scale*w).
        """

        n, t, c, d, h, w = lqs.size()

        feats = {}
        # compute spatial features
        feats_ = self.feat_extract(lqs.view(-1, c, d, h, w))
        feats_ = feats_.view(n, t, -1, d, h, w)
        feats['spatial'] = [feats_[:, i, :, :, :, :] for i in range(0, t)]

        # compute optical flow
        assert lqs.size(5) >= 64 and lqs.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(nn.Module):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue. Default: 10.

    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 deform_groups=4,
                 bias=True):
        super(SecondOrderDeformableAlignment, self).__init__()
        self.max_residue_magnitude = 10
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.deform_groups = deform_groups


        self.conv_offset = nn.Sequential(
            nn.Conv3d(3 * self.out_channels + 6, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv3d(self.out_channels, self.out_channels, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(self.out_channels, 81 * self.deform_groups, 3, 1, 1),
        )
        self.dcn = DeformConv(self.in_channels, self.out_channels, kernel_size=[3, 3, 3], stride=[1, 1, 1],padding=[1, 1, 1], deformable_groups=self.deform_groups)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        # offset = out + flow
        o1, o2 = torch.chunk(out, 2, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 3, 1,
                                                    1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 3, 1,
                                                    1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        return self.dcn(x, offset)

class RFFTAlignment(nn.Module):
    """Phase-space convolution module

    Args:
        in_channels (int): Same as nn.Conv2d.

    Return:
        Tensor: phase-aligned feature with shape (n, c, h, w)
    """

    def __init__(self, out_channels):
        self.out_channels = out_channels
        super().__init__()

        self.conv_angle = nn.Sequential(
            nn.Conv3d(2*self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv3d(self.out_channels, self.out_channels, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(self.out_channels, self.out_channels, 3, 1, 1),
        )

    def forward(self, x, feat_current):
        _,_,d,h,w = x.shape
        x_fft = torch.fft.rfftn(x, dim=(-3,-2,-1))
        x_amp = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)
        feat_current_fft = torch.fft.rfftn(feat_current, dim=(-3,-2,-1))
        feat_current_phase = torch.angle(feat_current_fft)

        phase_input = torch.cat([x_phase, feat_current_phase], dim = 1)
        phase_output = self.conv_angle(phase_input) 
        phase_output = phase_output + x_phase

        x_fft_align = torch.fft.irfftn(x_amp*torch.exp(1j * phase_output), s=(d, h, w), dim=(-3,-2,-1))


        return x_fft_align