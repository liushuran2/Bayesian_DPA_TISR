import torch
import torch.nn as nn
import torch.nn.functional as F

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.fc1 = nn.Conv3d(channel, channel // reduction, kernel_size=1)
        self.fc2 = nn.Conv3d(channel // reduction, channel, kernel_size=1)

    def forward(self, x):
        W = F.adaptive_avg_pool3d(x, 1)
        W = self.fc1(W)
        W = F.relu(W)
        W = self.fc2(W)
        W = torch.sigmoid(W)
        return x * W

class RCAB(nn.Module):
    def __init__(self, channel):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        self.ca = CALayer(channel)

    def forward(self, x):
        conv = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        conv = F.leaky_relu(self.conv2(conv), negative_slope=0.2)
        att = self.ca(conv)
        return att + x

class ResidualGroup(nn.Module):
    def __init__(self, channel, n_RCAB):
        super(ResidualGroup, self).__init__()
        self.rcabs = nn.ModuleList([RCAB(channel) for _ in range(n_RCAB)])

    def forward(self, x):
        for rcab in self.rcabs:
            x = rcab(x)
        return x

class RCAN3D(nn.Module):
    def __init__(self, input_channel, channel=64, n_ResGroup=3, n_RCAB=5):
        super(RCAN3D, self).__init__()
        self.conv_in = nn.Conv3d(input_channel, channel, kernel_size=3, padding=1)
        self.residual_groups = nn.ModuleList([ResidualGroup(channel, n_RCAB) for _ in range(n_ResGroup)])
        self.upconv = nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        self.final_conv = nn.Conv3d(channel, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for group in self.residual_groups:
            x = group(x)
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.upconv(x), negative_slope=0.2)
        x = self.final_conv(x)
        return F.leaky_relu(x, negative_slope=0.2)