"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.CBAM import CBAM


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, width_multiplier=1, trilinear=True, use_ds_conv=False):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          n_classes = number of output channels/classes
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(UNet, self).__init__()
        _channels = (16, 32, 64, 128, 256)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c*width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc_adc = DoubleConv(1, self.channels[0], conv_type=self.convtype)
        self.inc_dwi = DoubleConv(1, self.channels[0], conv_type=self.convtype)
        self.inc_flair = DoubleConv(1, self.channels[0], conv_type=self.convtype)
        self.inc_bn = torch.nn.BatchNorm3d(self.channels[0])
        self.inc_relu = torch.nn.ReLU()

        factor = 2 if trilinear else 1

        self.x_concat_conv = DoubleConv(self.channels[3] * 3, self.channels[4] // factor, conv_type=self.convtype)

        self.down1_adc = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down1_dwi = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down1_flair = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.bn1 = torch.nn.BatchNorm3d(self.channels[1])
        self.relu1 = torch.nn.ReLU()

        self.down2_adc = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down2_dwi = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down2_flair = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.bn2 = torch.nn.BatchNorm3d(self.channels[2])
        self.relu2 = torch.nn.ReLU()

        self.down3_adc = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        self.down3_dwi = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        self.down3_flair = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        self.bn3 = torch.nn.BatchNorm3d(self.channels[3])
        self.relu3 = torch.nn.ReLU()

        self.down4_adc = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.down4_dwi = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.down4_flair = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.bn4 = torch.nn.BatchNorm3d(self.channels[4] // factor)
        self.relu4 = torch.nn.ReLU()

        self.up1 = Up(self.channels[4] * 6, self.channels[3] * 6 // factor, trilinear)
        self.up2 = Up(self.channels[3] * 6, self.channels[2] * 6 // factor, trilinear)
        self.up3 = Up(self.channels[2] * 6, self.channels[1] * 6 // factor, trilinear)
        self.up4 = Up(self.channels[1] * 6, self.channels[0] * 6, trilinear)
        self.outc = OutConv(self.channels[0] * 6, n_classes)

        self.cbam = CBAM(self.channels[3] * 3)

    def forward(self, x):
        dwi = x[:, 0, :]
        adc = x[:, 1, :]
        flair = x[:, 2, :]

        x1_dwi = self.inc_dwi(dwi.resize(dwi.shape[0], 1, dwi.shape[1], dwi.shape[2], dwi.shape[3]))
        x1_adc = self.inc_adc(adc.resize(adc.shape[0], 1, adc.shape[1], adc.shape[2], adc.shape[3]))
        x1_flair = self.inc_flair(flair.resize(flair.shape[0], 1, flair.shape[1], flair.shape[2], flair.shape[3]))

        x1_multi_1 = x1_dwi.sum(dim=1,keepdim=True) * x1_adc.sum(dim=1,keepdim=True) * x1_flair.sum(dim=1,keepdim=True)
        x1_dwi_1 = x1_multi_1 * x1_dwi + x1_dwi
        x1_adc_1 = x1_multi_1 * x1_adc + x1_adc
        x1_flair_1 = x1_multi_1 * x1_flair + x1_flair

        x2_dwi = self.down1_dwi(x1_dwi)
        x2_adc = self.down1_adc(x1_adc)
        x2_flair = self.down1_flair(x1_flair)

        x2_multi_2 = x2_dwi.sum(dim=1, keepdim=True) * x2_adc.sum(dim=1, keepdim=True) * x2_flair.sum(dim=1, keepdim=True)
        x2_dwi_2 = x2_multi_2 * x2_dwi + x2_dwi
        x2_adc_2 = x2_multi_2 * x2_adc + x2_adc
        x2_flair_2 = x2_multi_2 * x2_flair + x2_flair

        x3_adc = self.down2_adc(x2_adc)
        x3_dwi = self.down2_dwi(x2_dwi)
        x3_flair = self.down2_flair(x2_flair)

        x3_multi_3 = x3_dwi.sum(dim=1, keepdim=True) * x3_adc.sum(dim=1, keepdim=True) * x3_flair.sum(dim=1, keepdim=True)
        x3_dwi_3 = x3_multi_3 * x3_dwi + x3_dwi
        x3_adc_3 = x3_multi_3 * x3_adc + x3_adc
        x3_flair_3 = x3_multi_3 * x3_flair + x3_flair

        x4_adc = self.down3_adc(x3_adc)
        x4_dwi = self.down3_dwi(x3_dwi)
        x4_flair = self.down3_flair(x3_flair)

        x4_multi_4 = x4_dwi.sum(dim=1, keepdim=True) * x4_adc.sum(dim=1, keepdim=True) * x4_flair.sum(dim=1, keepdim=True)
        x4_dwi_4 = x4_multi_4 * x4_dwi + x4_dwi
        x4_adc_4 = x4_multi_4 * x4_adc + x4_adc
        x4_flair_4 = x4_multi_4 * x4_flair + x4_flair

        x5_adc = self.down4_adc(x4_adc)
        x5_dwi = self.down4_dwi(x4_dwi)
        x5_flair = self.down4_flair(x4_flair)

        x5_multi_5 = x5_dwi.sum(dim=1, keepdim=True) * x5_adc.sum(dim=1, keepdim=True) * x5_flair.sum(dim=1, keepdim=True)
        x5_dwi_5 = x5_multi_5 * x5_dwi + x5_dwi
        x5_adc_5 = x5_multi_5 * x5_adc + x5_adc
        x5_flair_5 = x5_multi_5 * x5_flair + x5_flair

        x5_concat = torch.concat((x5_dwi_5, x5_adc_5, x5_flair_5, x5_dwi, x5_adc, x5_flair), 1)
        x4_concat = torch.concat((x4_dwi_4, x4_adc_4, x4_flair_4, x4_dwi, x4_adc, x4_flair), 1)
        x3_concat = torch.concat((x3_dwi_3, x3_adc_3, x3_flair_3, x3_dwi, x3_adc, x3_flair), 1)
        x2_concat = torch.concat((x2_dwi_2, x2_adc_2, x2_flair_2, x2_dwi, x2_adc, x2_flair), 1)
        x1_concat = torch.concat((x1_dwi_1, x1_adc_1, x1_flair_1, x1_dwi, x1_adc, x1_flair), 1)

        x = self.up1(x5_concat, x4_concat)
        x = self.up2(x, x3_concat)
        x = self.up3(x, x2_concat)
        x = self.up4(x, x1_concat)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class multi_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, x3, x4):
        x1 = self.up(x1)
        # input is CHW

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x4, x3, x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


if __name__ == '__main__':
        input_tensor = torch.rand(1, 1, 64, 64, 64)
        # ideal_out = torch.rand(1, 2, 32, 32, 32)
        model = UNet(n_channels=1, n_classes=2, width_multiplier=1, trilinear=True, use_ds_conv=False)
        out = model(input_tensor)
        print(out.shape)
