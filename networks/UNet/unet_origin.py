"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNETR, SwinUNETR, AttentionUnet, VNet, RegUNet, SegResNet
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.UXNet_3D.network_backbone_old import UXNET

from torchstat import stat
from torchsummary import summary
from thop import profile

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
        # _channels = (12, 24, 48, 96, 192)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c*width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc_dwi = DoubleConv(self.n_channels, self.channels[0], conv_type=self.convtype)

        factor = 2 if trilinear else 1

        self.down1_dwi = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2_dwi = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3_dwi = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        self.down4_dwi = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)

        self.up1 = multi_Up(self.channels[4], self.channels[3] // factor, trilinear)
        self.up2 = multi_Up(self.channels[3], self.channels[2] // factor, trilinear)
        self.up3 = multi_Up(self.channels[2], self.channels[1] // factor, trilinear)
        self.up4 = multi_Up(self.channels[1], self.channels[0], trilinear)
        self.outc = OutConv(self.channels[0], n_classes)

    def forward(self, x):
        # dwi = x[:, 0, :]
        # x1_dwi = self.inc_dwi(dwi.resize(dwi.shape[0], 1, dwi.shape[1], dwi.shape[2], dwi.shape[3]))
        x1_dwi = self.inc_dwi(x)
        x2_dwi = self.down1_dwi(x1_dwi)
        x3_dwi = self.down2_dwi(x2_dwi)
        x4_dwi = self.down3_dwi(x3_dwi)
        x5_dwi = self.down4_dwi(x4_dwi)

        x = self.up1(x5_dwi, x4_dwi)
        x = self.up2(x, x3_dwi)
        x = self.up3(x, x2_dwi)
        x = self.up4(x, x1_dwi)
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
        # model = UNet(n_channels=1, n_classes=2, width_multiplier=1, trilinear=True, use_ds_conv=False)
        # model = VNet(spatial_dims=3, in_channels=1, out_channels=2, act=('elu', {'inplace': True}),
        #              dropout_prob=0.5, dropout_dim=3, bias=False)
        # model = UNETR(
        #     in_channels=1,
        #     out_channels=2,
        #     img_size=(64, 64, 64),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed="perceptron",
        #     norm_name="instance",
        #     res_block=True,
        #     dropout_rate=0.0,
        # )
        # model = nnFormer(input_channels=1, num_classes=2)
        # model = SwinUNETR(
        #     img_size=(64, 64, 64),
        #     in_channels=1,
        #     out_channels=2,
        #     feature_size=96,
        #     use_checkpoint=False,
        # )
        # model = UXNET(
        #     in_chans=1,
        #     out_chans=2,
        #     depths=[2, 2, 2, 2],
        #     feat_size=[48, 96, 192, 384],
        #     drop_path_rate=0,
        #     layer_scale_init_value=1e-6,
        #     spatial_dims=3,
        # )

        # model = UNet(n_channels=1, n_classes=2, width_multiplier=1, trilinear=True, use_ds_conv=False)
        model = SegResNet(spatial_dims=3, init_filters=8, in_channels=1, out_channels=2, dropout_prob=None, act=('RELU', {'inplace': True}), norm=('GROUP', {'num_groups': 8}), norm_name='', num_groups=8, use_conv_final=True, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),)

        flops, params = profile(model, (input_tensor,))
        print('FLOPs: ', flops, 'params: ', params)
        print('FLOPs: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
