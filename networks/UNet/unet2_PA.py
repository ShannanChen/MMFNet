"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.CBAM import CBAM
from .basic_modules import *
from .attention_blocks import FCNHead, ParallelDecoder, AttentionGate
from networks.UXNet_3D.uxnet_encoder_old import uxnet_conv
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

head_list = ['fcn', 'parallel']
head_map = {'fcn': FCNHead, 'parallel': ParallelDecoder}

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, width_multiplier=1, trilinear=True, use_ds_conv=False, head='fcn', leaky=True, norm='INSTANCE', feat_size=[48, 96, 192, 384], depths=[2, 2, 2, 2], drop_path_rate=0, hidden_size=768):
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
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.depths = depths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.drop_path_rate = drop_path_rate
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)
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

        self.up1 = Up(self.hidden_size * 3 + self.feat_size[3] * 3, (self.hidden_size * 3 + self.feat_size[3] * 3) // factor, trilinear)
        self.up2 = Up((self.hidden_size * 3 + self.feat_size[3] * 3) // factor + self.feat_size[2] * 3, ((self.hidden_size * 3 + self.feat_size[3] * 3) // factor + self.feat_size[2] * 3) // factor, trilinear)
        self.up3 = Up(((self.hidden_size * 3 + self.feat_size[3] * 3) // factor + self.feat_size[2] * 3) // factor + self.feat_size[1] * 3, (((self.hidden_size * 3 + self.feat_size[3] * 3) // factor + self.feat_size[2] * 3) // factor + self.feat_size[1] * 3) // factor, trilinear)
        self.up4 = Up((((self.hidden_size * 3 + self.feat_size[3] * 3) // factor + self.feat_size[2] * 3) // factor + self.feat_size[1] * 3) // factor + self.channels[0] * 3, self.channels[0] * 3, trilinear)
        self.outc = OutConv(self.channels[0] * 3, n_classes)

        self.cbam = CBAM(self.channels[3] * 3)

        self.head = head
        self.one_stage_dwi = head_map[head](in_channels=(192, 384, 768), out_channels=1)
        self.one_stage_adc = head_map[head](in_channels=(192, 384, 768), out_channels=1)
        self.one_stage_flair = head_map[head](in_channels=(192, 384, 768), out_channels=1)

        self.skip_3 = AttentionConnection()
        self.skip_2 = AttentionConnection()
        self.skip_1 = AttentionConnection()
        self.skip_0 = AttentionConnection()

        # deep supervision
        self.convds3 = nn.Conv3d(self.channels[3] * 3 // factor, n_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(self.channels[2] * 3 // factor, n_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(self.channels[1] * 3 // factor, n_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(self.channels[1] * 3 // factor, n_classes, kernel_size=1)

        # downsample attention
        self.conv_down_dwi = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
        self.conv_down_adc = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
        self.conv_down_flair = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=2)

        self.leaky = leaky
        self.norm = norm

        self.uxnet_3d_dwi = uxnet_conv(
            in_chans=self.n_channels,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.uxnet_3d_adc = uxnet_conv(
            in_chans=self.n_channels,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.uxnet_3d_flair = uxnet_conv(
            in_chans=self.n_channels,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )

        self.encoder1_dwi = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.n_channels,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder1_adc = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.n_channels,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder1_flair = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.n_channels,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder2_dwi = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder2_adc = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder2_flair = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder3_dwi = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder3_adc = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder3_flair = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder4_dwi = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder4_adc = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder4_flair = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder5_dwi = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder5_adc = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder5_flair = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        if is_ds:
            loss_3 = criterion(outputs['level3'], targets)
            loss_2 = criterion(outputs['level2'], targets)
            loss_1 = criterion(outputs['level1'], targets)
            multi_loss = loss_out + loss_3 + loss_2 + loss_1
        else:
            multi_loss = loss_out
        return multi_loss

    def forward(self, x):
        size = x.shape[2:]
        dwi = x[:, 0, :]
        adc = x[:, 1, :]
        flair = x[:, 2, :]

        outs_dwi = self.uxnet_3d_dwi(dwi.resize(dwi.shape[0], 1, dwi.shape[1], dwi.shape[2], dwi.shape[3]))
        outs_adc = self.uxnet_3d_adc(adc.resize(adc.shape[0], 1, adc.shape[1], adc.shape[2], adc.shape[3]))
        outs_flair = self.uxnet_3d_flair(flair.resize(flair.shape[0], 1, flair.shape[1], flair.shape[2], flair.shape[3]))

        enc1_dwi = self.encoder1_dwi(dwi.resize(dwi.shape[0], 1, dwi.shape[1], dwi.shape[2], dwi.shape[3]))
        enc1_adc = self.encoder1_adc(adc.resize(adc.shape[0], 1, adc.shape[1], adc.shape[2], adc.shape[3]))
        enc1_flair = self.encoder1_flair(flair.resize(flair.shape[0], 1, flair.shape[1], flair.shape[2], flair.shape[3]))

        x2_dwi = outs_dwi[0]
        x2_adc = outs_adc[0]
        x2_flair = outs_flair[0]
        enc2_dwi = self.encoder2_dwi(x2_dwi)
        enc2_adc = self.encoder2_adc(x2_adc)
        enc2_flair = self.encoder2_flair(x2_flair)

        x3_dwi = outs_dwi[1]
        x3_adc = outs_adc[1]
        x3_flair = outs_flair[1]
        enc3_dwi = self.encoder3_dwi(x3_dwi)
        enc3_adc = self.encoder3_adc(x3_adc)
        enc3_flair = self.encoder3_flair(x3_flair)

        x4_dwi = outs_dwi[2]
        x4_adc = outs_adc[2]
        x4_flair = outs_flair[2]
        enc4_dwi = self.encoder4_dwi(x4_dwi)
        enc4_adc = self.encoder4_adc(x4_adc)
        enc4_flair = self.encoder4_flair(x4_flair)

        enc_hidden_dwi = self.encoder5_dwi(outs_dwi[3])
        enc_hidden_adc = self.encoder5_adc(outs_adc[3])
        enc_hidden_flair = self.encoder5_flair(outs_flair[3])

        attention_dwi = self.one_stage_dwi(enc3_dwi, enc4_dwi, enc_hidden_dwi)
        attention_adc = self.one_stage_adc(enc3_adc, enc4_adc, enc_hidden_adc)
        attention_flair = self.one_stage_flair(enc3_flair, enc4_flair, enc_hidden_flair)

        attention_multi = attention_dwi * attention_adc * attention_flair
        attention_dwi = attention_multi + attention_dwi
        attention_adc = attention_multi + attention_adc
        attention_flair = attention_multi + attention_flair

        act_attention_dwi = torch.sigmoid(attention_dwi)  # attention shape is the same as x2
        act_attention_adc = torch.sigmoid(attention_adc)  # attention shape is the same as x2
        act_attention_flair = torch.sigmoid(attention_flair)  # attention shape is the same as x2

        attention_x3_dwi = self.conv_down_dwi(act_attention_dwi)
        attention_x3_adc = self.conv_down_adc(act_attention_adc)
        attention_x3_flair = self.conv_down_flair(act_attention_flair)

        attention3_1_dwi = F.interpolate(attention_x3_dwi, size=x4_dwi.shape[2:], mode='trilinear', align_corners=False)
        attention2_2_dwi = F.interpolate(act_attention_dwi, size=x3_dwi.shape[2:], mode='trilinear', align_corners=False)
        attention1_3_dwi = F.interpolate(act_attention_dwi, size=x2_dwi.shape[2:], mode='trilinear', align_corners=False)
        attention0_4_dwi = F.interpolate(act_attention_dwi, size=enc1_dwi.shape[2:], mode='trilinear', align_corners=False)

        attention3_1_adc = F.interpolate(attention_x3_adc, size=x4_dwi.shape[2:], mode='trilinear', align_corners=False)
        attention2_2_adc = F.interpolate(act_attention_adc, size=x3_dwi.shape[2:], mode='trilinear', align_corners=False)
        attention1_3_adc = F.interpolate(act_attention_adc, size=x2_dwi.shape[2:], mode='trilinear', align_corners=False)
        attention0_4_adc = F.interpolate(act_attention_adc, size=enc1_dwi.shape[2:], mode='trilinear', align_corners=False)

        attention3_1_flair = F.interpolate(attention_x3_flair, size=x4_dwi.shape[2:], mode='trilinear', align_corners=False)
        attention2_2_flair = F.interpolate(act_attention_flair, size=x3_dwi.shape[2:], mode='trilinear',align_corners=False)
        attention1_3_flair = F.interpolate(act_attention_flair, size=x2_dwi.shape[2:], mode='trilinear',align_corners=False)
        attention0_4_flair = F.interpolate(act_attention_flair, size=enc1_dwi.shape[2:], mode='trilinear',align_corners=False)

        x5_concat = torch.concat((enc_hidden_dwi, enc_hidden_adc, enc_hidden_flair), 1)

        x3_1 = self.up1(x5_concat, self.skip_3(enc4_dwi, attention3_1_dwi), self.skip_3(enc4_adc, attention3_1_adc), self.skip_3(enc4_flair, attention3_1_flair))
        x2_2 = self.up2(x3_1, self.skip_2(enc3_dwi, attention2_2_dwi), self.skip_2(enc3_adc, attention2_2_adc), self.skip_2(enc3_flair, attention2_2_flair))
        x1_3 = self.up3(x2_2, self.skip_1(enc2_dwi, attention1_3_dwi), self.skip_1(enc2_adc, attention1_3_adc), self.skip_1(enc2_flair, attention1_3_flair))
        x0_4 = self.up4(x1_3, self.skip_0(enc1_dwi, attention0_4_dwi), self.skip_0(enc1_adc, attention0_4_adc), self.skip_0(enc1_flair, attention0_4_flair))
        # logits = self.outc(x)

        out = dict()
        out['stage1'] = F.interpolate(attention_dwi, size=size, mode='trilinear', align_corners=False)  # intermediate
        out['stage2'] = F.interpolate(attention_adc, size=size, mode='trilinear', align_corners=False)  # intermediate
        out['stage3'] = F.interpolate(attention_flair, size=size, mode='trilinear', align_corners=False)  # intermediate
        out['level3'] = F.interpolate(self.convds3(x3_1), size=size, mode='trilinear', align_corners=False)
        out['level2'] = F.interpolate(self.convds2(x2_2), size=size, mode='trilinear', align_corners=False)
        out['level1'] = F.interpolate(self.convds1(x1_3), size=size, mode='trilinear', align_corners=False)
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out


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
