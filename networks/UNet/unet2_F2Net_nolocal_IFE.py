"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from attention.CBAM import CBAM
from attention.non_local_dot_product import _NonLocalBlockND_spitail, _NonLocalBlockND_channel, _NonLocalBlockND_multi_spitial, _NonLocalBlockND_multi_channel_paper4
from typing import List
import math


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
        self.down1_fusion = Down(self.channels[0], self.channels[1], conv_type=self.convtype)

        self.down2_adc = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down2_dwi = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down2_flair = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down2_fusion = Down(self.channels[1] + self.channels[0], self.channels[2], conv_type=self.convtype)

        self.down3_adc = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        self.down3_dwi = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        self.down3_flair = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        self.down3_fusion = Down(self.channels[2] + + self.channels[1], self.channels[3], conv_type=self.convtype)

        self.down4_adc = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.down4_dwi = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.down4_flair = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.down4_fusion = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)

        self.up1_dwi = Up(self.channels[4], self.channels[3] // factor, trilinear)
        self.up2_dwi = Up(self.channels[3] + self.channels[1], self.channels[2] // factor, trilinear)
        self.up3_dwi = Up(self.channels[2] + self.channels[0], self.channels[1] // factor, trilinear)
        self.up4_dwi = Up(self.channels[1], self.channels[0], trilinear)
        self.outc_dwi = OutConv(self.channels[0], n_classes)

        self.up1_adc = Up(self.channels[4], self.channels[3] // factor, trilinear)
        self.up2_adc = Up(self.channels[3] + self.channels[1], self.channels[2] // factor, trilinear)
        self.up3_adc = Up(self.channels[2] + self.channels[0], self.channels[1] // factor, trilinear)
        self.up4_adc = Up(self.channels[1], self.channels[0], trilinear)
        self.outc_adc = OutConv(self.channels[0], n_classes)

        self.up1_flair = Up(self.channels[4], self.channels[3] // factor, trilinear)
        self.up2_flair = Up(self.channels[3] + self.channels[1], self.channels[2] // factor, trilinear)
        self.up3_flair = Up(self.channels[2] + self.channels[0], self.channels[1] // factor, trilinear)
        self.up4_flair = Up(self.channels[1], self.channels[0], trilinear)
        self.outc_flair = OutConv(self.channels[0], n_classes)

        self.up1_fusion = Up(640, self.channels[3] // factor, trilinear)
        self.up2_fusion = Up(448, self.channels[2] // factor, trilinear)
        self.up3_fusion = Up(224, self.channels[1] // factor, trilinear)
        self.up4_fusion = Up(80, self.channels[0], trilinear)
        self.outc_fusion = OutConv(self.channels[0], n_classes)

        self.fusion1 = EnhancedFeature_nonlocal(self.channels[0], is_first=True)
        self.fusion2 = EnhancedFeature_nonlocal_IFE(self.channels[1] + self.channels[0], is_first=False)
        self.fusion3 = EnhancedFeature_nonlocal_IFE(self.channels[2] + self.channels[1], is_first=False)
        self.fusion4 = EnhancedFeature_nonlocal(self.channels[3], is_first=False)

        self.ratio = [0.5, 0.5]
        self.ife1 = Entropy_Hist(self.ratio[0])
        self.ife2 = Entropy_Hist(self.ratio[1])

    def forward(self, x):
        dwi = x[:, 0, :]
        adc = x[:, 1, :]
        flair = x[:, 2, :]

        x1_dwi = self.inc_dwi(dwi.resize(dwi.shape[0], 1, dwi.shape[1], dwi.shape[2], dwi.shape[3]))
        x1_adc = self.inc_adc(adc.resize(adc.shape[0], 1, adc.shape[1], adc.shape[2], adc.shape[3]))
        x1_flair = self.inc_flair(flair.resize(flair.shape[0], 1, flair.shape[1], flair.shape[2], flair.shape[3]))

        x2_dwi = self.down1_dwi(x1_dwi)
        x2_adc = self.down1_adc(x1_adc)
        x2_flair = self.down1_flair(x1_flair)

        x3_adc = self.down2_adc(x2_adc)
        x3_dwi = self.down2_dwi(x2_dwi)
        x3_flair = self.down2_flair(x2_flair)

        x4_adc = self.down3_adc(x3_adc)
        x4_dwi = self.down3_dwi(x3_dwi)
        x4_flair = self.down3_flair(x3_flair)

        x5_adc = self.down4_adc(x4_adc)
        x5_dwi = self.down4_dwi(x4_dwi)
        x5_flair = self.down4_flair(x4_flair)

        x2_dwi = torch.cat([x2_dwi, self.ife1(x2_dwi)], dim=1)
        x3_dwi = torch.cat([x3_dwi, self.ife2(x3_dwi)], dim=1)

        x2_adc = torch.cat([x2_adc, self.ife1(x2_adc)], dim=1)
        x3_adc = torch.cat([x3_adc, self.ife2(x3_adc)], dim=1)

        x2_flair = torch.cat([x2_flair, self.ife1(x2_flair)], dim=1)
        x3_flair = torch.cat([x3_flair, self.ife2(x3_flair)], dim=1)

        x = self.up1_dwi(x5_dwi, x4_dwi)
        x = self.up2_dwi(x, x3_dwi)
        x = self.up3_dwi(x, x2_dwi)
        x = self.up4_dwi(x, x1_dwi)
        logits_dwi = self.outc_dwi(x)

        x = self.up1_adc(x5_adc, x4_adc)
        x = self.up2_adc(x, x3_adc)
        x = self.up3_adc(x, x2_adc)
        x = self.up4_adc(x, x1_adc)
        logits_adc = self.outc_adc(x)

        x = self.up1_flair(x5_flair, x4_flair)
        x = self.up2_flair(x, x3_flair)
        x = self.up3_flair(x, x2_flair)
        x = self.up4_flair(x, x1_flair)
        logits_flair = self.outc_flair(x)

        x1_fusion = self.fusion1(0, x1_dwi, x1_adc, x1_flair)
        x2_fusion = self.down1_fusion(x1_fusion)
        x2_fusion = self.fusion2(x2_fusion, x2_dwi, x2_adc, x2_flair)
        x3_fusion = self.down2_fusion(x2_fusion)
        x3_fusion = self.fusion3(x3_fusion, x3_dwi, x3_adc, x3_flair)
        x4_fusion = self.down3_fusion(x3_fusion)
        x4_fusion = self.fusion4(x4_fusion, x4_dwi, x4_adc, x4_flair)
        x5_fusion = self.down4_fusion(x4_fusion)

        x4_concat = torch.concat((x4_fusion, x4_dwi, x4_adc, x4_flair), 1)
        x3_concat = torch.concat((x3_fusion, x3_dwi, x3_adc, x3_flair), 1)
        x2_concat = torch.concat((x2_fusion, x2_dwi, x2_adc, x2_flair), 1)
        x1_concat = torch.concat((x1_fusion, x1_dwi, x1_adc, x1_flair), 1)

        x = self.up1_fusion(x5_fusion, x4_concat)
        x = self.up2_fusion(x, x3_concat)
        x = self.up3_fusion(x, x2_concat)
        x = self.up4_fusion(x, x1_concat)
        logits_fusion = self.outc_fusion(x)

        return logits_fusion, logits_dwi, logits_adc, logits_flair

class Entropy_Hist(nn.Module):
    def __init__(self, ratio, win_w=3, win_h=3, win_y=3):
        super(Entropy_Hist, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.win_y = win_y
        self.ratio = ratio

    def calcIJ_new(self, img_patch):
        total_p = img_patch.shape[-1] * img_patch.shape[-2] * img_patch.shape[-3]
        if total_p % 2 != 0:
            tem = torch.flatten(img_patch, start_dim=-3, end_dim=-1)
            center_p = tem[:, :, :, int(total_p / 2)]
            mean_p = (torch.sum(tem, dim=-1) - center_p) / (total_p - 1)
            if torch.is_tensor(img_patch):
                return center_p * 100 + mean_p
            else:
                return (center_p, mean_p)
        else:
            print("modify patch size")

    def histc_fork(self, ij):
        BINS = 256
        B, C = ij.shape
        N = 16
        BB = B // N
        min_elem = ij.min()
        max_elem = ij.max()
        ij = ij.view(N, BB, C)

        def f(x):
            with torch.no_grad():
                res = []
                for e in x:
                    res.append(torch.histc(e, bins=BINS, min=min_elem, max=max_elem))
                return res

        futures: List[torch.jit.Future[torch.Tensor]] = []

        for i in range(N):
            futures.append(torch.jit.fork(f, ij[i]))

        results = []
        for future in futures:
            results += torch.jit.wait(future)
        with torch.no_grad():
            out = torch.stack(results)
        return out

    def forward(self, img):
        with torch.no_grad():
            B, C, H, W, Z = img.shape
            ext_x = int(self.win_w / 2)  # 考虑滑动窗口大小，对原图进行扩边，扩展部分长度
            ext_y = int(self.win_h / 2)
            ext_z = int(self.win_y / 2)

            new_width = ext_x + W + ext_x  # 新的图像尺寸
            new_height = ext_y + H + ext_y
            new_depth = ext_z + Z + ext_z
            # 使用nn.Unfold依次获取每个滑动窗口的内容
            # nn_Unfold = nn.Unfold(kernel_size=(self.win_w, self.win_h, self.win_y), dilation=1, padding=ext_x, stride=1)
            # 能够获取到patch_img，shape=(B,C*K*K,L),L代表的是将每张图片由滑动窗口分割成多少块---->28*28的图像，3*3的滑动窗口，分成了28*28=784块
            # x = nn_Unfold(img) # (B,C*K*K,L)
            x = img.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1)
            x = x.permute(0, 1, 5, 6, 7, 2, 3, 4).reshape(B, C, 3, 3, 3, -1).permute(0, 1, 5, 2, 3, 4)  # (B,C*K*K,L) ---> (B,C,L,K,K)
            ij = self.calcIJ_new(x).reshape(B * C, -1)  # 计算滑动窗口内中心的灰度值和窗口内除了中心像素的灰度均值,(B,C,L,K,K)---> (B,C,L) ---> (B*C,L)

            fij_packed = self.histc_fork(ij)
            p = fij_packed / (new_width * new_height * new_depth)
            h_tem = -p * torch.log(torch.clamp(p, min=1e-40)) / math.log(2)

            a = torch.sum(h_tem, dim=1)  # 对所有二维熵求和，得到这张图的二维熵
            H = a.reshape(B, C)

            _, index = torch.topk(H, int(self.ratio * C), dim=1)  # Nx3
        selected = []
        for i in range(img.shape[0]):
            selected.append(torch.index_select(img[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)

        return selected

class EnhancedFeature(nn.Module):
    def __init__(self, in_chan, is_first=False):
        super(EnhancedFeature, self).__init__()
        self.inchan = in_chan
        self.is_first = is_first
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=4 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=2 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.non_local_channel = _NonLocalBlockND_channel(in_chan, inter_channels=None, dimension=3, sub_sample=False, bn_layer=True)

        self.non_local_spitial = _NonLocalBlockND_spitail(in_chan, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True)


    def forward(self, x0, x1, x2, x3):
        w = torch.sigmoid(self.conv1(torch.cat((x1, x2, x3), dim=1)))
        # w2 = self.non_local_channel(w)
        # w2_ = self.non_local_spitial(w)
        # w_all = w2 + w2_
        feat_x1 = torch.mul(x1, w)
        feat_x2 = torch.mul(x2, w)
        feat_x3 = torch.mul(x3, w)
        x = self.conv3(torch.cat((self.conv2(feat_x1 + feat_x2 + feat_x3), x1, x2, x3), dim=1))
        if not self.is_first:
            x = self.conv(torch.cat((x0, x), dim=1))
        return x


class EnhancedFeature_nonlocal(nn.Module):
    def __init__(self, in_chan, is_first=False):
        super(EnhancedFeature_nonlocal, self).__init__()
        self.inchan = in_chan
        self.is_first = is_first
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=4 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=2 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.non_local_channel = _NonLocalBlockND_channel(3 * in_chan, inter_channels=None, dimension=3, sub_sample=False, bn_layer=True)

        self.non_local_spitial = _NonLocalBlockND_spitail(in_chan, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True)


    def forward(self, x0, x1, x2, x3):
        w = torch.sigmoid(self.conv1(self.non_local_channel(torch.cat((x1, x2, x3), dim=1))))
        # w = self.conv1(torch.cat((x1, x2, x3), dim=1))
        # w2 = self.non_local_channel(w)
        # w2_ = self.non_local_spitial(w)
        # w_all = w2 + w2_
        feat_x1 = torch.mul(x1, w)
        feat_x2 = torch.mul(x2, w)
        feat_x3 = torch.mul(x3, w)
        x = self.conv3(torch.cat((self.conv2(feat_x1 + feat_x2 + feat_x3), x1, x2, x3), dim=1))
        if not self.is_first:
            x = self.conv(torch.cat((x0, x), dim=1))
        return x


class EnhancedFeature_nonlocal_IFE(nn.Module):
    def __init__(self, in_chan, is_first=False):
        super(EnhancedFeature_nonlocal_IFE, self).__init__()
        self.inchan = in_chan
        self.is_first = is_first
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=4 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=2 * (in_chan//3) + in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.non_local_channel = _NonLocalBlockND_channel(3 * in_chan, inter_channels=None, dimension=3, sub_sample=False, bn_layer=True)
        self.non_local_spitial = _NonLocalBlockND_spitail(in_chan, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True)


    def forward(self, x0, x1, x2, x3):
        w = torch.sigmoid(self.conv1(self.non_local_channel(torch.cat((x1, x2, x3), dim=1))))
        # w = self.conv1(torch.cat((x1, x2, x3), dim=1))
        # w2 = self.non_local_channel(w)
        # w2_ = self.non_local_spitial(w)
        # w_all = w2 + w2_
        feat_x1 = torch.mul(x1, w)
        feat_x2 = torch.mul(x2, w)
        feat_x3 = torch.mul(x3, w)
        x = self.conv3(torch.cat((self.conv2(feat_x1 + feat_x2 + feat_x3), x1, x2, x3), dim=1))
        if not self.is_first:
            x = self.conv(torch.cat((x0, x), dim=1))
        return x


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
