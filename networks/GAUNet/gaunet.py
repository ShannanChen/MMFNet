import torch.nn as nn
import torch.nn.functional as F
import math
import torch

# adapt from https://github.com/MIC-DKFZ/BraTS2017
try:
    from .sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass


class sort(nn.Module):
    def __init__(self, channel,norm='sync_bn'):
        super(sort, self).__init__()
        self.conv3x3 = nn.Conv3d(channel, channel,3, 1, 1, bias=False)
        self.bn = normalization(channel, norm)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv3x3(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x2 = self.conv3x3(x1)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        out = torch.mul(x1,x2)

        out = torch.add(out, x)

        return out


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))

        y = self.bn3(self.conv3(x))
        return self.relu(y)


class ConvU(nn.Module):
    def __init__(self, planes, norm='gn', first=False):
        super(ConvU, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 1, bias=False)
            self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes//2, 1, 1, 0, bias=False)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))

        y = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)
        y = self.relu(self.bn2(self.conv2(y)))

        y = torch.cat([prev, y], 1)
        y = self.relu(self.bn3(self.conv3(y)))

        return y





class sounet(nn.Module):
    def __init__(self, c=4, n=32, dropout=0.5, norm='gn', num_classes=5):
        super(sounet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=False)

        self.convd1 = ConvD(c,     n, dropout, norm, first=True)
        self.convd2 = ConvD(n,   2*n, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, dropout, norm)

        self.nl4= sort(8*n)
        self.convu4 = ConvU(16*n, norm, True)
        self.nl3 = sort(4*n)
        self.convu3 = ConvU(8*n, norm)
        self.nl2 = sort(2*n)
        self.convu2 = ConvU(4*n, norm)
        self.nl1= sort(n)
        self.convu1 = ConvU(2*n, norm)


        self.seg3 = nn.Conv3d(8*n, num_classes, 1)
        self.seg2 = nn.Conv3d(4*n, num_classes, 1)
        self.seg1 = nn.Conv3d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        # x4 = self.nl4(x4)
        y4 = self.convu4(x5, x4)
        # x3 = self.nl3(x3)
        y3 = self.convu3(y4, x3)
        # x2 = self.nl2(x2)
        y2 = self.convu2(y3, x2)
        # x1 = self.nl1(x1)
        y1 = self.convu1(y2, x1)


        y3 = self.seg3(y3)
        y2 = self.seg2(y2) + self.upsample(y3)
        y1 = self.seg1(y1) + self.upsample(y2)

        return y1


