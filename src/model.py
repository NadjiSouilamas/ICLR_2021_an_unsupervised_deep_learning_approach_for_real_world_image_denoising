# model.py
"""
    UNet model version 1
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class ConvBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, input_channel_size, output_channel_size):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(

            nn.Conv2d(input_channel_size, output_channel_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel_size, output_channel_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channel_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, input_channel_size, output_channel_size):
        
        super(UpConv, self).__init__()

        self.up_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_channel_size, output_channel_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channel_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up_block(x)


class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, input_channel_size=3, output_channel_size=3):
        super(UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(input_channel_size, filters[0])
        self.conv2 = ConvBlock(filters[0], filters[1])
        self.conv3 = ConvBlock(filters[1], filters[2])
        self.conv4 = ConvBlock(filters[2], filters[3])
        self.conv5 = ConvBlock(filters[3], filters[4])

        self.up5 = UpConv(filters[4], filters[3])
        self.upConv5 = ConvBlock(filters[4], filters[3])

        self.up4 = UpConv(filters[3], filters[2])
        self.upConv4 = ConvBlock(filters[3], filters[2])

        self.up3 = UpConv(filters[2], filters[1])
        self.upConv3 = ConvBlock(filters[2], filters[1])

        self.up2 = UpConv(filters[1], filters[0])
        self.upConv2 = ConvBlock(filters[1], filters[0])

        self.conv = nn.Conv2d(filters[0], output_channel_size, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.pool1(e1)
        e2 = self.conv2(e2)

        e3 = self.pool2(e2)
        e3 = self.conv3(e3)

        e4 = self.pool3(e3)
        e4 = self.conv4(e4)

        e5 = self.pool4(e4)
        e5 = self.conv5(e5)

        d5 = self.up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d2 = self.conv(d2)

        out = self.sigmoid(d2)

        return out