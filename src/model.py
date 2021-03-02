import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class ConvBlock(nn.Module):
    """
    Standard 2-layer convolution block 
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
    Upsampling block with convolutions
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

    def __init__(self, input_channel_size=3, output_channel_size=3):
        super(UNet, self).__init__()

        nb_filters = 64

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(input_channel_size, nb_filters)
        self.conv2 = ConvBlock(nb_filters, 2 * nb_filters)
        self.conv3 = ConvBlock(2 * nb_filters, 4 * nb_filters)
        self.conv4 = ConvBlock(4 * nb_filters, 8 * nb_filters)
        self.conv5 = ConvBlock(8 * nb_filters, 16 * nb_filters)

        self.up5 = UpConv(16 * nb_filters, 8 * nb_filters)
        self.up_conv5 = ConvBlock(16 * nb_filters, 8 * nb_filters)

        self.up4 = UpConv(8 * nb_filters, 4 * nb_filters)
        self.up_conv4 = ConvBlock(8 * nb_filters, 4 * nb_filters)

        self.up3 = UpConv(4 * nb_filters, 2 * nb_filters)
        self.up_conv3 = ConvBlock(4 * nb_filters, 2 * nb_filters)

        self.up2 = UpConv(2 * nb_filters, nb_filters)
        self.up_conv2 = ConvBlock(2 * nb_filters, nb_filters)

        self.conv = nn.Conv2d(nb_filters, output_channel_size, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        e1 = self.conv1(x)

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