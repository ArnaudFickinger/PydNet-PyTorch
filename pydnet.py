from __future__ import absolute_import, division, print_function
from collections import namedtuple
import torch.nn as nn
import torch

class Pydnet(nn.Module):
    def __init__(self, conv_dim=64):
        super(Pydnet, self).__init__()

        # Features extractor
        self.conv_ext_1 = self.conv_down_block(3,16)
        self.conv_ext_2 = self.conv_down_block(16, 32)
        self.conv_ext_3 = self.conv_down_block(32, 64)
        self.conv_ext_4 = self.conv_down_block(64, 96)
        self.conv_ext_5 = self.conv_down_block(96, 128)
        self.conv_ext_6 = self.conv_down_block(128, 192)

        # Depth Decoder
        self.conv_dec_1 = self.conv_disp_block(16)
        self.conv_dec_2 = self.conv_disp_block(32)
        self.conv_dec_3 = self.conv_disp_block(64)
        self.conv_dec_4 = self.conv_disp_block(96)
        self.conv_dec_5 = self.conv_disp_block(128)
        self.conv_dec_6 = self.conv_disp_block(192)
        self.disp = nn.Sigmoid()

        # Upsampling
        deconv1 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2)

    def conv_down_block(self, in_channels, out_channels):
        conv_down_block = []
        conv_down_block += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        conv_down_block += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()]
        return nn.Sequential(*conv_down_block)


    def conv_disp_block(self, in_channels):
        conv_disp_block = []
        conv_disp_block += [nn.Conv2d(in_channels = in_channels, out_channels= 96 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU()]
        conv_disp_block += [nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding = 1), nn.LeakyReLU()]
        conv_disp_block += [nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()]
        conv_disp_block += [nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*conv_disp_block)


    def forward(self, x):

        conv1 = self.conv_ext_1(x)
        conv2 = self.conv_ext_2(conv1)
        conv3 = self.conv_ext_3(conv2)
        conv4 = self.conv_ext_4(conv3)
        conv5 = self.conv_ext_5(conv4)
        conv6 = self.conv_ext_6(conv5)

        conv6b = self.conv_dec_6(conv6)
        disp6 = self.disp(conv6b)
        concat5 = torch.cat((conv5, conv6b))
        conv5b = self.conv_dec_5(concat5)
        disp5 = self.disp(conv5b)
        concat4 = torch.cat((conv4, conv5b))
        conv4b = self.conv_dec_4(concat4)
        disp4 = self.disp(conv4b)
        concat3 = torch.cat((conv3, conv4b))
        conv3b = self.conv_dec_3(concat3)
        disp3 = self.disp(conv3b)
        concat2 = torch.cat((conv2, conv3b))
        conv2b = self.conv_dec_2(concat2)
        disp2 = self.disp(conv2b)
        concat1 = torch.cat((conv1, conv2b))
        conv1b = self.conv_dec_1(concat1)
        disp1 = self.disp(conv1b)

        return [disp1, disp2, disp3, disp4, disp5, disp6]