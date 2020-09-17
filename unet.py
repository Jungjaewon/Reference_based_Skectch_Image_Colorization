""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_part import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        factor = 2 if bilinear else 1
        self.down4 = Down(in_channels=512, out_channels=1024 // factor)
        self.up1 = Up(in_channels=1024, out_channels=512 // factor, bilinear=bilinear)
        self.up2 = Up(in_channels=512, out_channels=256 // factor, bilinear=bilinear)
        self.up3 = Up(in_channels=256, out_channels=128 // factor, bilinear=bilinear)
        self.up4 = Up(in_channels=128, out_channels=64, bilinear=bilinear)
        self.outc = OutConv(in_channels=64, out_channels=n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits