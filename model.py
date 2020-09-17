import torch.nn as nn
import torch

from block import ConvDownBlock

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, spec_norm=True):
        super(Discriminator, self).__init__()
        self.main = list()
        self.main.append(ConvDownBlock(3, 16, spec_norm, stride=2)) # 256 -> 128
        self.main.append(ConvDownBlock(16, 32, spec_norm, stride=2)) # 128 -> 64
        self.main.append(ConvDownBlock(32, 64, spec_norm, stride=2)) # 64 -> 32
        self.main.append(ConvDownBlock(64, 128, spec_norm, stride=2)) # 32 -> 16
        self.main.append(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)

class ResBlockNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlockNet, self).__init__()
        self.main = list()
        self.main.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.main.append(nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True))
        self.main.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)

class Encoder(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, in_channels=3, spec_norm=False, LR=0.2):
        super(Encoder, self).__init__()

        self.layer1 = ConvDownBlock(in_channels, 16, spec_norm, LR=LR)
        self.layer2 = ConvDownBlock(16, 16, spec_norm, LR=LR)
        self.layer3 = ConvDownBlock(16, 32, spec_norm, stride=2, LR=LR)
        self.layer4 = ConvDownBlock(32, 32, spec_norm, LR=LR)
        self.layer5 = ConvDownBlock(32, 64, spec_norm, stride=2, LR=LR)
        self.layer6 = ConvDownBlock(64, 64, spec_norm, LR=LR)
        self.layer7 = ConvDownBlock(64, 128, spec_norm, stride=2, LR=LR)
        self.layer8 = ConvDownBlock(128, 128, spec_norm, LR=LR)
        self.layer9 = ConvDownBlock(128, 256, spec_norm, stride=2, LR=LR)
        self.layer10 = ConvDownBlock(128, 256, spec_norm, LR=LR)


    def forward(self, x):

        feature_map_list = list()
        feature_map1 = self.layer1(x)
        feature_map2 = self.layer2(feature_map1)
        feature_map3 = self.layer3(feature_map2)
        feature_map4 = self.layer4(feature_map3)
        feature_map5 = self.layer5(feature_map4)
        feature_map6 = self.layer6(feature_map5)
        feature_map7 = self.layer7(feature_map6)
        feature_map8 = self.layer8(feature_map7)
        feature_map9 = self.layer9(feature_map8)
        feature_map10 = self.layer10(feature_map9)

        feature_map_list.append(feature_map1)
        feature_map_list.append(feature_map2)
        feature_map_list.append(feature_map3)
        feature_map_list.append(feature_map4)
        feature_map_list.append(feature_map5)
        feature_map_list.append(feature_map6)
        feature_map_list.append(feature_map7)
        feature_map_list.append(feature_map8)
        feature_map_list.append(feature_map9)
        feature_map_list.append(feature_map10)

        return feature_map_list

