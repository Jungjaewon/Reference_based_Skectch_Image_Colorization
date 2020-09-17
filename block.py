import torch
import torch.nn as nn

from spectral_normalization import SpectralNorm


class ConvDownBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spec_norm=False, LR=0.01, stride=1):
        super(ConvDownBlock, self).__init__()

        if spec_norm:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )
        else:
            self.main = nn.Sequential(
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )

    def forward(self, x):
        return self.main(x)

class ConvUpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spec_norm=False, LR=0.01):
        super(ConvUpBlock, self).__init__()

        if spec_norm:
            self.main = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )
        else:
            self.main = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )

    def forward(self, x):
        return self.main(x)