import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [Bn] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),  # Downsample by factor of 2
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: tensor from the previous layer
        x2: tensor from the encoder layer (skip connection)
        """
        x1 = self.up(x1)

        target_h, target_w = x1.shape[2], x1.shape[3]
        skip_h, skip_w = x2.shape[2], x2.shape[3]

        # Crop step to ensure that x1 and x2 have the same shape
        if skip_h > target_h or skip_w > target_w:
            diff_y = skip_h - target_h
            diff_x = skip_w - target_w

            crop_top = diff_y // 2
            crop_bottom = diff_y - crop_top
            crop_left = diff_x // 2
            crop_right = diff_x - crop_left

            if crop_bottom < 0:
                crop_bottom = 0
            if crop_right < 0:
                crop_right = 0

            x2 = x2[
                :, :, crop_top : skip_h - crop_bottom, crop_left : skip_w - crop_right
            ]

        x = torch.cat([x1, x2], dim=1)  # Concatenate along the channel dimension
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
