from src.blocks import (
    DoubleConv,
    Down,
    Up,
    OutConv,
)
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        f = [64, 128, 256, 512, 1024]
        # encoder
        self.inc = DoubleConv(n_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        self.down4 = Down(f[3], f[4])

        # decoder
        self.up1 = Up(f[4], f[3])
        self.up2 = Up(f[3], f[2])
        self.up3 = Up(f[2], f[1])
        self.up4 = Up(f[1], f[0])

        # output
        self.outc = OutConv(f[0], n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder
        d1 = self.up1(x5, x4)  # x4, x5, ... are the skip connection
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        # output
        out = self.outc(d4)
        return out