"""OceanEmbedNet: a U-Net that turns surface fields into a subsurface temperature
volume, with an explicit latent bottleneck that serves as the satellite embedding.

  surface (10, H, W)
      -> encoder (3 downsampling stages)
      -> bottleneck  == satellite embedding (256 x H/8 x W/8)
      -> decoder (3 upsampling stages, U-Net skips)
      -> temperature (15, H, W)

Fully convolutional: trained on 64x64 patches, run on the full 101x241 grid.
`encode()` exposes the bottleneck so the embedding can be visualized.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )


class OceanEmbedNet(nn.Module):
    def __init__(self, in_ch: int = 10, out_ch: int = 15, base: int = 32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base * 4, base * 8)  # embedding = base*8 channels

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    @staticmethod
    def _pad(x: torch.Tensor, mult: int = 8):
        _, _, h, w = x.shape
        ph, pw = (mult - h % mult) % mult, (mult - w % mult) % mult
        return F.pad(x, (0, pw, 0, ph), mode="reflect"), (h, w)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._pad(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        return self.bottleneck(self.pool(e3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (h, w) = self._pad(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        out = self.head(d1)
        return out[:, :, :h, :w]
