"""
E2a global-only residual U-Net ablation.

The architecture is retained from the Phase-6 E2a source used in the
manuscript comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_group_norm(num_channels):
    # Use the largest valid group count up to 8.
    for groups in [8, 4, 2, 1]:
        if num_channels % groups == 0:
            return nn.GroupNorm(
                num_groups=groups,
                num_channels=num_channels
            )

    raise ValueError(
        f"Could not construct GroupNorm for {num_channels} channels."
    )


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = make_group_norm(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = make_group_norm(out_channels)

        self.activation = nn.SiLU(inplace=True)

        self.dropout = (
            nn.Dropout2d(dropout)
            if dropout > 0
            else nn.Identity()
        )

        self.skip = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.activation(out)

        return out


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.0,
    ):
        super().__init__()

        self.downsample = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.block = ResidualConvBlock(
            out_channels,
            out_channels,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        dropout=0.0,
    ):
        super().__init__()

        self.reduce = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.block = ResidualConvBlock(
            out_channels + skip_channels,
            out_channels,
            dropout=dropout,
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)

        return x


class E2aPlainResidualUNet(nn.Module):
    def __init__(
        self,
        in_channels=10,
        base_channels=32,
        dropout=0.05,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.encoder1 = ResidualConvBlock(
            in_channels,
            c1,
            dropout=0.0,
        )

        self.encoder2 = DownBlock(
            c1,
            c2,
            dropout=dropout,
        )

        self.encoder3 = DownBlock(
            c2,
            c3,
            dropout=dropout,
        )

        self.encoder4 = DownBlock(
            c3,
            c4,
            dropout=dropout,
        )

        self.bottleneck = ResidualConvBlock(
            c4,
            c4,
            dropout=dropout,
        )

        self.decoder3 = UpBlock(
            in_channels=c4,
            skip_channels=c3,
            out_channels=c3,
            dropout=dropout,
        )

        self.decoder2 = UpBlock(
            in_channels=c3,
            skip_channels=c2,
            out_channels=c2,
            dropout=dropout,
        )

        self.decoder1 = UpBlock(
            in_channels=c2,
            skip_channels=c1,
            out_channels=c1,
            dropout=0.0,
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(
                c1,
                c1,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                c1,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        e1 = self.encoder1(x)    # 256 × 256
        e2 = self.encoder2(e1)   # 128 × 128
        e3 = self.encoder3(e2)   # 64 × 64
        e4 = self.encoder4(e3)   # 32 × 32

        bottleneck = self.bottleneck(e4)

        d3 = self.decoder3(bottleneck, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        residual_normalized = self.output_head(d1)

        return residual_normalized
