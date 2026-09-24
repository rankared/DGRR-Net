"""
Proposed E2c dual-scale residual U-Net.

These model definitions are extracted from the successfully executed
Phase-6 E2c authoritative source cell (code-cell 63). They retain the
exact module names and tensor operations required for strict loading of
the frozen manuscript checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_norm(channels):

    return nn.GroupNorm(
        num_groups=8,
        num_channels=channels,
        affine=True,
    )


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm1 = make_norm(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm2 = make_norm(
            out_channels
        )

        self.activation = nn.SiLU(
            inplace=False
        )

        self.dropout = nn.Dropout2d(
            dropout
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

        output = self.conv1(x)
        output = self.norm1(output)
        output = self.activation(
            output
        )
        output = self.dropout(
            output
        )

        output = self.conv2(
            output
        )
        output = self.norm2(
            output
        )

        output = (
            output + identity
        )

        output = self.activation(
            output
        )

        return output


class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
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

        self.block = ResidualBlock(
            out_channels,
            out_channels,
            dropout,
        )

    def forward(self, x):

        return self.block(
            self.downsample(x)
        )


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        dropout,
    ):
        super().__init__()

        self.reduce = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.block = ResidualBlock(
            out_channels
            + skip_channels,
            out_channels,
            dropout,
        )

    def forward(
        self,
        x,
        skip,
    ):

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = self.reduce(x)

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.block(x)


class E2cDualScaleResidualUNet(nn.Module):

    def __init__(
        self,
        input_channels=11,
        base_channels=32,
        dropout=0.05,
    ):
        super().__init__()

        base = base_channels

        self.encoder1 = ResidualBlock(
            input_channels,
            base,
            dropout,
        )

        self.encoder2 = DownBlock(
            base,
            base * 2,
            dropout,
        )

        self.encoder3 = DownBlock(
            base * 2,
            base * 4,
            dropout,
        )

        self.encoder4 = DownBlock(
            base * 4,
            base * 8,
            dropout,
        )

        self.bottleneck = ResidualBlock(
            base * 8,
            base * 8,
            dropout,
        )

        self.decoder3 = UpBlock(
            base * 8,
            base * 4,
            base * 4,
            dropout,
        )

        self.decoder2 = UpBlock(
            base * 4,
            base * 2,
            base * 2,
            dropout,
        )

        self.decoder1 = UpBlock(
            base * 2,
            base,
            base,
            dropout,
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(
                base,
                base,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.SiLU(
                inplace=False
            ),
            nn.Conv2d(
                base,
                1,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, x):

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        bottleneck = self.bottleneck(
            e4
        )

        d3 = self.decoder3(
            bottleneck,
            e3,
        )

        d2 = self.decoder2(
            d3,
            e2,
        )

        d1 = self.decoder1(
            d2,
            e1,
        )

        return self.output_head(
            d1
        )
