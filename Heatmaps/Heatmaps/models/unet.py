import math
import torch
from torch import nn
from torch.nn import functional as F

MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30
from .common import OutputActivationMixin, build_normalisation, build_activation, build_channels, validate_unet_args


class ConvBlock(nn.Module):
    """Run two convolutional layers at one U-Net level."""

    def __init__(
        self, in_channels, out_channels, normalisation='batch', activation='relu', dropout=0.0, padding_mode='zeros'
    ):
        super().__init__()
        use_bias = normalisation is None or str(normalisation).lower() == 'none'
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode, bias=use_bias),
            build_normalisation(normalisation, out_channels),
            build_activation(activation),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode, bias=use_bias),
            build_normalisation(normalisation, out_channels),
            build_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample once, then apply a convolution block."""

    def __init__(
        self, in_channels, out_channels, normalisation='batch', activation='relu', dropout=0.0, padding_mode='zeros'
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels, normalisation, activation, dropout, padding_mode),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample once, concatenate the skip connection, then apply a convolution block."""

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        upsampling='bilinear',
        normalisation='batch',
        activation='relu',
        dropout=0.0,
        padding_mode='zeros',
    ):
        super().__init__()
        upsampling = str(upsampling).lower()

        if upsampling == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        elif upsampling == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            raise ValueError(f'Unknown upsampling: {upsampling}')

        self.conv = ConvBlock(
            in_channels + skip_channels, out_channels, normalisation, activation, dropout, padding_mode
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        return self.conv(torch.cat((skip, x), dim=1))


class UNetHeatmap(OutputActivationMixin, nn.Module):
    """U-Net heatmap regressor with one output channel per landmark."""

    def __init__(
        self,
        num_of_points,
        input_channels=1,
        base_channels=32,
        depth=4,
        channel_multiplier=2,
        max_channels=512,
        normalisation='batch',
        activation='relu',
        dropout=0.0,
        upsampling='bilinear',
        output_activation='none',
        padding_mode='zeros',
        final_kernel_size=1,
    ):
        super().__init__()
        validate_unet_args(
            num_of_points=num_of_points,
            input_channels=input_channels,
            base_channels=base_channels,
            depth=depth,
            channel_multiplier=channel_multiplier,
            max_channels=max_channels,
            dropout=dropout,
            final_kernel_size=final_kernel_size,
        )
        self.num_of_points = int(num_of_points)
        self.input_channels = int(input_channels)
        self.configure_output_activation(output_activation)
        channels = build_channels(base_channels, depth, channel_multiplier, max_channels)
        self.input_block = ConvBlock(input_channels, channels[0], normalisation, activation, dropout, padding_mode)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(channels[index], channels[index + 1], normalisation, activation, dropout, padding_mode)
                for index in range(depth)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(
                    channels[index + 1],
                    channels[index],
                    channels[index],
                    upsampling,
                    normalisation,
                    activation,
                    dropout,
                    padding_mode,
                )
                for index in range(depth - 1, -1, -1)
            ]
        )
        self.output_layer = nn.Conv2d(
            channels[0], num_of_points, kernel_size=final_kernel_size, padding=final_kernel_size // 2
        )

    def forward(self, x):
        skips = []
        x = self.input_block(x)
        skips.append(x)

        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)

        x = skips.pop()

        for up_block in self.up_blocks:
            x = up_block(x, skips.pop())

        return self.apply_output_activation(self.output_layer(x))
