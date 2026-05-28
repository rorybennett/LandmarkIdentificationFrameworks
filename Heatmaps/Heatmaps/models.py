"""
Configurable heatmap-regression models for landmark localisation.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Run two convolutional layers at one U-Net level."""

    def __init__(self, in_channels, out_channels, normalisation='batch', activation='relu', dropout=0.0, padding_mode='zeros'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode, bias=normalisation is None),
            build_normalisation(normalisation, out_channels),
            build_activation(activation),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode, bias=normalisation is None),
            build_normalisation(normalisation, out_channels),
            build_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample once, then apply a convolution block."""

    def __init__(self, in_channels, out_channels, normalisation='batch', activation='relu', dropout=0.0, padding_mode='zeros'):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), ConvBlock(in_channels, out_channels, normalisation, activation, dropout, padding_mode))

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample once, concatenate the skip connection, then apply a convolution block."""

    def __init__(self, in_channels, skip_channels, out_channels, upsampling='bilinear', normalisation='batch', activation='relu', dropout=0.0, padding_mode='zeros'):
        super().__init__()
        upsampling = str(upsampling).lower()

        if upsampling == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif upsampling == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            raise ValueError(f'Unknown upsampling: {upsampling}')

        self.conv = ConvBlock(in_channels + skip_channels, out_channels, normalisation, activation, dropout, padding_mode)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        return self.conv(torch.cat((skip, x), dim=1))


class UNetHeatmap(nn.Module):
    """U-Net heatmap regressor with one output channel per landmark."""

    def __init__(self, num_of_points, input_channels=1, base_channels=32, depth=4, channel_multiplier=2, max_channels=512, normalisation='batch', activation='relu', dropout=0.0, upsampling='bilinear', output_activation='none', padding_mode='zeros', final_kernel_size=1):
        super().__init__()
        validate_unet_args(num_of_points, input_channels, base_channels, depth, channel_multiplier, final_kernel_size)
        self.num_of_points = int(num_of_points)
        self.input_channels = int(input_channels)
        self.output_activation = None if str(output_activation).lower() in ('none', 'identity', '') else str(output_activation).lower()
        channels = build_channels(base_channels, depth, channel_multiplier, max_channels)
        self.input_block = ConvBlock(input_channels, channels[0], normalisation, activation, dropout, padding_mode)
        self.down_blocks = nn.ModuleList([DownBlock(channels[index], channels[index + 1], normalisation, activation, dropout, padding_mode) for index in range(depth)])
        self.up_blocks = nn.ModuleList([UpBlock(channels[index + 1], channels[index], channels[index], upsampling, normalisation, activation, dropout, padding_mode) for index in range(depth - 1, -1, -1)])
        self.output_layer = nn.Conv2d(channels[0], num_of_points, kernel_size=final_kernel_size, padding=final_kernel_size // 2)

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

        x = self.output_layer(x)
        return self.apply_output_activation(x)

    def apply_output_activation(self, x):
        """Apply the configured output activation."""
        if self.output_activation is None:
            return x

        if self.output_activation == 'sigmoid':
            return torch.sigmoid(x)

        if self.output_activation == 'softplus':
            return F.softplus(x)

        raise ValueError(f'Unknown output_activation: {self.output_activation}')


def build_normalisation(normalisation, channels):
    """Create a normalisation layer."""
    if normalisation is None or str(normalisation).lower() in ('none', 'identity'):
        return nn.Identity()

    normalisation = str(normalisation).lower()

    if normalisation == 'batch':
        return nn.BatchNorm2d(channels)

    if normalisation == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)

    if normalisation == 'group':
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)

    raise ValueError(f'Unknown normalisation: {normalisation}')


def build_activation(activation):
    """Create an activation layer."""
    activation = str(activation).lower()

    if activation == 'relu':
        return nn.ReLU(inplace=True)

    if activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)

    if activation == 'elu':
        return nn.ELU(inplace=True)

    if activation == 'gelu':
        return nn.GELU()

    raise ValueError(f'Unknown activation: {activation}')


def build_channels(base_channels, depth, channel_multiplier, max_channels):
    """Create encoder channel widths."""
    return [min(int(base_channels) * (int(channel_multiplier) ** index), int(max_channels)) for index in range(int(depth) + 1)]


def validate_unet_args(num_of_points, input_channels, base_channels, depth, channel_multiplier, final_kernel_size):
    """Validate U-Net construction values."""
    if int(num_of_points) < 1:
        raise ValueError('num_of_points must be at least 1.')

    if int(input_channels) < 1:
        raise ValueError('input_channels must be at least 1.')

    if int(base_channels) < 1:
        raise ValueError('base_channels must be at least 1.')

    if int(depth) < 1:
        raise ValueError('depth must be at least 1.')

    if int(channel_multiplier) < 1:
        raise ValueError('channel_multiplier must be at least 1.')

    if int(final_kernel_size) not in (1, 3):
        raise ValueError('final_kernel_size must be 1 or 3.')


def count_trainable_parameters(model):
    """Return trainable parameter count."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
