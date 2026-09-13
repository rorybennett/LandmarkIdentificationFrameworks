"""Shared building blocks and validation for heatmap models."""

import math
import torch
from torch import nn
from torch.nn import functional as F

MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30


class OutputActivationMixin:
    """Apply the configured output activation to heatmap tensors."""

    def configure_output_activation(self, output_activation):
        self.output_activation = (
            None
            if output_activation is None or str(output_activation).lower() == 'none'
            else str(output_activation).lower()
        )

    def apply_output_activation(self, x):
        if self.output_activation is None:
            return x

        if self.output_activation == 'sigmoid':
            return torch.sigmoid(x)

        if self.output_activation == 'softplus':
            return F.softplus(x)

        raise ValueError(f'Unknown output_activation: {self.output_activation}')


class ResidualBlock(nn.Module):
    """Residual convolution block shared by HRNet and hourglass models."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        normalisation='batch',
        activation='relu',
        dropout=0.0,
        padding_mode='zeros',
    ):
        super().__init__()
        use_bias = normalisation is None or str(normalisation).lower() == 'none'
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode=padding_mode, bias=use_bias
        )
        self.norm1 = build_normalisation(normalisation, out_channels)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode, bias=use_bias
        )
        self.norm2 = build_normalisation(normalisation, out_channels)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=use_bias),
                build_normalisation(normalisation, out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


def make_residual_sequence(in_channels, out_channels, blocks, normalisation, activation, dropout, padding_mode):
    """Create a sequence of residual blocks."""
    layers = [
        ResidualBlock(
            in_channels,
            out_channels,
            normalisation=normalisation,
            activation=activation,
            dropout=dropout,
            padding_mode=padding_mode,
        )
    ]
    layers.extend(
        [
            ResidualBlock(
                out_channels,
                out_channels,
                normalisation=normalisation,
                activation=activation,
                dropout=dropout,
                padding_mode=padding_mode,
            )
            for _ in range(int(blocks) - 1)
        ]
    )
    return nn.Sequential(*layers)


def unpack_heatmap_output(model_output):
    """Return the final heatmaps and optional auxiliary heatmaps from a model output."""
    if torch.is_tensor(model_output):
        return model_output, []

    if isinstance(model_output, dict) and torch.is_tensor(model_output.get('heatmaps')):
        auxiliary = model_output.get('auxiliary_heatmaps', [])

        if not isinstance(auxiliary, (list, tuple)) or not all(torch.is_tensor(item) for item in auxiliary):
            raise TypeError('auxiliary_heatmaps must be a list or tuple of tensors.')

        return model_output['heatmaps'], list(auxiliary)

    raise TypeError(
        'Heatmap models must return a tensor or a dictionary containing heatmaps and optional auxiliary_heatmaps.'
    )


def build_normalisation(normalisation, channels):
    """Create a normalisation layer."""
    if normalisation is None or str(normalisation).lower() == 'none':
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
    return [
        min(int(base_channels) * (int(channel_multiplier) ** index), int(max_channels))
        for index in range(int(depth) + 1)
    ]


def validate_common_model_args(num_of_points, input_channels, dropout, final_kernel_size):
    """Validate values shared by heatmap models."""
    if int(num_of_points) < MIN_POINTS_PER_IMAGE or int(num_of_points) > MAX_POINTS_PER_IMAGE:
        raise ValueError(
            f'num_of_points must be between {MIN_POINTS_PER_IMAGE} and {MAX_POINTS_PER_IMAGE}. Got: {num_of_points}'
        )

    if int(input_channels) < 1:
        raise ValueError('input_channels must be at least 1.')

    if float(dropout) < 0 or float(dropout) >= 1:
        raise ValueError('dropout must be in the range [0, 1).')

    if int(final_kernel_size) not in (1, 3):
        raise ValueError('final_kernel_size must be 1 or 3.')


def validate_unet_args(
    num_of_points, input_channels, base_channels, depth, channel_multiplier, max_channels, dropout, final_kernel_size
):
    """Validate U-Net construction values."""
    validate_common_model_args(num_of_points, input_channels, dropout, final_kernel_size)

    if int(base_channels) < 1:
        raise ValueError('base_channels must be at least 1.')

    if int(depth) < 1:
        raise ValueError('depth must be at least 1.')

    if int(channel_multiplier) < 1:
        raise ValueError('channel_multiplier must be at least 1.')

    if int(max_channels) < int(base_channels):
        raise ValueError('max_channels must be greater than or equal to base_channels.')


def validate_hrnet_args(
    num_of_points, input_channels, hrnet_width, hrnet_modules, hrnet_blocks, dropout, final_kernel_size
):
    """Validate HRNet construction values."""
    validate_common_model_args(num_of_points, input_channels, dropout, final_kernel_size)

    if int(hrnet_width) < 4:
        raise ValueError('hrnet_width must be at least 4.')

    if int(hrnet_modules) < 1:
        raise ValueError('hrnet_modules must be at least 1.')

    if int(hrnet_blocks) < 1:
        raise ValueError('hrnet_blocks must be at least 1.')


def validate_hourglass_args(
    num_of_points,
    input_channels,
    hourglass_features,
    hourglass_stacks,
    hourglass_depth,
    hourglass_blocks,
    dropout,
    final_kernel_size,
):
    """Validate stacked-hourglass construction values."""
    validate_common_model_args(num_of_points, input_channels, dropout, final_kernel_size)

    if int(hourglass_features) < 16:
        raise ValueError('hourglass_features must be at least 16.')

    if int(hourglass_stacks) < 1:
        raise ValueError('hourglass_stacks must be at least 1.')

    if int(hourglass_depth) < 1:
        raise ValueError('hourglass_depth must be at least 1.')

    if int(hourglass_blocks) < 1:
        raise ValueError('hourglass_blocks must be at least 1.')


def count_trainable_parameters(model):
    """Return trainable parameter count."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
