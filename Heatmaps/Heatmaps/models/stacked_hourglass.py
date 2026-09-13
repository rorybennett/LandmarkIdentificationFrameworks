import math
import torch
from torch import nn
from torch.nn import functional as F

MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30
from .common import (
    OutputActivationMixin,
    ResidualBlock,
    make_residual_sequence,
    build_normalisation,
    build_activation,
    validate_hourglass_args,
)


class HourglassModule(nn.Module):
    """Recursive bottom-up and top-down feature-processing module."""

    def __init__(
        self, depth, features, blocks, normalisation='batch', activation='relu', dropout=0.0, padding_mode='zeros'
    ):
        super().__init__()
        self.depth = int(depth)
        self.upper = make_residual_sequence(
            features, features, blocks, normalisation, activation, dropout, padding_mode
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lower_pre = make_residual_sequence(
            features, features, blocks, normalisation, activation, dropout, padding_mode
        )
        self.lower = (
            HourglassModule(depth - 1, features, blocks, normalisation, activation, dropout, padding_mode)
            if depth > 1
            else make_residual_sequence(features, features, blocks, normalisation, activation, dropout, padding_mode)
        )
        self.lower_post = make_residual_sequence(
            features, features, blocks, normalisation, activation, dropout, padding_mode
        )

    def forward(self, x):
        upper = self.upper(x)
        lower = self.lower_pre(self.pool(x))
        lower = self.lower(lower)
        lower = self.lower_post(lower)
        lower = F.interpolate(lower, size=upper.shape[-2:], mode='nearest')
        return upper + lower


class StackedHourglassHeatmap(OutputActivationMixin, nn.Module):
    """Stacked hourglass model with intermediate heatmap feedback and auxiliary outputs."""

    def __init__(
        self,
        num_of_points,
        input_channels=1,
        hourglass_features=128,
        hourglass_stacks=2,
        hourglass_depth=4,
        hourglass_blocks=1,
        normalisation='batch',
        activation='relu',
        dropout=0.0,
        output_activation='none',
        padding_mode='zeros',
        final_kernel_size=1,
    ):
        super().__init__()
        validate_hourglass_args(
            num_of_points=num_of_points,
            input_channels=input_channels,
            hourglass_features=hourglass_features,
            hourglass_stacks=hourglass_stacks,
            hourglass_depth=hourglass_depth,
            hourglass_blocks=hourglass_blocks,
            dropout=dropout,
            final_kernel_size=final_kernel_size,
        )
        self.num_of_points = int(num_of_points)
        self.input_channels = int(input_channels)
        self.configure_output_activation(output_activation)
        use_bias = normalisation is None or str(normalisation).lower() == 'none'
        features = int(hourglass_features)
        stem_channels = max(features // 2, 16)
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                padding_mode=padding_mode,
                bias=use_bias,
            ),
            build_normalisation(normalisation, stem_channels),
            build_activation(activation),
            ResidualBlock(
                stem_channels,
                features,
                normalisation=normalisation,
                activation=activation,
                dropout=dropout,
                padding_mode=padding_mode,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(
                features,
                features,
                normalisation=normalisation,
                activation=activation,
                dropout=dropout,
                padding_mode=padding_mode,
            ),
        )
        self.hourglasses = nn.ModuleList(
            [
                HourglassModule(
                    hourglass_depth, features, hourglass_blocks, normalisation, activation, dropout, padding_mode
                )
                for _ in range(int(hourglass_stacks))
            ]
        )
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    make_residual_sequence(
                        features, features, hourglass_blocks, normalisation, activation, dropout, padding_mode
                    ),
                    nn.Conv2d(features, features, kernel_size=1, bias=use_bias),
                    build_normalisation(normalisation, features),
                    build_activation(activation),
                )
                for _ in range(int(hourglass_stacks))
            ]
        )
        self.heatmap_heads = nn.ModuleList(
            [
                nn.Conv2d(features, num_of_points, kernel_size=final_kernel_size, padding=final_kernel_size // 2)
                for _ in range(int(hourglass_stacks))
            ]
        )
        self.feature_feedback = nn.ModuleList(
            [nn.Conv2d(features, features, kernel_size=1) for _ in range(int(hourglass_stacks) - 1)]
        )
        self.heatmap_feedback = nn.ModuleList(
            [nn.Conv2d(num_of_points, features, kernel_size=1) for _ in range(int(hourglass_stacks) - 1)]
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.stem(x)
        heatmaps = []

        for stack_index, (hourglass, feature_block, heatmap_head) in enumerate(
            zip(self.hourglasses, self.features, self.heatmap_heads)
        ):
            stack_features = feature_block(hourglass(features))
            stack_heatmaps = heatmap_head(stack_features)
            heatmaps.append(stack_heatmaps)

            if stack_index < len(self.hourglasses) - 1:
                features = (
                    features
                    + self.feature_feedback[stack_index](stack_features)
                    + self.heatmap_feedback[stack_index](stack_heatmaps)
                )

        resized_heatmaps = [
            self.apply_output_activation(F.interpolate(heatmap, size=input_size, mode='bilinear', align_corners=False))
            for heatmap in heatmaps
        ]
        return {'heatmaps': resized_heatmaps[-1], 'auxiliary_heatmaps': resized_heatmaps[:-1]}
