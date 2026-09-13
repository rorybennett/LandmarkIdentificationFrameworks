import math
import torch
from torch import nn
from torch.nn import functional as F

MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30
from .common import OutputActivationMixin, ResidualBlock, build_normalisation, build_activation, validate_hrnet_args


class HighResolutionModule(nn.Module):
    """Process parallel resolutions and repeatedly fuse their representations."""

    def __init__(
        self, channels, blocks_per_branch=2, normalisation='batch', activation='relu', dropout=0.0, padding_mode='zeros'
    ):
        super().__init__()
        self.channels = [int(channel) for channel in channels]
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ResidualBlock(
                            channel,
                            channel,
                            normalisation=normalisation,
                            activation=activation,
                            dropout=dropout,
                            padding_mode=padding_mode,
                        )
                        for _ in range(int(blocks_per_branch))
                    ]
                )
                for channel in self.channels
            ]
        )
        self.fuse_layers = nn.ModuleList(
            [
                self.build_fuse_row(
                    target_index=index, normalisation=normalisation, activation=activation, padding_mode=padding_mode
                )
                for index in range(len(self.channels))
            ]
        )
        self.activation = build_activation(activation)

    def build_fuse_row(self, target_index, normalisation, activation, padding_mode):
        """Build conversions from every source branch into one target branch."""
        row = nn.ModuleList()
        use_bias = normalisation is None or str(normalisation).lower() == 'none'

        for source_index, source_channels in enumerate(self.channels):
            target_channels = self.channels[target_index]

            if source_index == target_index:
                row.append(nn.Identity())
            elif source_index > target_index:
                row.append(
                    nn.Sequential(
                        nn.Conv2d(source_channels, target_channels, kernel_size=1, bias=use_bias),
                        build_normalisation(normalisation, target_channels),
                    )
                )
            else:
                layers = []
                current_channels = source_channels

                for downsample_index in range(target_index - source_index):
                    final_step = downsample_index == target_index - source_index - 1
                    next_channels = target_channels if final_step else current_channels
                    layers.extend(
                        [
                            nn.Conv2d(
                                current_channels,
                                next_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                padding_mode=padding_mode,
                                bias=use_bias,
                            ),
                            build_normalisation(normalisation, next_channels),
                        ]
                    )

                    if not final_step:
                        layers.append(build_activation(activation))

                    current_channels = next_channels

                row.append(nn.Sequential(*layers))

        return row

    def forward(self, inputs):
        branch_outputs = [branch(branch_input) for branch, branch_input in zip(self.branches, inputs)]
        fused_outputs = []

        for target_index, fuse_row in enumerate(self.fuse_layers):
            target_size = branch_outputs[target_index].shape[-2:]
            fused = None

            for source_index, conversion in enumerate(fuse_row):
                converted = conversion(branch_outputs[source_index])

                if source_index > target_index:
                    converted = F.interpolate(converted, size=target_size, mode='bilinear', align_corners=False)

                fused = converted if fused is None else fused + converted

            fused_outputs.append(self.activation(fused))

        return fused_outputs


class HRNetHeatmap(OutputActivationMixin, nn.Module):
    """Compact HRNet-style model with four parallel resolutions and repeated fusion."""

    def __init__(
        self,
        num_of_points,
        input_channels=1,
        hrnet_width=32,
        hrnet_modules=3,
        hrnet_blocks=2,
        normalisation='batch',
        activation='relu',
        dropout=0.0,
        output_activation='none',
        padding_mode='zeros',
        final_kernel_size=1,
    ):
        super().__init__()
        validate_hrnet_args(
            num_of_points=num_of_points,
            input_channels=input_channels,
            hrnet_width=hrnet_width,
            hrnet_modules=hrnet_modules,
            hrnet_blocks=hrnet_blocks,
            dropout=dropout,
            final_kernel_size=final_kernel_size,
        )
        self.num_of_points = int(num_of_points)
        self.input_channels = int(input_channels)
        self.configure_output_activation(output_activation)
        use_bias = normalisation is None or str(normalisation).lower() == 'none'
        width = int(hrnet_width)
        channels = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels, width, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=use_bias
            ),
            build_normalisation(normalisation, width),
            build_activation(activation),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=use_bias),
            build_normalisation(normalisation, width),
            build_activation(activation),
            ResidualBlock(
                width,
                width,
                normalisation=normalisation,
                activation=activation,
                dropout=dropout,
                padding_mode=padding_mode,
            ),
        )
        self.transitions = nn.ModuleList(
            [nn.Identity()]
            + [
                self.build_transition(width, channels[index], index, normalisation, activation, padding_mode)
                for index in range(1, len(channels))
            ]
        )
        self.fusion_modules = nn.ModuleList(
            [
                HighResolutionModule(
                    channels=channels,
                    blocks_per_branch=hrnet_blocks,
                    normalisation=normalisation,
                    activation=activation,
                    dropout=dropout,
                    padding_mode=padding_mode,
                )
                for _ in range(int(hrnet_modules))
            ]
        )
        head_channels = sum(channels)
        self.head = nn.Sequential(
            nn.Conv2d(head_channels, width * 4, kernel_size=1, bias=use_bias),
            build_normalisation(normalisation, width * 4),
            build_activation(activation),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(width * 4, num_of_points, kernel_size=final_kernel_size, padding=final_kernel_size // 2),
        )

    @staticmethod
    def build_transition(in_channels, out_channels, downsample_steps, normalisation, activation, padding_mode):
        """Create a lower-resolution branch from the stem representation."""
        use_bias = normalisation is None or str(normalisation).lower() == 'none'
        layers = []
        current_channels = in_channels

        for step_index in range(int(downsample_steps)):
            next_channels = out_channels if step_index == int(downsample_steps) - 1 else current_channels
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        next_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        padding_mode=padding_mode,
                        bias=use_bias,
                    ),
                    build_normalisation(normalisation, next_channels),
                    build_activation(activation),
                ]
            )
            current_channels = next_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.shape[-2:]
        stem = self.stem(x)
        branches = [transition(stem) for transition in self.transitions]

        for module in self.fusion_modules:
            branches = module(branches)

        high_resolution_size = branches[0].shape[-2:]
        fused = torch.cat(
            [
                (
                    branch
                    if index == 0
                    else F.interpolate(branch, size=high_resolution_size, mode='bilinear', align_corners=False)
                )
                for index, branch in enumerate(branches)
            ],
            dim=1,
        )
        heatmaps = self.head(fused)
        heatmaps = F.interpolate(heatmaps, size=input_size, mode='bilinear', align_corners=False)
        return self.apply_output_activation(heatmaps)
