"""
Configurable heatmap-regression models for landmark localisation.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30


class OutputActivationMixin:
    """Apply the configured output activation to heatmap tensors."""

    def configure_output_activation(self, output_activation):
        self.output_activation = None if output_activation is None or str(output_activation).lower() == 'none' else str(output_activation).lower()

    def apply_output_activation(self, x):
        if self.output_activation is None:
            return x

        if self.output_activation == 'sigmoid':
            return torch.sigmoid(x)

        if self.output_activation == 'softplus':
            return F.softplus(x)

        raise ValueError(f'Unknown output_activation: {self.output_activation}')


class ConvBlock(nn.Module):
    """Run two convolutional layers at one U-Net level."""

    def __init__(self, in_channels, out_channels, normalisation='batch', activation='relu', dropout=0.0, padding_mode='zeros'):
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
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        elif upsampling == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            raise ValueError(f'Unknown upsampling: {upsampling}')

        self.conv = ConvBlock(in_channels + skip_channels, out_channels, normalisation, activation, dropout, padding_mode)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        return self.conv(torch.cat((skip, x), dim=1))


class UNetHeatmap(OutputActivationMixin, nn.Module):
    """U-Net heatmap regressor with one output channel per landmark."""

    def __init__(self, num_of_points, input_channels=1, base_channels=32, depth=4, channel_multiplier=2, max_channels=512, normalisation='batch', activation='relu', dropout=0.0, upsampling='bilinear', output_activation='none', padding_mode='zeros', final_kernel_size=1):
        super().__init__()
        validate_unet_args(num_of_points=num_of_points, input_channels=input_channels, base_channels=base_channels, depth=depth, channel_multiplier=channel_multiplier,
                           max_channels=max_channels, dropout=dropout, final_kernel_size=final_kernel_size)
        self.num_of_points = int(num_of_points)
        self.input_channels = int(input_channels)
        self.configure_output_activation(output_activation)
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

        return self.apply_output_activation(self.output_layer(x))


class ResidualBlock(nn.Module):
    """Residual convolution block shared by HRNet and hourglass models."""

    def __init__(self, in_channels, out_channels, stride=1, normalisation='batch', activation='relu', dropout=0.0, padding_mode='zeros'):
        super().__init__()
        use_bias = normalisation is None or str(normalisation).lower() == 'none'
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode=padding_mode, bias=use_bias)
        self.norm1 = build_normalisation(normalisation, out_channels)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode, bias=use_bias)
        self.norm2 = build_normalisation(normalisation, out_channels)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=use_bias), build_normalisation(normalisation, out_channels))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class HighResolutionModule(nn.Module):
    """Process parallel resolutions and repeatedly fuse their representations."""

    def __init__(self, channels, blocks_per_branch=2, normalisation='batch', activation='relu', dropout=0.0, padding_mode='zeros'):
        super().__init__()
        self.channels = [int(channel) for channel in channels]
        self.branches = nn.ModuleList([
            nn.Sequential(*[ResidualBlock(channel, channel, normalisation=normalisation, activation=activation, dropout=dropout, padding_mode=padding_mode)
                            for _ in range(int(blocks_per_branch))]) for channel in self.channels
        ])
        self.fuse_layers = nn.ModuleList([self.build_fuse_row(target_index=index, normalisation=normalisation, activation=activation, padding_mode=padding_mode)
                                          for index in range(len(self.channels))])
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
                row.append(nn.Sequential(nn.Conv2d(source_channels, target_channels, kernel_size=1, bias=use_bias), build_normalisation(normalisation, target_channels)))
            else:
                layers = []
                current_channels = source_channels

                for downsample_index in range(target_index - source_index):
                    final_step = downsample_index == target_index - source_index - 1
                    next_channels = target_channels if final_step else current_channels
                    layers.extend([
                        nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=use_bias),
                        build_normalisation(normalisation, next_channels),
                    ])

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

    def __init__(self, num_of_points, input_channels=1, hrnet_width=32, hrnet_modules=3, hrnet_blocks=2, normalisation='batch', activation='relu', dropout=0.0, output_activation='none', padding_mode='zeros', final_kernel_size=1):
        super().__init__()
        validate_hrnet_args(num_of_points=num_of_points, input_channels=input_channels, hrnet_width=hrnet_width, hrnet_modules=hrnet_modules,
                            hrnet_blocks=hrnet_blocks, dropout=dropout, final_kernel_size=final_kernel_size)
        self.num_of_points = int(num_of_points)
        self.input_channels = int(input_channels)
        self.configure_output_activation(output_activation)
        use_bias = normalisation is None or str(normalisation).lower() == 'none'
        width = int(hrnet_width)
        channels = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=use_bias),
            build_normalisation(normalisation, width),
            build_activation(activation),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=use_bias),
            build_normalisation(normalisation, width),
            build_activation(activation),
            ResidualBlock(width, width, normalisation=normalisation, activation=activation, dropout=dropout, padding_mode=padding_mode),
        )
        self.transitions = nn.ModuleList([nn.Identity()] + [self.build_transition(width, channels[index], index, normalisation, activation, padding_mode)
                                                            for index in range(1, len(channels))])
        self.fusion_modules = nn.ModuleList([HighResolutionModule(channels=channels, blocks_per_branch=hrnet_blocks, normalisation=normalisation,
                                                           activation=activation, dropout=dropout, padding_mode=padding_mode)
                                      for _ in range(int(hrnet_modules))])
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
            layers.extend([
                nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode, bias=use_bias),
                build_normalisation(normalisation, next_channels),
                build_activation(activation),
            ])
            current_channels = next_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.shape[-2:]
        stem = self.stem(x)
        branches = [transition(stem) for transition in self.transitions]

        for module in self.fusion_modules:
            branches = module(branches)

        high_resolution_size = branches[0].shape[-2:]
        fused = torch.cat([branch if index == 0 else F.interpolate(branch, size=high_resolution_size, mode='bilinear', align_corners=False)
                           for index, branch in enumerate(branches)], dim=1)
        heatmaps = self.head(fused)
        heatmaps = F.interpolate(heatmaps, size=input_size, mode='bilinear', align_corners=False)
        return self.apply_output_activation(heatmaps)


class HourglassModule(nn.Module):
    """Recursive bottom-up and top-down feature-processing module."""

    def __init__(self, depth, features, blocks, normalisation='batch', activation='relu', dropout=0.0, padding_mode='zeros'):
        super().__init__()
        self.depth = int(depth)
        self.upper = make_residual_sequence(features, features, blocks, normalisation, activation, dropout, padding_mode)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lower_pre = make_residual_sequence(features, features, blocks, normalisation, activation, dropout, padding_mode)
        self.lower = HourglassModule(depth - 1, features, blocks, normalisation, activation, dropout, padding_mode) if depth > 1 else make_residual_sequence(features, features, blocks, normalisation, activation, dropout, padding_mode)
        self.lower_post = make_residual_sequence(features, features, blocks, normalisation, activation, dropout, padding_mode)

    def forward(self, x):
        upper = self.upper(x)
        lower = self.lower_pre(self.pool(x))
        lower = self.lower(lower)
        lower = self.lower_post(lower)
        lower = F.interpolate(lower, size=upper.shape[-2:], mode='nearest')
        return upper + lower


class StackedHourglassHeatmap(OutputActivationMixin, nn.Module):
    """Stacked hourglass model with intermediate heatmap feedback and auxiliary outputs."""

    def __init__(self, num_of_points, input_channels=1, hourglass_features=128, hourglass_stacks=2, hourglass_depth=4, hourglass_blocks=1, normalisation='batch', activation='relu', dropout=0.0, output_activation='none', padding_mode='zeros', final_kernel_size=1):
        super().__init__()
        validate_hourglass_args(num_of_points=num_of_points, input_channels=input_channels, hourglass_features=hourglass_features,
                                hourglass_stacks=hourglass_stacks, hourglass_depth=hourglass_depth, hourglass_blocks=hourglass_blocks,
                                dropout=dropout, final_kernel_size=final_kernel_size)
        self.num_of_points = int(num_of_points)
        self.input_channels = int(input_channels)
        self.configure_output_activation(output_activation)
        use_bias = normalisation is None or str(normalisation).lower() == 'none'
        features = int(hourglass_features)
        stem_channels = max(features // 2, 16)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_channels, kernel_size=7, stride=2, padding=3, padding_mode=padding_mode, bias=use_bias),
            build_normalisation(normalisation, stem_channels),
            build_activation(activation),
            ResidualBlock(stem_channels, features, normalisation=normalisation, activation=activation, dropout=dropout, padding_mode=padding_mode),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(features, features, normalisation=normalisation, activation=activation, dropout=dropout, padding_mode=padding_mode),
        )
        self.hourglasses = nn.ModuleList([HourglassModule(hourglass_depth, features, hourglass_blocks, normalisation, activation, dropout, padding_mode)
                                          for _ in range(int(hourglass_stacks))])
        self.features = nn.ModuleList([nn.Sequential(make_residual_sequence(features, features, hourglass_blocks, normalisation, activation, dropout, padding_mode),
                                                     nn.Conv2d(features, features, kernel_size=1, bias=use_bias), build_normalisation(normalisation, features),
                                                     build_activation(activation)) for _ in range(int(hourglass_stacks))])
        self.heatmap_heads = nn.ModuleList([nn.Conv2d(features, num_of_points, kernel_size=final_kernel_size, padding=final_kernel_size // 2)
                                            for _ in range(int(hourglass_stacks))])
        self.feature_feedback = nn.ModuleList([nn.Conv2d(features, features, kernel_size=1) for _ in range(int(hourglass_stacks) - 1)])
        self.heatmap_feedback = nn.ModuleList([nn.Conv2d(num_of_points, features, kernel_size=1) for _ in range(int(hourglass_stacks) - 1)])

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.stem(x)
        heatmaps = []

        for stack_index, (hourglass, feature_block, heatmap_head) in enumerate(zip(self.hourglasses, self.features, self.heatmap_heads)):
            stack_features = feature_block(hourglass(features))
            stack_heatmaps = heatmap_head(stack_features)
            heatmaps.append(stack_heatmaps)

            if stack_index < len(self.hourglasses) - 1:
                features = features + self.feature_feedback[stack_index](stack_features) + self.heatmap_feedback[stack_index](stack_heatmaps)

        resized_heatmaps = [self.apply_output_activation(F.interpolate(heatmap, size=input_size, mode='bilinear', align_corners=False)) for heatmap in heatmaps]
        return {'heatmaps': resized_heatmaps[-1], 'auxiliary_heatmaps': resized_heatmaps[:-1]}


class ViTPoseHeatmap(OutputActivationMixin, nn.Module):
    """Plain Vision Transformer backbone with a lightweight heatmap decoder."""

    def __init__(self, num_of_points, input_channels=1, image_size=(512, 512), vit_patch_size=16, vit_embed_dim=384, vit_depth=8, vit_heads=6, vit_mlp_ratio=4.0, vit_dropout=0.0, vit_decoder_channels=256, output_activation='none', final_kernel_size=1):
        super().__init__()
        validate_vitpose_args(num_of_points=num_of_points, input_channels=input_channels, image_size=image_size, vit_patch_size=vit_patch_size,
                              vit_embed_dim=vit_embed_dim, vit_depth=vit_depth, vit_heads=vit_heads, vit_mlp_ratio=vit_mlp_ratio,
                              vit_dropout=vit_dropout, vit_decoder_channels=vit_decoder_channels, final_kernel_size=final_kernel_size)
        self.num_of_points = int(num_of_points)
        self.input_channels = int(input_channels)
        self.image_size = tuple(int(value) for value in image_size)
        self.patch_size = int(vit_patch_size)
        self.grid_size = (math.ceil(self.image_size[0] / self.patch_size), math.ceil(self.image_size[1] / self.patch_size))
        self.configure_output_activation(output_activation)
        embed_dim = int(vit_embed_dim)
        decoder_channels = int(vit_decoder_channels)
        self.patch_embedding = nn.Conv2d(input_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.grid_size[0] * self.grid_size[1], embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=int(vit_heads), dim_feedforward=int(embed_dim * float(vit_mlp_ratio)),
                                                   dropout=float(vit_dropout), activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(vit_depth), norm=nn.LayerNorm(embed_dim), enable_nested_tensor=False)
        decoder_layers = [nn.Conv2d(embed_dim, decoder_channels, kernel_size=1), build_normalisation('group', decoder_channels), nn.GELU()]
        upsample_steps = int(math.log2(self.patch_size))

        for _ in range(upsample_steps):
            next_channels = max(decoder_channels // 2, 32)
            decoder_layers.extend([nn.ConvTranspose2d(decoder_channels, next_channels, kernel_size=4, stride=2, padding=1), build_normalisation('group', next_channels), nn.GELU()])
            decoder_channels = next_channels

        decoder_layers.append(nn.Conv2d(decoder_channels, num_of_points, kernel_size=final_kernel_size, padding=final_kernel_size // 2))
        self.decoder = nn.Sequential(*decoder_layers)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        input_size = x.shape[-2:]
        pad_height = (self.patch_size - input_size[0] % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - input_size[1] % self.patch_size) % self.patch_size
        padded = F.pad(x, (0, pad_width, 0, pad_height)) if pad_height or pad_width else x
        features = self.patch_embedding(padded)
        grid_height, grid_width = features.shape[-2:]

        if (grid_height, grid_width) != self.grid_size:
            raise ValueError(f'ViTPose received an image producing a {grid_height} x {grid_width} patch grid, expected {self.grid_size[0]} x {self.grid_size[1]}. '
                             f'Configured image_size is {self.image_size}.')

        tokens = features.flatten(2).transpose(1, 2) + self.position_embedding
        tokens = self.encoder(tokens)
        features = tokens.transpose(1, 2).reshape(x.shape[0], -1, grid_height, grid_width)
        heatmaps = self.decoder(features)

        if heatmaps.shape[-2:] != padded.shape[-2:]:
            heatmaps = F.interpolate(heatmaps, size=padded.shape[-2:], mode='bilinear', align_corners=False)

        heatmaps = heatmaps[:, :, :input_size[0], :input_size[1]]
        return self.apply_output_activation(heatmaps)


def make_residual_sequence(in_channels, out_channels, blocks, normalisation, activation, dropout, padding_mode):
    """Create a sequence of residual blocks."""
    layers = [ResidualBlock(in_channels, out_channels, normalisation=normalisation, activation=activation, dropout=dropout, padding_mode=padding_mode)]
    layers.extend([ResidualBlock(out_channels, out_channels, normalisation=normalisation, activation=activation, dropout=dropout, padding_mode=padding_mode)
                   for _ in range(int(blocks) - 1)])
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

    raise TypeError('Heatmap models must return a tensor or a dictionary containing heatmaps and optional auxiliary_heatmaps.')


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
    return [min(int(base_channels) * (int(channel_multiplier) ** index), int(max_channels)) for index in range(int(depth) + 1)]


def validate_common_model_args(num_of_points, input_channels, dropout, final_kernel_size):
    """Validate values shared by heatmap models."""
    if int(num_of_points) < MIN_POINTS_PER_IMAGE or int(num_of_points) > MAX_POINTS_PER_IMAGE:
        raise ValueError(f'num_of_points must be between {MIN_POINTS_PER_IMAGE} and {MAX_POINTS_PER_IMAGE}. Got: {num_of_points}')

    if int(input_channels) < 1:
        raise ValueError('input_channels must be at least 1.')

    if float(dropout) < 0 or float(dropout) >= 1:
        raise ValueError('dropout must be in the range [0, 1).')

    if int(final_kernel_size) not in (1, 3):
        raise ValueError('final_kernel_size must be 1 or 3.')


def validate_unet_args(num_of_points, input_channels, base_channels, depth, channel_multiplier, max_channels, dropout, final_kernel_size):
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


def validate_hrnet_args(num_of_points, input_channels, hrnet_width, hrnet_modules, hrnet_blocks, dropout, final_kernel_size):
    """Validate HRNet construction values."""
    validate_common_model_args(num_of_points, input_channels, dropout, final_kernel_size)

    if int(hrnet_width) < 4:
        raise ValueError('hrnet_width must be at least 4.')

    if int(hrnet_modules) < 1:
        raise ValueError('hrnet_modules must be at least 1.')

    if int(hrnet_blocks) < 1:
        raise ValueError('hrnet_blocks must be at least 1.')


def validate_hourglass_args(num_of_points, input_channels, hourglass_features, hourglass_stacks, hourglass_depth, hourglass_blocks, dropout, final_kernel_size):
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


def validate_vitpose_args(num_of_points, input_channels, image_size, vit_patch_size, vit_embed_dim, vit_depth, vit_heads, vit_mlp_ratio, vit_dropout, vit_decoder_channels, final_kernel_size):
    """Validate ViTPose construction values."""
    validate_common_model_args(num_of_points, input_channels, vit_dropout, final_kernel_size)
    image_size = tuple(int(value) for value in image_size)

    if len(image_size) != 2 or min(image_size) < 1:
        raise ValueError('image_size must contain two positive values.')

    if int(vit_patch_size) < 2 or int(vit_patch_size) & (int(vit_patch_size) - 1):
        raise ValueError('vit_patch_size must be a power of two greater than or equal to 2.')

    if min(image_size) < int(vit_patch_size):
        raise ValueError('Both image dimensions must be at least vit_patch_size.')

    if int(vit_embed_dim) < 8:
        raise ValueError('vit_embed_dim must be at least 8.')

    if int(vit_depth) < 1:
        raise ValueError('vit_depth must be at least 1.')

    if int(vit_heads) < 1 or int(vit_embed_dim) % int(vit_heads) != 0:
        raise ValueError('vit_heads must be at least 1 and divide vit_embed_dim exactly.')

    if float(vit_mlp_ratio) <= 0:
        raise ValueError('vit_mlp_ratio must be greater than 0.')

    if int(vit_decoder_channels) < 16:
        raise ValueError('vit_decoder_channels must be at least 16.')


def count_trainable_parameters(model):
    """Return trainable parameter count."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
