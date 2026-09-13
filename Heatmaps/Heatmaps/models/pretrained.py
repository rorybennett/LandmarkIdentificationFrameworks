"""Shared full-resolution decoder and staged fine-tuning for pretrained ViTs."""

from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def validate_model_options(image_size, input_channels, decoder_channels, output_activation, final_kernel_size):
    if not isinstance(image_size, int) or image_size < 16 or input_channels != 3:
        raise ValueError('Pretrained ViT models require image_size >=16 and three channels.')
    if (
        decoder_channels < 16
        or output_activation not in ('none', 'sigmoid', 'softplus')
        or final_kernel_size not in (1, 3)
    ):
        raise ValueError('Invalid pretrained heatmap decoder options.')


def group_norm(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class PretrainedHeatmap(nn.Module):
    """Fine-tune only the image encoder and the new landmark head."""

    def __init__(
        self,
        num_of_points,
        input_channels=3,
        image_size=512,
        decoder_channels=256,
        output_activation='none',
        final_kernel_size=1,
        gradient_checkpointing=True,
        pretrained_checkpoint=None,
        initialise_pretrained=True,
    ):
        super().__init__()
        validate_model_options(image_size, input_channels, decoder_channels, output_activation, final_kernel_size)
        if not 1 <= int(num_of_points) <= 30:
            raise ValueError('num_of_points must be between 1 and 30.')
        if initialise_pretrained and (not pretrained_checkpoint or not Path(pretrained_checkpoint).is_file()):
            raise FileNotFoundError(
                'A local pretrained ViT-B checkpoint is required. Training from random encoder weights is disabled.'
            )
        self.image_size = image_size
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.output_activation = output_activation
        self.image_encoder = self.create_encoder(image_size)
        channels = int(decoder_channels)
        layers = [nn.Conv2d(256, channels, 1), group_norm(channels), nn.GELU()]
        for _ in range(4):
            next_channels = max(channels // 2, 32)
            layers.extend(
                [
                    nn.ConvTranspose2d(channels, next_channels, 4, stride=2, padding=1),
                    group_norm(next_channels),
                    nn.GELU(),
                ]
            )
            channels = next_channels
        layers.append(nn.Conv2d(channels, int(num_of_points), final_kernel_size, padding=final_kernel_size // 2))
        self.decoder = nn.Sequential(*layers)
        self._weights_ready = False
        self._encoder_enabled = False
        if initialise_pretrained:
            self.load_pretrained_encoder(pretrained_checkpoint)
        self.set_finetuning_stage(False, 12)

    def create_encoder(self, image_size):
        raise NotImplementedError

    def load_pretrained_encoder(self, path):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict:
            raise ValueError('Landmark checkpoints must be loaded strictly; partial encoder loading is disabled.')
        result = super().load_state_dict(state_dict, strict=True, assign=assign)
        self._weights_ready = True
        return result

    def decoder_parameters(self):
        return list(self.decoder.parameters())

    def finetuning_parameters(self, last_blocks):
        """Select the neck and final blocks, or the entire encoder when all 12 are selected."""
        if not 1 <= int(last_blocks) <= 12:
            raise ValueError('last_blocks must be between 1 and 12.')
        if int(last_blocks) == 12:
            return list(self.image_encoder.parameters())
        return [
            parameter
            for block in list(self.image_encoder.blocks)[-int(last_blocks) :]
            for parameter in block.parameters()
        ] + list(self.image_encoder.neck.parameters())

    def set_finetuning_stage(self, enabled, last_blocks):
        self._encoder_enabled = bool(enabled)
        self.image_encoder.requires_grad_(False)
        if enabled:
            for parameter in self.finetuning_parameters(last_blocks):
                parameter.requires_grad_(True)
        self.decoder.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if not self._encoder_enabled:
            self.image_encoder.eval()
        else:
            for module in [self.image_encoder.patch_embed, *self.image_encoder.blocks, self.image_encoder.neck]:
                if not any(parameter.requires_grad for parameter in module.parameters()):
                    module.eval()
        return self

    def encode(self, image):
        if not self._encoder_enabled or not self.training or not self.gradient_checkpointing:
            return self.image_encoder(image)
        features = self.image_encoder.patch_embed(image)
        if self.image_encoder.pos_embed is not None:
            features = features + self.image_encoder.pos_embed
        for block in self.image_encoder.blocks:
            if any(parameter.requires_grad for parameter in block.parameters()):
                features = checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
        return self.image_encoder.neck(features.permute(0, 3, 1, 2))

    def forward(self, image):
        if not self._weights_ready:
            raise RuntimeError(
                'Load a complete pretrained encoder or landmark checkpoint before prediction or training.'
            )
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(
                f'Expected a batch of three-channel {self.image_size} x {self.image_size} letterboxed images.'
            )
        padded_size = ((self.image_size + 15) // 16) * 16
        image = F.pad(image, (0, padded_size - self.image_size, 0, padded_size - self.image_size))
        heatmaps = self.decoder(self.encode(image))[:, :, : self.image_size, : self.image_size]
        if tuple(heatmaps.shape[-2:]) != (self.image_size, self.image_size):
            raise RuntimeError('The Pretrained ViT encoder/heatmap decoder produced an incompatible spatial shape.')
        if self.output_activation == 'sigmoid':
            return torch.sigmoid(heatmaps)
        if self.output_activation == 'softplus':
            return F.softplus(heatmaps)
        return heatmaps
