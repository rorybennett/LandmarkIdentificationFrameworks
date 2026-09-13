"""ViTPose-B-compatible encoder and shared landmark decoder.

Architecture reference: ViTAE-Transformer/ViTPose, mmpose/models/backbones/vit.py.
Uses the standard COCO 256x192 backbone weights, excluding the human pose head.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .pretrained import PretrainedHeatmap


class PatchEmbed(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Conv2d(3, width, 16, stride=16, padding=2)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3, bias=True)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        weights = ((q * (q.shape[-1] ** -0.5)) @ k.transpose(-2, -1)).softmax(-1)
        return self.proj((weights @ v).transpose(1, 2).reshape(b, n, c))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DropPath(nn.Module):
    """Per-sample stochastic depth, with no checkpoint tensors."""

    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0], 1, 1)).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, width, heads, drop_path=0):
        super().__init__()
        self.drop_path = DropPath(drop_path)
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = MLP(width)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ViTPoseEncoder(nn.Module):
    def __init__(self, image_size, width=768, depth=12, heads=12):
        super().__init__()
        self.grid = (image_size + 15) // 16
        self.patch_embed = PatchEmbed(width)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid**2 + 1, width))
        self.blocks = nn.ModuleList([Block(width, heads, float(rate)) for rate in torch.linspace(0, 0.3, depth)])
        self.last_norm = nn.LayerNorm(width, eps=1e-6)
        self.neck = nn.Sequential(nn.Conv2d(width, 256, 1), nn.GroupNorm(32, 256))

    def forward(self, image, checkpoint_blocks=False):
        x = self.patch_embed(image) + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for block in self.blocks:
            x = (
                checkpoint(block, x, use_reentrant=False)
                if checkpoint_blocks and any(p.requires_grad for p in block.parameters())
                else block(x)
            )
        x = self.last_norm(x).transpose(1, 2).reshape(image.shape[0], -1, self.grid, self.grid)
        return self.neck(x)


class ViTPoseHeatmap(PretrainedHeatmap):
    def create_encoder(self, image_size):
        return ViTPoseEncoder(image_size)

    def load_pretrained_encoder(self, path):
        """Load a standard ViTPose-B COCO 256x192 pose checkpoint backbone."""
        state = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict):
            raise ValueError('Expected a ViTPose-B pose checkpoint state_dict.')
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        encoder = {key.removeprefix('backbone.'): value for key, value in state.items() if key.startswith('backbone.')}
        expected = {key: value for key, value in self.image_encoder.state_dict().items() if not key.startswith('neck.')}
        if set(encoder) != set(expected):
            raise ValueError(
                f'ViTPose-B backbone keys mismatch: missing={set(expected)-set(encoder)}, unexpected={set(encoder)-set(expected)}'
            )
        pos = encoder['pos_embed']
        if tuple(pos.shape) != (1, 193, expected['pos_embed'].shape[-1]):
            raise ValueError('Use the standard ViTPose-B 256x192 pose checkpoint (16x12 positional grid).')
        spatial = pos[:, 1:].reshape(1, 16, 12, -1).permute(0, 3, 1, 2)
        spatial = F.interpolate(spatial, size=(self.image_encoder.grid,) * 2, mode='bicubic', align_corners=False)
        encoder['pos_embed'] = torch.cat((pos[:, :1], spatial.flatten(2).transpose(1, 2)), dim=1)
        self.image_encoder.load_state_dict(dict(self.image_encoder.state_dict(), **encoder), strict=True)
        self._weights_ready = True

    def decoder_parameters(self):
        # This projection has no pose-pretrained weights: train it from epoch one.
        return super().decoder_parameters() + list(self.image_encoder.neck.parameters())

    def finetuning_parameters(self, last_blocks):
        values = super().finetuning_parameters(last_blocks)
        neck_ids = {id(parameter) for parameter in self.image_encoder.neck.parameters()}
        values = [parameter for parameter in values if id(parameter) not in neck_ids]
        if last_blocks != 12:
            values += list(self.image_encoder.last_norm.parameters())
        return values

    def set_finetuning_stage(self, enabled, last_blocks):
        super().set_finetuning_stage(enabled, last_blocks)
        self.image_encoder.neck.requires_grad_(True)

    def encode(self, image):
        mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return self.image_encoder(
            (image - mean) / std,
            checkpoint_blocks=self.training and self._encoder_enabled and self.gradient_checkpointing,
        )
