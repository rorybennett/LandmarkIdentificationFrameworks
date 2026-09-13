"""MedSAM ViT-B encoder, adapted to a configured heatmap canvas."""

from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from .sam.image_encoder import ImageEncoderViT
from .pretrained import PretrainedHeatmap


def create_image_encoder(image_size):
    return ImageEncoderViT(
        img_size=((image_size + 15) // 16) * 16,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        out_chans=256,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
    )


class MedSAMHeatmap(PretrainedHeatmap):
    def create_encoder(self, image_size):
        return create_image_encoder(image_size)

    def load_pretrained_encoder(self, path):
        """Require every encoder tensor to match; ignore only the unused SAM modules."""
        state = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(state, dict):
            raise ValueError('Expected the original MedSAM ViT-B state dictionary.')
        for wrapper in ('model', 'state_dict'):
            if wrapper in state and isinstance(state[wrapper], dict):
                state = state[wrapper]
                break
        state = {str(name).removeprefix('module.'): value for name, value in state.items()}
        encoder_state = {
            name.removeprefix('image_encoder.'): value
            for name, value in state.items()
            if name.startswith('image_encoder.')
        }
        if not encoder_state:
            raise ValueError(
                'No image_encoder.* tensors found. Supply original medsam_vit_b.pth, not a landmark checkpoint or another MedSAM variant.'
            )
        expected = self.image_encoder.state_dict()
        for name, target in expected.items():
            source = encoder_state.get(name)
            if source is None or source.shape == target.shape:
                continue
            if (
                name == 'pos_embed'
                and source.ndim == 4
                and source.shape[0] == target.shape[0]
                and source.shape[-1] == target.shape[-1]
                and source.shape[1:3] == (64, 64)
            ):
                encoder_state[name] = (
                    F.interpolate(
                        source.permute(0, 3, 1, 2).float(), size=target.shape[1:3], mode='bicubic', align_corners=False
                    )
                    .permute(0, 2, 3, 1)
                    .to(source.dtype)
                )
            elif (
                name.endswith(('attn.rel_pos_h', 'attn.rel_pos_w'))
                and source.ndim == 2
                and source.shape[1] == target.shape[1]
                and source.shape[0] == 127
            ):
                encoder_state[name] = (
                    F.interpolate(
                        source.T.unsqueeze(0).float(), size=target.shape[0], mode='linear', align_corners=False
                    )
                    .squeeze(0)
                    .T.to(source.dtype)
                )
        self.image_encoder.load_state_dict(encoder_state, strict=True)
        self._weights_ready = True
