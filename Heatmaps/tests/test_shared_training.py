"""Shared checkpoint/progress contracts and pose-compatible encoder loading."""

import json
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from Heatmaps.models import vitpose
from Heatmaps.models.sam.image_encoder import ImageEncoderViT
from Heatmaps.train_model import TrainModel
from Heatmaps.utils.heatmap_inference_utils import load_model_from_checkpoint
from medsam_helpers import configs, pretrained, encoder_double, limited_threads


@pytest.mark.parametrize('name', ['unet_basic', 'hrnet', 'stacked_hourglass'])
def test_shared_progress_and_dual_exports(tmp_path, pretrained, name):
    data, train, model = configs(tmp_path, pretrained)
    data = replace(data, image_size=64, num_of_points=2, oversampling_factor=1)
    data.mark_list_file.write_text('one.png (10, 4) (10, 12)\ntwo.png (12, 4) (12, 12)\n')
    train = replace(
        train,
        max_training_epochs=2,
        device='cpu',
        landmark_constraint_loss='prostate-saus',
        visualise_validation_progress_images=2,
        visualise_validation_progress_epochs=1,
    )
    model = replace(
        model,
        network_name=name,
        base_channels=4,
        depth=1,
        max_channels=8,
        hrnet_width=4,
        hrnet_modules=1,
        hrnet_blocks=1,
        hourglass_features=16,
        hourglass_stacks=2,
        hourglass_depth=1,
        hourglass_blocks=1,
    )
    trainer = TrainModel(data, train, model, tmp_path / name)
    metrics = [{'loss': 1.0, 'error_px': 10.0}, {'loss': 2.0, 'error_px': 8.0}]
    with patch.object(TrainModel, 'validate', side_effect=metrics):
        trainer.train()
    for label, expected_epoch in [('best_validation_loss', 1), ('best_validation_pixel_error', 2)]:
        output = tmp_path / name / 'validation_progress' / 'epoch_0002' / label
        selection = json.loads((output / 'selection.json').read_text())
        assert selection['sample_names'] == ['two']
        assert selection['checkpoint_epoch'] == expected_epoch
        assert list(output.rglob('*.png'))
    for directory in ['validation_best_loss', 'validation_best_pixel_error']:
        assert (tmp_path / name / directory / 'validation_predictions.csv').exists()
    selected = torch.load(tmp_path / name / 'model_last_epoch.pth', weights_only=False)
    # Drawing must leave the training state and RNG trajectory unchanged.
    quiet = TrainModel(
        data, replace(train, visualise_validation_progress_epochs=0), model, tmp_path / (name + '_quiet')
    )
    with patch.object(TrainModel, 'validate', side_effect=metrics):
        quiet.train()
    baseline = torch.load(quiet.get_checkpoint_path('last_epoch'), weights_only=False)
    for key, value in selected['state_dict'].items():
        torch.testing.assert_close(value, baseline['state_dict'][key], rtol=0, atol=0)
    assert not (quiet.output_path / 'validation_progress').exists()


def test_vitpose_checkpoint_loading_and_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vitpose.ViTPoseHeatmap,
        'create_encoder',
        lambda self, size: vitpose.ViTPoseEncoder(size, width=32, depth=12, heads=4),
    )
    source = vitpose.ViTPoseEncoder(32, width=32, depth=12, heads=4)
    state = {'backbone.' + key: value for key, value in source.state_dict().items() if not key.startswith('neck.')}
    state['backbone.pos_embed'] = torch.randn(1, 193, 32)
    state['keypoint_head.final_layer.weight'] = torch.zeros(17, 256, 1, 1)
    path = tmp_path / 'pose.pth'
    torch.save({'state_dict': state}, path)
    model = vitpose.ViTPoseHeatmap(2, image_size=35, decoder_channels=16, pretrained_checkpoint=path)
    assert model.image_encoder.patch_embed.proj.padding == (2, 2)
    assert not hasattr(model.image_encoder, 'cls_token')
    assert model.image_encoder.pos_embed.shape == (1, 10, 32)
    assert not model.image_encoder.blocks[-1].attn.qkv.weight.requires_grad
    assert model.image_encoder.neck[0].weight.requires_grad
    torch.testing.assert_close(model.image_encoder.blocks[0].attn.qkv.weight, source.blocks[0].attn.qkv.weight)
    out = model(torch.rand(1, 3, 35, 35))
    assert out.shape == (1, 2, 35, 35)
    out.square().mean().backward()
    assert model.image_encoder.neck[0].weight.grad is not None
    model.set_finetuning_stage(True, 4)
    assert not model.image_encoder.blocks[7].attn.qkv.weight.requires_grad
    assert model.image_encoder.blocks[8].attn.qkv.weight.requires_grad
    assert not ({id(p) for p in model.decoder_parameters()} & {id(p) for p in model.finetuning_parameters(4)})
    state.pop('backbone.last_norm.bias')
    torch.save({'state_dict': state}, path)
    with pytest.raises(ValueError, match='keys mismatch'):
        model.load_pretrained_encoder(path)


def test_bundled_sam_global_and_window_attention():
    encoder = ImageEncoderViT(
        img_size=48,
        patch_size=16,
        embed_dim=32,
        depth=2,
        num_heads=4,
        out_chans=16,
        use_rel_pos=True,
        window_size=2,
        global_attn_indexes=(1,),
    )
    image = torch.randn(1, 3, 48, 48, requires_grad=True)
    result = encoder(image)
    assert result.shape == (1, 16, 3, 3)
    result.square().mean().backward()
    assert torch.isfinite(image.grad).all()


@pytest.mark.parametrize('name', ['unet_basic', 'hrnet', 'stacked_hourglass'])
def test_cnn_non_patch_aligned_canvas(name):
    from Heatmaps.model_registry import build_heatmap_model, get_model_kwargs
    from Heatmaps.models import unpack_heatmap_output
    from Heatmaps.train_model import HeatmapModelConfig
    config = HeatmapModelConfig(network_name=name, base_channels=4, depth=1, max_channels=8,
        hrnet_width=4, hrnet_modules=1, hrnet_blocks=1, hourglass_features=16,
        hourglass_stacks=2, hourglass_depth=1, hourglass_blocks=1)
    model = build_heatmap_model(name, 2, 1, 65, **get_model_kwargs(name, config)).eval()
    with torch.no_grad():
        heatmaps, auxiliary = unpack_heatmap_output(model(torch.zeros(1, 1, 65, 65)))
    assert heatmaps.shape == (1, 2, 65, 65)
    assert all(value.shape == heatmaps.shape for value in auxiliary)
