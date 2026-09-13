"""CPU contract tests using a small encoder double; no MedSAM downloads."""
import csv
from types import SimpleNamespace
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from openpyxl import load_workbook
from torch import nn
from torch.nn import functional as F

from Heatmaps.models import vit_medsam as models
from Heatmaps.custom_dataset import HeatmapDataset, HeatmapDatasetConfig
from Heatmaps.heatmap_training_pipeline import parse_args, build_configs, HeatmapTrainingPipeline
from Heatmaps.model_registry import AVAILABLE_MODELS, get_available_model_names
from Heatmaps.train_model import TrainModel, TrainConfig, HeatmapDataConfig, HeatmapModelConfig, HISTORY_FIELDS
from Heatmaps.utils.heatmap_inference_utils import (HeatmapImageInferer, HeatmapImageRecord, load_model_from_checkpoint,
    build_config_from_checkpoint_metadata, run_heatmap_inference_for_records)
from Heatmaps.utils.io_utils import prepare_image, remove_padding


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 8, 1)

    def forward(self, image):
        return self.proj(F.avg_pool2d(image, 16)).permute(0, 2, 3, 1)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(8))
        self.bias = nn.Parameter(torch.zeros(8))

    def forward(self, features):
        return features * self.scale + self.bias


class TinyEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.pos_embed = nn.Parameter(torch.zeros(1, kwargs.get('img_size', 1024)//16, kwargs.get('img_size', 1024)//16, 8))
        self.blocks = nn.ModuleList([TinyBlock() for _ in range(12)])
        self.neck = nn.Conv2d(8, 256, 1)

    def forward(self, image):
        features = self.patch_embed(image) + self.pos_embed
        for block in self.blocks:
            features = block(features)
        return self.neck(features.permute(0, 3, 1, 2))


class SmallHead(nn.Module):
    def __init__(self, points):
        super().__init__()
        self.output = nn.Conv2d(256, points, 1)

    def forward(self, features):
        return F.interpolate(self.output(features), scale_factor=16, mode='bilinear', align_corners=False)


def small_model(**kwargs):
    model = models.MedSAMHeatmap(**kwargs)
    model.decoder = SmallHead(kwargs['num_of_points'])
    return model


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def encoder_double(monkeypatch):
    monkeypatch.setattr(models, 'ImageEncoderViT', TinyEncoder)
    monkeypatch.setitem(AVAILABLE_MODELS['vit-medsam'], 'builder', small_model)


@pytest.fixture
def pretrained(tmp_path, encoder_double):
    path = tmp_path / 'medsam_vit_b.pth'
    torch.manual_seed(10)
    state = {'image_encoder.' + name: value for name, value in TinyEncoder().state_dict().items()}
    state['prompt_encoder.unused'] = torch.zeros(1)
    state['mask_decoder.unused'] = torch.zeros(1)
    torch.save(state, path)
    return path


def configs(tmp_path, pretrained):
    fold = tmp_path / 'folds' / 'repetition_1'
    fold.mkdir(parents=True, exist_ok=True)
    (fold / 'training_f1.txt').write_text('one\n')
    (fold / 'val_f1.txt').write_text('two\n')
    (fold / 'training_f2.txt').write_text('two\n')
    (fold / 'val_f2.txt').write_text('one\n')
    (fold / 'training_fall.txt').write_text('one\ntwo\n')
    (fold / 'val_fall.txt').write_text('one\ntwo\n')
    marks = tmp_path / 'marks.txt'
    marks.write_text('one.png (10, 8)\ntwo.png (12, 7)\n')
    for name in ('one', 'two'):
        cv2.imwrite(str(tmp_path / f'{name}.png'), np.arange(16 * 32, dtype=np.uint8).reshape(16, 32))
    data = HeatmapDataConfig(1, 1, 'test', 1, tmp_path / 'folds', marks, tmp_path, 1024, oversampling_factor=2, enforce_greyscale=True)
    train = TrainConfig(batch_size=1, learning_rate=1e-4, max_training_epochs=3, num_workers=0, freeze_encoder_epochs=1,
                        early_stop_warmup_epochs=2, pretrained_checkpoint=str(pretrained), save_validation_predictions=True, lr_schedule='step', lr_step_size=1)
    return data, train, HeatmapModelConfig(network_name='vit-medsam', decoder_channels=16)


