"""Regression checks for the sole longest-edge, square-canvas contract."""
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from Heatmaps.custom_dataset import HeatmapDataset, HeatmapDatasetConfig
from Heatmaps.heatmap_training_pipeline import parse_args
from Heatmaps.model_registry import build_heatmap_model, get_model_kwargs
from Heatmaps.models import unpack_heatmap_output
from Heatmaps.train_model import TrainModel, TrainConfig, HeatmapDataConfig, HeatmapModelConfig
from Heatmaps.utils.io_utils import (LETTERBOX_POLICY, letterbox_geometry, prepare_image, remove_padding,
    resize_channel_first, scale_points, scale_points_to_original, heatmaps_to_points, valid_content_mask)
from Heatmaps.utils.heatmap_inference_utils import (HeatmapImageInferer, HeatmapImageRecord,
    load_model_from_checkpoint, build_config_from_checkpoint_metadata, run_heatmap_inference_for_records)
from Heatmaps.utils.visualisation_utils import create_combined_heatmap_overlay
import test_inference as inference_fixtures


class LetterboxTests(unittest.TestCase):
    def test_geometry_and_coordinate_roundtrip_for_rectangles_and_rounding(self):
        for original in ((4, 8), (8, 4), (7, 13), (13, 7), (1, 100), (9, 9)):
            size = 17
            h, w, top, left = letterbox_geometry(original, size)
            self.assertEqual(max(h, w), size)
            canvas = resize_channel_first(np.ones((3, *original), np.float32), size)
            self.assertEqual(canvas.shape, (3, size, size))
            self.assertAlmostEqual(float(canvas.sum()), 3 * h * w, places=3)
            np.testing.assert_allclose(remove_padding(canvas, original, size), 1, atol=1e-6)
            points = np.array([[0, 0], [original[1] - 1, original[0] - 1],
                               [(original[1] - 1) / 2, (original[0] - 1) / 2]], np.float32)
            mapped = scale_points(points, original, size)
            restored = scale_points_to_original(torch.tensor(mapped)[None], torch.tensor([original]), size)
            np.testing.assert_allclose(restored[0], points, atol=1e-5)
        self.assertEqual(letterbox_geometry((4, 8), 8), (4, 8, 2, 0))
        np.testing.assert_allclose(scale_points([[2, 1]], (4, 8), 8), [[2, 3]])
        with self.assertRaisesRegex(ValueError, 'one positive integer'):
            resize_channel_first(np.ones((1, 8, 8)), (8, 8))

    def test_padding_cannot_win_argmax_even_with_negative_content(self):
        originals = torch.tensor([[4, 8], [8, 4]])
        mask = valid_content_mask(originals, 8)
        maps = torch.full((2, 1, 8, 8), 10000.0).masked_fill(mask, -10)
        maps[0, 0, 3, 2] = -1
        maps[1, 0, 2, 3] = -1
        points = heatmaps_to_points(maps, originals, 8)
        restored = scale_points_to_original(points, originals, 8)
        torch.testing.assert_close(restored, torch.tensor([[[2., 1.]], [[1., 2.]]]))

    def test_all_losses_and_auxiliary_heads_ignore_padding_and_weight_images_equally(self):
        trainer = object.__new__(TrainModel)
        trainer.model_config = SimpleNamespace(auxiliary_loss_weight=0.5)
        mask = valid_content_mask([[2, 8], [8, 8]], 8)
        targets = torch.zeros(2, 2, 8, 8)
        for name in ('mse', 'weighted_mse', 'smooth_l1', 'bce_logits'):
            trainer.train_config = TrainConfig(batch_size=2, learning_rate=0.001, max_training_epochs=1, loss_name=name)
            criterion = trainer.build_criterion()
            values = torch.ones_like(targets)
            values[1] = 2
            values = values.masked_fill(~mask, 10000).requires_grad_()
            loss = trainer.calculate_model_loss(values, [values], targets, criterion, mask)
            first = criterion(torch.ones(1), torch.zeros(1)).mean()
            second = criterion(torch.full((1,), 2.), torch.zeros(1)).mean()
            torch.testing.assert_close(loss, (first + second) / 2 * 1.5)
            loss.backward()
            self.assertTrue(torch.all(values.grad.masked_select(~mask.expand_as(values)) == 0))
            self.assertTrue(torch.isfinite(values.grad).all())

    def test_dataset_inference_normalisation_and_targets_agree(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fold = root / 'folds' / 'repetition_1'
            fold.mkdir(parents=True)
            (fold / 'training_f1.txt').write_text('sample\n')
            (root / 'marks.txt').write_text('sample.png (2, 1)\n')
            image = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
            cv2.imwrite(str(root / 'sample.png'), image)
            config = HeatmapDatasetConfig(1, 1, 'training', 1, root / 'folds', root / 'marks.txt', root,
                                          8, 1., input_channels=3)
            dataset = HeatmapDataset(config)
            mean, std = dataset.calculate_normalisation_statistics()
            source = np.moveaxis(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), -1, 0) / 255.
            np.testing.assert_allclose(mean, source.mean(axis=(1, 2)), atol=1e-7)
            config.normalisation_mean, config.normalisation_std = mean, std
            sample = dataset[0]
            inferer = object.__new__(HeatmapImageInferer)
            inferer.config = SimpleNamespace(input_channels=3, enforce_greyscale=False, image_size=8, num_points=1,
                                             normalisation_mean=mean, normalisation_std=std)
            prepared = inferer.prepare_record(HeatmapImageRecord('sample', root / 'sample.png', [(2, 1)]))
            torch.testing.assert_close(sample['image'], prepared['image'])
            self.assertTrue(torch.all(sample['image'][:, :2] == 0))
            self.assertTrue(torch.all(sample['heatmaps'][:, :2] == 0))
            self.assertEqual(float(sample['heatmaps'][0, 3, 2]), 1.)

    def test_overlays_crop_before_resizing(self):
        display = np.full((4, 8, 3), 60, np.uint8)
        clean = np.zeros((1, 8, 8), np.float32)
        clean[0, 3, 2] = 1
        dirty = clean.copy()
        dirty[:, :2] = 1000
        dirty[:, 6:] = 1000
        np.testing.assert_array_equal(create_combined_heatmap_overlay(display, clean),
                                      create_combined_heatmap_overlay(display, dirty))

    def test_all_architectures_produce_square_maps_for_letterboxed_inputs(self):
        image = torch.from_numpy(prepare_image(np.ones((1, 32, 64), np.float32), 64))[None]
        for name in ('unet_basic', 'hrnet', 'stacked_hourglass', 'vitpose'):
            config = inference_fixtures.StandaloneInferenceTests.model_config(name)
            model = build_heatmap_model(name, 2, 1, 64, **get_model_kwargs(name, config)).eval()
            with torch.no_grad():
                maps, auxiliary = unpack_heatmap_output(model(image))
            self.assertEqual(tuple(maps.shape), (1, 2, 64, 64))
            for value in auxiliary:
                self.assertEqual(tuple(value.shape), tuple(maps.shape))

    def test_old_checkpoint_and_two_value_cli_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = inference_fixtures.StandaloneInferenceTests().write_checkpoint(Path(td), 'unet_basic')
            checkpoint = torch.load(path, weights_only=False)
            checkpoint['metadata']['preprocessing'].pop('resize')
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'older checkpoints are unsupported'):
                load_model_from_checkpoint(path, 'cpu')
        args = ['train', '1', '1', 'task', 'true', 'false', '--image-size', '16',
                '--run-dir', '.', '--num-points', '1', '--fold-lists-path', '.', '--mark-list-file', 'marks.txt', '--image-data-dir', '.']
        with patch('sys.argv', args):
            self.assertEqual(parse_args().image_size, 16)
        with patch('sys.argv', args + ['16']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_rectangular_training_export_and_standalone_inference_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fold = root / 'folds' / 'repetition_1'
            fold.mkdir(parents=True)
            (fold / 'training_f1.txt').write_text('train\n')
            (fold / 'val_f1.txt').write_text('val\n')
            (root / 'marks.txt').write_text('train.png (2, 3)\nval.png (3, 2)\n')
            cv2.imwrite(str(root / 'train.png'), np.arange(16 * 8, dtype=np.uint8).reshape(16, 8))
            cv2.imwrite(str(root / 'val.png'), np.arange(8 * 16, dtype=np.uint8).reshape(8, 16))
            trainer = TrainModel(
                HeatmapDataConfig(1, 1, 'task', 1, root / 'folds', root / 'marks.txt', root, 16, 1.),
                TrainConfig(batch_size=1, learning_rate=0.001, max_training_epochs=1, num_workers=0),
                HeatmapModelConfig(base_channels=2, depth=1, max_channels=4), root / 'outputs', torch.device('cpu'))
            trainer.train()
            checkpoint = root / 'outputs' / 'model_best_validation_loss.pth'
            loaded = load_model_from_checkpoint(checkpoint, 'cpu')
            self.assertEqual(loaded.metadata['raw_checkpoint_metadata']['preprocessing']['resize'], LETTERBOX_POLICY)
            config = build_config_from_checkpoint_metadata(loaded.metadata, root / 'inference', save_raw_heatmaps=True)
            result = run_heatmap_inference_for_records(loaded.model, config,
                [HeatmapImageRecord('val', root / 'val.png', [(3, 2)])], 'cpu')[0]
            with open(root / 'outputs' / 'validation_results' / 'validation_endpoints.csv') as f:
                row = next(csv.DictReader(f))
            for field in ('pred_x', 'pred_y', 'error_px'):
                self.assertAlmostEqual(float(row[field]), result['endpoint_rows'][0][field], places=5)
            raw = next((root / 'inference' / 'inference_raw_heatmaps').glob('*.npy'))
            self.assertEqual(np.load(raw).shape, (1, 8, 16))


if __name__ == '__main__':
    unittest.main()
