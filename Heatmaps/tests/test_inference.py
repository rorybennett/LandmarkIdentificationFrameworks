"""Regression tests for standalone Heatmaps inference."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from Heatmaps.model_registry import build_heatmap_model, get_model_kwargs
from Heatmaps.models import unpack_heatmap_output
from Heatmaps.train_model import HeatmapModelConfig
from Heatmaps.utils.heatmap_inference_utils import (build_config_from_checkpoint_metadata, build_image_records,
                                                    load_inference_image_as_float, load_model_from_checkpoint,
                                                    run_heatmap_inference_for_records)
from Heatmaps.utils.io_utils import load_image_as_float


class StandaloneInferenceTests(unittest.TestCase):
    image_size = (16, 16)
    num_points = 2
    input_channels = 1

    @staticmethod
    def model_config(network_name):
        return HeatmapModelConfig(
            network_name=network_name,
            base_channels=2,
            depth=1,
            max_channels=4,
            hrnet_width=4,
            hrnet_modules=1,
            hrnet_blocks=1,
            hourglass_features=16,
            hourglass_stacks=1,
            hourglass_depth=1,
            hourglass_blocks=1,
            vit_patch_size=8,
            vit_embed_dim=16,
            vit_depth=1,
            vit_heads=2,
            vit_decoder_channels=16,
        )

    def write_checkpoint(self, root, network_name):
        model_config = self.model_config(network_name)
        model_kwargs = get_model_kwargs(network_name, model_config)
        model = build_heatmap_model(network_name=network_name, num_of_points=self.num_points,
                                    input_channels=self.input_channels, image_size=self.image_size, **model_kwargs)
        init_args = {'num_of_points': self.num_points, 'input_channels': self.input_channels, **model_kwargs}

        if network_name == 'vitpose':
            init_args['image_size'] = list(self.image_size)

        metadata = {
            'schema': 'heatmap_checkpoint_metadata',
            'schema_version': 3,
            'task': {'name': 'inference_test', 'repetition': 2, 'fold': 3, 'num_points': self.num_points},
            'model': {'registry_name': network_name, 'init_args': init_args},
            'data': {'repetition': 2, 'fold': 3},
            'preprocessing': {
                'image_size': {'height': self.image_size[0], 'width': self.image_size[1]},
                'input_channels': self.input_channels,
            },
            'inference': {'heatmap_to_point': 'argmax', 'scale_back_to_original': True},
            'checkpoint': {'type': 'best_validation_loss'},
        }
        checkpoint_path = root / f'{network_name}.pth'
        torch.save({'state_dict': model.state_dict(), 'metadata': metadata,
                    'checkpoint_type': 'best_validation_loss'}, checkpoint_path)
        return checkpoint_path

    def test_loader_reconstructs_every_registered_model(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)

            for network_name in ('unet_basic', 'hrnet', 'stacked_hourglass', 'vitpose'):
                with self.subTest(network_name=network_name):
                    checkpoint_path = self.write_checkpoint(root, network_name)
                    loaded = load_model_from_checkpoint(checkpoint_path, device='cpu')
                    model_output = loaded.model(torch.zeros((1, self.input_channels, *self.image_size)))
                    heatmaps, _ = unpack_heatmap_output(model_output)
                    self.assertEqual(tuple(heatmaps.shape), (1, self.num_points, *self.image_size))
                    self.assertEqual(loaded.metadata['network_name'], network_name)

    def test_inference_replicates_greyscale_images_for_rgb_models(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            image_path = Path(temporary_dir) / 'greyscale.png'
            image = np.arange(12 * 10, dtype=np.uint8).reshape(12, 10)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            with self.assertRaisesRegex(ValueError, r'expected 3 channel\(s\), but found 1'):
                load_image_as_float(image_path, input_channels=3)

            inference_image = load_inference_image_as_float(image_path, input_channels=3)
            self.assertEqual(inference_image.shape, (3, 12, 10))
            self.assertTrue(np.array_equal(inference_image[0], inference_image[1]))
            self.assertTrue(np.array_equal(inference_image[1], inference_image[2]))

    def test_inference_writes_predictions_visuals_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image_path = root / 'patient.png'
            mark_list = root / 'marks.txt'
            output_dir = root / 'outputs'
            image = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            mark_list.write_text('patient.png (4, 5) (20, 22)\n', encoding='utf-8')
            checkpoint_path = self.write_checkpoint(root, 'unet_basic')
            loaded = load_model_from_checkpoint(checkpoint_path, device='cpu')
            config = build_config_from_checkpoint_metadata(
                loaded.metadata,
                output_dir=output_dir,
                batch_size=1,
                save_raw_heatmaps=True,
                clear_cuda_cache_between_batches=False,
                checkpoint_path=checkpoint_path,
            )
            records = build_image_records(image_path, num_points=self.num_points, mark_list_path=mark_list)
            results = run_heatmap_inference_for_records(loaded.model, config, records, device='cpu')

            self.assertEqual(len(results), 1)
            self.assertEqual(len(results[0]['endpoint_rows']), self.num_points)
            self.assertIsNotNone(results[0]['summary']['mean_error_px'])
            expected_files = {
                'inference_summary.xlsx',
                'inference_image_summary.csv',
                'inference_endpoints.csv',
                'inference_predictions.csv',
                'inference_heatmap_overlays',
                'inference_point_overlays',
                'inference_raw_heatmaps',
                'inference_logs',
            }
            self.assertTrue(expected_files.issubset({path.name for path in output_dir.iterdir()}))
            raw_heatmaps = np.load(output_dir / 'inference_raw_heatmaps' / 'patient_inference_heatmaps.npy')
            self.assertEqual(raw_heatmaps.shape, (self.num_points, *self.image_size))
            self.assertTrue((output_dir / 'inference_logs' / 'inference_run_metadata.json').is_file())


if __name__ == '__main__':
    unittest.main()
