"""Regression tests for enforced three-channel greyscale."""
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from IPV.greyscale import to_three_channel_greyscale


class GreyscaleTests(unittest.TestCase):
    def test_rgb_luminance_and_replication(self):
        rgb = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
        result = to_three_channel_greyscale(rgb)
        np.testing.assert_allclose(result[0, :, 0], [0.299, 0.587, 0.114], atol=1e-7)
        np.testing.assert_array_equal(result[..., 0], result[..., 1])
        np.testing.assert_array_equal(result[..., 0], result[..., 2])

    def test_single_channel_rgba_and_idempotence(self):
        grey = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
        rgb = to_three_channel_greyscale(grey)
        np.testing.assert_array_equal(rgb[..., 0], grey)
        np.testing.assert_array_equal(to_three_channel_greyscale(grey[..., None]), rgb)
        np.testing.assert_array_equal(to_three_channel_greyscale(rgb), rgb)
        np.testing.assert_array_equal(to_three_channel_greyscale(np.dstack([rgb, grey * 0])), rgb)

    def test_pretrained_greyscale_uses_equal_training_statistics(self):
        from types import SimpleNamespace
        from IPV.train_model import TrainModel
        class Dataset:
            transform = None
            def __len__(self): return 1
            def __getitem__(self, index):
                image = to_three_channel_greyscale(np.array([[0., .3], [.6, 1.]], dtype=np.float32))
                return {'image': np.moveaxis(image, -1, 0)[None]}
        trainer = TrainModel.__new__(TrainModel)
        trainer.train_config = SimpleNamespace(normalise_inputs=True, enforce_greyscale=True)
        trainer.quadruplet_config = SimpleNamespace(network_name='resnet18_pretrained')
        trainer.input_channels = 3
        trainer.configure_input_normalisation(Dataset(), Dataset())
        self.assertEqual(trainer.normalisation_source, 'training_split_patches')
        self.assertEqual(len(set(trainer.normalisation_mean)), 1)
        self.assertEqual(len(set(trainer.normalisation_std)), 1)
        self.assertEqual(trainer.build_normalisation_metadata()['calculated_from'], 'training_split_only')
