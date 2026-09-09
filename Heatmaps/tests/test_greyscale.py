"""Regression tests for enforced three-channel greyscale."""
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from Heatmaps.greyscale import to_three_channel_greyscale


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

    def test_training_inference_and_normalisation_match(self):
        from Heatmaps.utils.io_utils import load_image_as_float, prepare_image, remove_padding
        from Heatmaps.utils.heatmap_inference_utils import load_inference_image_as_float
        from Heatmaps.normalisation import ChannelStatistics
        from Heatmaps.heatmap_transforms import get_default_heatmap_transforms
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'watermark.png'
            image = np.full((12, 24, 3), 100, dtype=np.uint8)
            image[2:5, 2:5] = [0, 0, 255]
            cv2.imwrite(str(path), image)
            original = load_image_as_float(path, 3)
            self.assertFalse(np.array_equal(original[0], original[2]))
            training = load_image_as_float(path, 3, enforce_greyscale=True)
            inference = load_inference_image_as_float(path, 3, enforce_greyscale=True)
            np.testing.assert_array_equal(training, inference)
            stats = ChannelStatistics()
            stats.update(training)
            mean, std = stats.finalise()
            self.assertEqual(len(set(mean)), 1)
            self.assertEqual(len(set(std)), 1)
            transformed, points = get_default_heatmap_transforms()(training, np.array([[8., 5.]], dtype=np.float32))
            np.testing.assert_array_equal(transformed[0], transformed[1])
            np.testing.assert_array_equal(transformed[1], transformed[2])
            canvas = prepare_image(training, 32, mean, std)
            np.testing.assert_array_equal(canvas[0], canvas[1])
            self.assertTrue(np.all(canvas[:, :8] == 0))
