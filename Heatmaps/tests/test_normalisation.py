"""Tests for Heatmaps input normalisation."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from Heatmaps.custom_dataset import HeatmapDataset, HeatmapDatasetConfig
from Heatmaps.normalisation import ChannelStatistics, normalise_channel_first


class HeatmapNormalisationTests(unittest.TestCase):
    def test_channel_statistics_keep_three_channels_distinct(self):
        image = np.asarray([
            [[0.0, 0.2], [0.4, 0.6]],
            [[0.1, 0.4], [0.7, 1.0]],
            [[0.8, 0.6], [0.4, 0.2]],
        ], dtype=np.float32)
        statistics = ChannelStatistics()
        statistics.update(image)
        mean, standard_deviation = statistics.finalise()
        np.testing.assert_allclose(mean, image.reshape(3, -1).mean(axis=1), atol=1e-7)
        np.testing.assert_allclose(standard_deviation, image.reshape(3, -1).std(axis=1), atol=1e-7)
        self.assertEqual(len(set(round(value, 6) for value in mean)), 3)

    def test_dataset_statistics_use_unaugmented_training_records_only(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = np.asarray([[[0, 20, 200], [40, 80, 160]], [[60, 100, 120], [80, 140, 80]]], dtype=np.uint8)
            second = np.asarray([[[20, 40, 180], [60, 100, 140]], [[80, 120, 100], [100, 160, 60]]], dtype=np.uint8)
            first_path = root / 'first.png'
            second_path = root / 'second.png'
            self.assertTrue(cv2.imwrite(str(first_path), cv2.cvtColor(first, cv2.COLOR_RGB2BGR)))
            self.assertTrue(cv2.imwrite(str(second_path), cv2.cvtColor(second, cv2.COLOR_RGB2BGR)))

            dataset = HeatmapDataset.__new__(HeatmapDataset)
            dataset.config = HeatmapDatasetConfig(
                repetition=1, fold=1, split_name='training', num_of_points=1, fold_lists_path=root,
                mark_list_file=root / 'marks.txt', image_data_dir=root, image_size=(2, 2), heatmap_sigma=1.0,
                input_channels=3,
            )
            dataset.records = [{'image_path': first_path}, {'image_path': second_path}]
            mean, standard_deviation = dataset.calculate_normalisation_statistics()
            expected = np.concatenate([np.moveaxis(first.astype(np.float32) / 255.0, -1, 0),
                                       np.moveaxis(second.astype(np.float32) / 255.0, -1, 0)], axis=1).reshape(3, -1)
            np.testing.assert_allclose(mean, expected.mean(axis=1), atol=1e-7)
            np.testing.assert_allclose(standard_deviation, expected.std(axis=1), atol=1e-7)

    def test_normalisation_applies_three_distinct_channel_constants(self):
        image = np.asarray([[[0.3]], [[0.5]], [[0.9]]], dtype=np.float32)
        normalised = normalise_channel_first(image, [0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(normalised.reshape(3), [2.0, 1.5, 2.0], atol=1e-6)

    def test_one_channel_normalisation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly 3 channels'):
            normalise_channel_first(np.zeros((1, 2, 2), dtype=np.float32), [0.1, 0.2, 0.3], [1.0, 1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
