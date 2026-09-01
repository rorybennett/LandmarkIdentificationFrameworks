"""Tests for IPV input normalisation."""

import unittest

import numpy as np
import torch

from IPV.custom_dataset import ToTensor
from IPV.normalisation import ChannelStatistics, IMAGENET_RGB_MEAN, IMAGENET_RGB_STD
from IPV.train_model import QuadrupletConfig, TrainConfig, TrainModel


class DummyDataset:
    def __init__(self, images):
        self.images = images
        self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {'image': self.images[index]}


class IPVNormalisationTests(unittest.TestCase):
    def test_channel_statistics_keep_three_channels_distinct(self):
        samples = np.asarray([
            [[[0.0, 0.2]], [[0.1, 0.4]], [[0.7, 1.0]]],
            [[[0.4, 0.8]], [[0.5, 0.9]], [[0.2, 0.6]]],
        ], dtype=np.float32)
        statistics = ChannelStatistics()
        statistics.update(samples)
        mean, standard_deviation = statistics.finalise()
        expected = np.moveaxis(samples, 1, 0).reshape(3, -1)
        np.testing.assert_allclose(mean, expected.mean(axis=1), rtol=0, atol=1e-7)
        np.testing.assert_allclose(standard_deviation, expected.std(axis=1), rtol=0, atol=1e-7)
        self.assertEqual(len(set(round(value, 6) for value in mean)), 3)

    def test_to_tensor_applies_each_channel_constant(self):
        image = np.asarray([[[[0.3]]], [[[0.5]]], [[[0.9]]]], dtype=np.float32).reshape(1, 3, 1, 1)
        sample = {'image': image, 'sample_name': 'sample', 'coordinates': np.asarray([1, 2]),
                  'labels': np.asarray([0, 1])}
        transformed = ToTensor([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])(sample)
        torch.testing.assert_close(transformed['image'].reshape(3), torch.tensor([2.0, 1.5, 2.0]))

    def test_pretrained_backbone_uses_imagenet_constants(self):
        trainer = TrainModel(
            repetition=1, current_fold=1, num_of_points=1, data_save_path='.', tasks_classes=[[(0, 1)], [(0, 360)]],
            train_config=TrainConfig(batch_size=1, learning_rate=1e-3, max_training_epochs=1, num_workers=0,
                                     normalise_inputs=True),
            quadruplet_config=QuadrupletConfig(network_name='resnet18_pretrained'),
        )
        trainer.input_channels = 3
        training_dataset = DummyDataset([])
        validation_dataset = DummyDataset([])
        trainer.configure_input_normalisation(training_dataset, validation_dataset)
        self.assertEqual(tuple(trainer.normalisation_mean), IMAGENET_RGB_MEAN)
        self.assertEqual(tuple(trainer.normalisation_std), IMAGENET_RGB_STD)
        self.assertEqual(trainer.normalisation_source, 'torchvision_imagenet_pretrained_weights')
        self.assertEqual(trainer.build_normalisation_metadata()['calculated_from'], 'pretrained_weight_recipe')

    def test_untrained_backbone_uses_training_samples_only(self):
        training_images = [
            np.asarray([[[[0.0, 0.2]]], [[[0.1, 0.3]]], [[[0.4, 0.8]]]], dtype=np.float32).reshape(1, 3, 1, 2),
            np.asarray([[[[0.4, 0.6]]], [[[0.5, 0.7]]], [[[0.2, 1.0]]]], dtype=np.float32).reshape(1, 3, 1, 2),
        ]
        trainer = TrainModel(
            repetition=1, current_fold=1, num_of_points=1, data_save_path='.', tasks_classes=[[(0, 1)], [(0, 360)]],
            train_config=TrainConfig(batch_size=1, learning_rate=1e-3, max_training_epochs=1, num_workers=0,
                                     normalise_inputs=True),
            quadruplet_config=QuadrupletConfig(network_name='small_cnn', small_input_stem=False),
        )
        trainer.input_channels = 3
        training_dataset = DummyDataset(training_images)
        validation_dataset = DummyDataset([np.full((1, 3, 1, 2), 100.0, dtype=np.float32)])
        trainer.configure_input_normalisation(training_dataset, validation_dataset)
        expected = np.moveaxis(np.concatenate(training_images, axis=0), 1, 0).reshape(3, -1)
        np.testing.assert_allclose(trainer.normalisation_mean, expected.mean(axis=1), atol=1e-7)
        np.testing.assert_allclose(trainer.normalisation_std, expected.std(axis=1), atol=1e-7)
        self.assertEqual(trainer.build_normalisation_metadata()['calculated_from'], 'training_split_only')


if __name__ == '__main__':
    unittest.main()
