"""Intensity augmentation correctness and deterministic sampling."""
import unittest
import numpy as np
from Heatmaps import heatmap_transforms as h
from Heatmaps.utils.verify_transforms import make_transform


class IntensityTests(unittest.TestCase):
    def test_known_factors_preserve_background_alpha_points_and_input(self):
        image = np.array([[[0., .2, .6]]]*3 + [[[.3, .4, .5]]], dtype=np.float32)
        original = image.copy()
        points = np.array([[1., 0.]], dtype=np.float32)
        cases = [(h.RandomGlobalGain((2., 2.), 1.), [0., .4, 1.]),
                 (h.RandomContrast((2., 2.), 1.), [0., 0., .8]),
                 (h.RandomGamma((2., 2.), 1.), [0., .04, .36])]
        for transform, expected in cases:
            result, returned_points = transform(image, points)
            for channel in result[:3]:
                np.testing.assert_allclose(channel[0], expected, atol=1e-7)
            np.testing.assert_array_equal(result[3], image[3])
            np.testing.assert_array_equal(image, original)
            self.assertIs(returned_points, points)
            self.assertTrue(transform.last_params['applied'])
            self.assertEqual(transform.last_params['factor'], 2.)

    def test_skip_empty_and_seeded_sampling(self):
        for cls in (h.RandomGlobalGain, h.RandomContrast, h.RandomGamma):
            image = np.full((1, 4, 4), .5, dtype=np.float32)
            transform = cls(probability=0.)
            result, _ = transform(image, [])
            np.testing.assert_array_equal(result, image)
            self.assertFalse(transform.last_params['applied'])
            transform = cls(probability=1.)
            np.random.seed(123)
            first, _ = transform(image, [])
            params = transform.last_params.copy()
            np.random.seed(123)
            second, _ = transform(image, [])
            np.testing.assert_array_equal(first, second)
            self.assertEqual(params, transform.last_params)
            result, _ = transform(np.zeros_like(image), [])
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(np.all(result == 0))
            with self.assertRaises(ValueError):
                cls(factor_range=(-1, 2))(image, [])

    def test_policy_matches_pipeline_and_previews(self):
        transforms = h.get_default_heatmap_transforms().transforms
        self.assertEqual([type(t).__name__ for t in transforms], h.get_augmentation_policy()['transform_order'])
        self.assertEqual([type(t).__name__ for t in transforms], [t['name'] for t in h.get_augmentation_policy()['transforms']])
        for name in ('gain', 'contrast', 'gamma'):
            self.assertEqual(make_transform(name).probability, 1.)


if __name__ == '__main__':
    unittest.main()
