"""
Default stochastic transforms for heatmap landmark training.

Edit this file to change the oversampling augmentation policy. The defaults are intentionally conservative for
ultrasound images and preserve greyscale RGB images by applying intensity transforms consistently across colour
channels.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

AFFINE_DEGREES = 30
AFFINE_SHEAR = 15
AFFINE_TRANSLATE = (0.1, 0.1)
AFFINE_SCALE = (0.8, 1.1)
AFFINE_MAX_ATTEMPTS = 10000
RANDOM_ERASING_PROBABILITY = 0.5
RANDOM_ERASING_SCALE = (0.02, 0.08)
RANDOM_ERASING_RATIO = (0.3, 3.3)
GAUSSIAN_NOISE_MEAN = 0.0
GAUSSIAN_NOISE_SIGMA = 0.1
GAUSSIAN_NOISE_CLIP = True
GAUSSIAN_BLUR_KERNEL_SIZE = 5


@dataclass
class Compose:
    transforms: list

    def __call__(self, image, points):
        """Apply each transform and store the sampled parameters from every stage."""
        self.last_params = []

        for transform in self.transforms:
            image, points = transform(image=image, points=points)
            self.last_params.append(get_last_params(transform))

        return image, points


@dataclass
class RandomErasing:
    probability: float = RANDOM_ERASING_PROBABILITY
    scale: tuple[float, float] = RANDOM_ERASING_SCALE
    ratio: tuple[float, float] = RANDOM_ERASING_RATIO
    fill_value: float = 0.0
    blend_strength_range: tuple[float, float] = (0.35, 0.75)

    def __call__(self, image, points):
        self.last_params = {'transform': 'random_erasing', 'applied': False, 'probability': float(self.probability), 'reason': 'skipped_probability'}

        if np.random.random() >= self.probability:
            return image, points

        _, height, width = image.shape
        area = float(height * width)
        image = image.copy().astype(np.float32)
        self.last_params = {'transform': 'random_erasing', 'applied': False, 'probability': float(self.probability), 'reason': 'no_valid_ellipse'}

        for attempt in range(1, 11):
            target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = math.exp(np.random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))

            semi_axis_x = int(round(math.sqrt((target_area * aspect_ratio) / math.pi)))
            semi_axis_y = int(round(math.sqrt(target_area / (aspect_ratio * math.pi))))

            if semi_axis_x < 1 or semi_axis_y < 1:
                continue

            if (2 * semi_axis_x) >= width or (2 * semi_axis_y) >= height:
                continue

            centre_x = np.random.randint(semi_axis_x, width - semi_axis_x)
            centre_y = np.random.randint(semi_axis_y, height - semi_axis_y)
            blend_strength = float(np.random.uniform(self.blend_strength_range[0], self.blend_strength_range[1]))

            y_grid, x_grid = np.ogrid[:height, :width]
            ellipse_distance = ((x_grid - centre_x) / float(semi_axis_x)) ** 2 + ((y_grid - centre_y) / float(semi_axis_y)) ** 2
            ellipse_mask = ellipse_distance <= 1.0

            blend_mask = np.zeros((height, width), dtype=np.float32)
            blend_mask[ellipse_mask] = blend_strength

            background = np.full_like(image, fill_value=self.fill_value, dtype=np.float32)
            image = image * (1.0 - blend_mask[np.newaxis, :, :]) + background * blend_mask[np.newaxis, :, :]

            self.last_params = {
                'transform': 'random_erasing',
                'applied': True,
                'probability': float(self.probability),
                'attempts': int(attempt),
                'centre_x': int(centre_x),
                'centre_y': int(centre_y),
                'semi_axis_x': int(semi_axis_x),
                'semi_axis_y': int(semi_axis_y),
                'fill_value': float(self.fill_value),
                'blend_strength': float(blend_strength),
            }
            break

        return image.astype(np.float32), points


@dataclass
class RandomAffine:
    degrees: float = AFFINE_DEGREES
    shear: float = AFFINE_SHEAR
    translate: tuple[float, float] = AFFINE_TRANSLATE
    scale: tuple[float, float] = AFFINE_SCALE
    max_attempts: int = AFFINE_MAX_ATTEMPTS

    def __call__(self, image, points):
        """Apply a sampled affine transform only when every transformed landmark remains inside the image."""
        _, height, width = image.shape
        points = np.asarray(points, dtype=np.float32)
        max_attempts = int(self.max_attempts)

        if max_attempts < 1:
            raise ValueError(f'max_attempts must be at least 1. Got: {self.max_attempts}')

        for attempt in range(1, max_attempts + 1):
            matrix = self.sample_matrix(width=width, height=height)
            transformed_points = transform_points(points=points, matrix=matrix)

            if points_inside_image(points=transformed_points, width=width, height=height):
                self.last_params['attempts'] = int(attempt)
                return warp_image(image=image, matrix=matrix), transformed_points.astype(np.float32)

        raise RuntimeError(f'No valid affine transform was found after {max_attempts} attempts. Reduce affine ranges or inspect landmarks near the image border.')

    def sample_matrix(self, width, height):
        """Sample an affine source-to-destination matrix around the image centre."""
        angle = np.random.uniform(-self.degrees, self.degrees)
        shear_x = np.random.uniform(-self.shear, self.shear)
        scale_value = np.random.uniform(self.scale[0], self.scale[1])
        translate_x = np.random.uniform(-self.translate[0], self.translate[0]) * width
        translate_y = np.random.uniform(-self.translate[1], self.translate[1]) * height
        centre_x = (width - 1) / 2.0
        centre_y = (height - 1) / 2.0
        matrix = translation_matrix(translate_x, translate_y) @ translation_matrix(centre_x, centre_y) @ rotation_matrix(angle) @ shear_matrix(shear_x) @ scale_matrix(
            scale_value) @ translation_matrix(-centre_x, -centre_y)
        self.last_params = {
            'transform': 'random_affine',
            'angle_degrees': float(angle),
            'shear_x_degrees': float(shear_x),
            'scale': float(scale_value),
            'translate_x_pixels': float(translate_x),
            'translate_y_pixels': float(translate_y),
            'matrix': matrix.tolist(),
        }
        return matrix




@dataclass
class GaussianNoise:
    mean: float = GAUSSIAN_NOISE_MEAN
    sigma: float = GAUSSIAN_NOISE_SIGMA
    clip: bool = GAUSSIAN_NOISE_CLIP
    preserve_greyscale_rgb: bool = True

    def __call__(self, image, points):
        """Add Gaussian noise while preserving equal RGB channels for greyscale RGB ultrasound images."""
        image = image.copy()
        channels, height, width = image.shape
        colour_channels = min(channels, 3)

        if self.preserve_greyscale_rgb:
            noise = np.random.normal(self.mean, self.sigma, size=(1, height, width)).astype(np.float32)
            image[:colour_channels] = image[:colour_channels] + noise
        else:
            noise = np.random.normal(self.mean, self.sigma, size=(colour_channels, height, width)).astype(np.float32)
            image[:colour_channels] = image[:colour_channels] + noise

        if self.clip:
            image[:colour_channels] = np.clip(image[:colour_channels], 0.0, 1.0)

        self.last_params = {
            'transform': 'gaussian_noise',
            'mean': float(self.mean),
            'sigma': float(self.sigma),
            'clip': bool(self.clip),
            'preserve_greyscale_rgb': bool(self.preserve_greyscale_rgb),
            'sampled_noise_mean': float(np.mean(noise)),
            'sampled_noise_std': float(np.std(noise)),
        }
        return image.astype(np.float32), points


@dataclass
class GaussianBlur:
    kernel_size: int = GAUSSIAN_BLUR_KERNEL_SIZE

    def __call__(self, image, points):
        """Blur image colour channels without moving landmarks or changing an alpha channel."""
        kernel_size = make_odd_kernel_size(self.kernel_size)
        colour_channels = min(image.shape[0], 3)
        blurred_channels = [cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0) for channel in image[:colour_channels]]

        if image.shape[0] > colour_channels:
            blurred_channels.extend(channel.copy() for channel in image[colour_channels:])

        self.last_params = {'transform': 'gaussian_blur', 'kernel_size': int(kernel_size)}
        return np.stack(blurred_channels, axis=0).astype(np.float32), points



def get_last_params(transform):
    """Return the last sampled parameters from a transform in a consistent format."""
    return transform.last_params


def get_default_heatmap_transforms():
    """Return task-agnostic default oversampling transforms for heatmap landmark training."""
    return Compose([
        RandomAffine(),
        GaussianNoise(),
        GaussianBlur(),
    ])


def get_augmentation_policy():
    """Return the default augmentation policy recorded in checkpoints."""
    return {
        'name': 'default_heatmap_oversampling_v5',
        'random_source': 'numpy.random seeded by training random_seed and DataLoader worker seeds',
        'transform_order': ['RandomAffine', 'GaussianNoise', 'GaussianBlur'],
        'transforms': [
            {'name': 'RandomAffine', 'degrees': float(AFFINE_DEGREES), 'shear': float(AFFINE_SHEAR), 'translate': tuple(float(value) for value in AFFINE_TRANSLATE),
             'scale': tuple(float(value) for value in AFFINE_SCALE), 'max_attempts': int(AFFINE_MAX_ATTEMPTS)},
            {'name': 'GaussianNoise', 'mean': float(GAUSSIAN_NOISE_MEAN), 'sigma': float(GAUSSIAN_NOISE_SIGMA), 'clip': bool(GAUSSIAN_NOISE_CLIP),
             'preserve_greyscale_rgb': True},
            {'name': 'GaussianBlur', 'kernel_size': int(GAUSSIAN_BLUR_KERNEL_SIZE)},
        ],
        'available_not_default': [
            {'name': 'RandomErasing', 'probability': float(RANDOM_ERASING_PROBABILITY), 'scale': tuple(float(value) for value in RANDOM_ERASING_SCALE),
             'ratio': tuple(float(value) for value in RANDOM_ERASING_RATIO)}
        ],
    }


def make_odd_kernel_size(kernel_size):
    """Return a positive odd OpenCV kernel size."""
    kernel_size = max(1, int(kernel_size))
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def translation_matrix(x, y):
    """Return a homogeneous translation matrix."""
    return np.asarray([[1.0, 0.0, float(x)], [0.0, 1.0, float(y)], [0.0, 0.0, 1.0]], dtype=np.float32)


def rotation_matrix(angle_degrees):
    """Return a homogeneous rotation matrix."""
    angle_radians = math.radians(float(angle_degrees))
    cos_value = math.cos(angle_radians)
    sin_value = math.sin(angle_radians)
    return np.asarray([[cos_value, -sin_value, 0.0], [sin_value, cos_value, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def shear_matrix(shear_degrees):
    """Return a homogeneous x-shear matrix."""
    return np.asarray([[1.0, math.tan(math.radians(float(shear_degrees))), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def scale_matrix(scale_value):
    """Return a homogeneous isotropic scale matrix."""
    return np.asarray([[float(scale_value), 0.0, 0.0], [0.0, float(scale_value), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def transform_points(points, matrix):
    """Apply a homogeneous affine matrix to xy landmark points."""
    if len(points) == 0:
        return points.astype(np.float32)

    homogeneous_points = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (matrix @ homogeneous_points.T).T[:, :2]


def points_inside_image(points, width, height):
    """Return True when every point is inside the image bounds."""
    if len(points) == 0:
        return True

    x_inside = np.logical_and(points[:, 0] >= 0, points[:, 0] <= width - 1)
    y_inside = np.logical_and(points[:, 1] >= 0, points[:, 1] <= height - 1)
    return bool(np.all(np.logical_and(x_inside, y_inside)))


def warp_image(image, matrix):
    """Warp a channel-first image using an affine source-to-destination matrix."""
    channels, height, width = image.shape
    affine_matrix = matrix[:2, :].astype(np.float32)
    interpolation = cv2.INTER_LINEAR

    if channels == 1:
        warped = cv2.warpAffine(image[0], affine_matrix, (width, height), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        return warped[np.newaxis, :, :].astype(np.float32)

    hwc_image = np.moveaxis(image, 0, -1)
    border_value = tuple(0.0 for _ in range(channels))
    warped = cv2.warpAffine(hwc_image, affine_matrix, (width, height), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

    if warped.ndim == 2:
        warped = warped[:, :, np.newaxis]

    return np.moveaxis(warped, -1, 0).astype(np.float32)
