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
HORIZONTAL_FLIP_PROBABILITY = 0.2
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
        """Apply each transform to a channel-first image and xy landmarks."""
        for transform in self.transforms:
            image, points = transform(image=image, points=points)

        return image, points


@dataclass
class RandomErasing:
    probability: float = RANDOM_ERASING_PROBABILITY
    scale: tuple[float, float] = RANDOM_ERASING_SCALE
    ratio: tuple[float, float] = RANDOM_ERASING_RATIO
    fill_value: float = 0.0

    def __call__(self, image, points):
        """Erase one random rectangular region without moving landmarks."""
        if np.random.random() >= self.probability:
            return image, points

        channels, height, width = image.shape
        area = float(height * width)
        image = image.copy()

        for _ in range(10):
            target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = math.exp(np.random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            erase_height = int(round(math.sqrt(target_area / aspect_ratio)))
            erase_width = int(round(math.sqrt(target_area * aspect_ratio)))

            if erase_height < height and erase_width < width:
                top = np.random.randint(0, height - erase_height + 1)
                left = np.random.randint(0, width - erase_width + 1)
                image[:, top:top + erase_height, left:left + erase_width] = self.fill_value
                break

        return image.astype(np.float32), points


@dataclass
class RandomAffine:
    degrees: float = AFFINE_DEGREES
    shear: float = AFFINE_SHEAR
    translate: tuple[float, float] = AFFINE_TRANSLATE
    scale: tuple[float, float] = AFFINE_SCALE
    max_attempts: int | None = AFFINE_MAX_ATTEMPTS

    def __call__(self, image, points):
        """Apply a sampled affine transform only when every transformed landmark remains inside the image."""
        _, height, width = image.shape
        points = np.asarray(points, dtype=np.float32)
        attempt = 0

        while self.max_attempts is None or attempt < int(self.max_attempts):
            attempt += 1
            matrix = self.sample_matrix(width=width, height=height)
            transformed_points = transform_points(points=points, matrix=matrix)

            if points_inside_image(points=transformed_points, width=width, height=height):
                return warp_image(image=image, matrix=matrix), transformed_points.astype(np.float32)

        raise RuntimeError(f'No valid affine transform was found after {self.max_attempts} attempts. Reduce affine ranges or inspect landmarks near the image border.')

    def sample_matrix(self, width, height):
        """Sample an affine source-to-destination matrix around the image centre."""
        angle = np.random.uniform(-self.degrees, self.degrees)
        shear_x = np.random.uniform(-self.shear, self.shear)
        scale_value = np.random.uniform(self.scale[0], self.scale[1])
        translate_x = np.random.uniform(-self.translate[0], self.translate[0]) * width
        translate_y = np.random.uniform(-self.translate[1], self.translate[1]) * height
        centre_x = (width - 1) / 2.0
        centre_y = (height - 1) / 2.0

        return translation_matrix(translate_x, translate_y) @ translation_matrix(centre_x, centre_y) @ rotation_matrix(angle) @ shear_matrix(shear_x) @ scale_matrix(scale_value) @ translation_matrix(-centre_x, -centre_y)


@dataclass
class RandomHorizontalFlip:
    probability: float = HORIZONTAL_FLIP_PROBABILITY
    point_index_swaps: tuple[tuple[int, int], ...] = ()

    def __call__(self, image, points):
        """Flip the image horizontally and optionally swap symmetric landmark channels."""
        if np.random.random() >= self.probability:
            return image, points

        _, _, width = image.shape
        flipped_image = np.flip(image, axis=2).copy()
        flipped_points = np.asarray(points, dtype=np.float32).copy()
        flipped_points[:, 0] = (width - 1) - flipped_points[:, 0]

        for left_index, right_index in self.point_index_swaps:
            flipped_points[[left_index, right_index]] = flipped_points[[right_index, left_index]]

        return flipped_image.astype(np.float32), flipped_points.astype(np.float32)


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
            image[:colour_channels] = image[:colour_channels] + np.random.normal(self.mean, self.sigma, size=(colour_channels, height, width)).astype(np.float32)

        if self.clip:
            image[:colour_channels] = np.clip(image[:colour_channels], 0.0, 1.0)

        return image.astype(np.float32), points


@dataclass
class GaussianBlur:
    kernel_size: int = GAUSSIAN_BLUR_KERNEL_SIZE

    def __call__(self, image, points):
        """Blur each image channel without moving landmarks."""
        kernel_size = make_odd_kernel_size(self.kernel_size)
        blurred_channels = [cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0) for channel in image]
        return np.stack(blurred_channels, axis=0).astype(np.float32), points


def get_default_heatmap_transforms(num_of_points=None):
    """Return default oversampling transforms for heatmap landmark training."""
    return Compose([
        RandomErasing(),
        RandomAffine(),
        RandomHorizontalFlip(point_index_swaps=get_default_horizontal_flip_swaps(num_of_points=num_of_points)),
        GaussianNoise(),
        GaussianBlur(),
    ])


def get_default_horizontal_flip_swaps(num_of_points=None):
    """Return prostate transverse point swaps for horizontal flips when four landmarks are used."""
    if int(num_of_points or 0) == 4:
        return ((1, 3),)

    return ()


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
