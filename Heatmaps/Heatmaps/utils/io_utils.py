"""
File, image, and landmark I/O helpers for heatmap training.
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage import io
from skimage.util import img_as_float32

POINT_PATTERN = re.compile(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)')
TRAIN_LIST_PATTERN = re.compile(r'^train_f(\d+)\.txt$')
SUPPORTED_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def str_to_bool(value):
    """Convert command-line strings to booleans."""
    if isinstance(value, bool):
        return value

    value = str(value).lower().strip()

    if value in ('true', 't', 'yes', 'y', '1'):
        return True

    if value in ('false', 'f', 'no', 'n', '0'):
        return False

    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def natural_key(value):
    """Sort strings naturally, so A2 comes before A10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(value))]


def safe_file_stem(value):
    """Create a safe filename stem."""
    safe_value = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._-')

    if not safe_value:
        raise ValueError(f'Cannot create a safe filename from: {value}')

    return safe_value


def discover_fold_numbers(fold_lists_path):
    """Return contiguous fold numbers from train_fN.txt files."""
    fold_lists_path = Path(fold_lists_path)

    if not fold_lists_path.is_dir():
        raise ValueError(f'fold_lists_path does not exist or is not a directory: {fold_lists_path}')

    fold_numbers = []

    for file_path in fold_lists_path.iterdir():
        if file_path.is_file():
            match = TRAIN_LIST_PATTERN.fullmatch(file_path.name)
            if match:
                fold_numbers.append(int(match.group(1)))

    fold_numbers = sorted(set(fold_numbers))

    if not fold_numbers:
        raise ValueError(f'No train_fN.txt files found in {fold_lists_path}')

    expected = list(range(1, fold_numbers[-1] + 1))

    if fold_numbers != expected:
        raise ValueError(f'Fold files must be contiguous from train_f1.txt. Found {fold_numbers}, expected {expected}')

    for fold_number in fold_numbers:
        for prefix in ('train', 'val'):
            fold_file = fold_lists_path / f'{prefix}_f{fold_number}.txt'
            if not fold_file.is_file():
                raise ValueError(f'Missing fold list file: {fold_file}')

    return fold_numbers


def read_split_names(fold_lists_path, split_name, fold):
    """Read sample names for one fold split."""
    split_file = Path(fold_lists_path) / f'{split_name.lower()}_f{int(fold)}.txt'

    if not split_file.is_file():
        raise FileNotFoundError(f'Split file not found: {split_file}')

    names = []

    with open(split_file, 'r', encoding='utf-8') as split_handle:
        for line in split_handle:
            line = line.strip()
            if line:
                names.append(line.split()[0])

    if not names:
        raise ValueError(f'Split file is empty: {split_file}')

    return names


def read_mark_list(mark_list_file, expected_points):
    """Read a mark-list file keyed by sample stem."""
    mark_list_file = Path(mark_list_file)

    if not mark_list_file.is_file():
        raise FileNotFoundError(f'Mark-list file not found: {mark_list_file}')

    records = {}

    with open(mark_list_file, 'r', encoding='utf-8') as mark_handle:
        for line_number, line in enumerate(mark_handle, start=1):
            line = line.strip()

            if not line:
                continue

            image_name = line.split()[0]
            points = [(float(x), float(y)) for x, y in POINT_PATTERN.findall(line)]

            if len(points) < int(expected_points):
                raise ValueError(f'Mark-list row {line_number} for {image_name} has {len(points)} points, expected at least {expected_points}.')

            sample_stem = Path(image_name).stem

            if sample_stem in records:
                raise ValueError(f'Duplicate sample stem in mark list: {sample_stem}')

            records[sample_stem] = {'image_name': image_name, 'points': points[:int(expected_points)]}

    if not records:
        raise ValueError(f'No valid mark-list rows found in {mark_list_file}')

    return records


def resolve_mark_record(sample_name, mark_records):
    """Match a fold-list sample name to a mark-list record."""
    candidates = [Path(sample_name).stem, Path(sample_name).name, str(sample_name)]

    for candidate in candidates:
        if candidate in mark_records:
            return candidate, mark_records[candidate]

    raise KeyError(f'Sample {sample_name} was not found in the mark list.')


def resolve_image_path(image_data_dir, image_name, sample_stem, recursive=False, supported_suffixes=SUPPORTED_IMAGE_SUFFIXES):
    """Find the image file for one sample."""
    image_data_dir = Path(image_data_dir)
    candidates = [image_data_dir / image_name, image_data_dir / f'{sample_stem}{Path(image_name).suffix}']

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    suffixes = tuple(suffix.lower() for suffix in supported_suffixes)
    search_iter = image_data_dir.rglob('*') if recursive else image_data_dir.iterdir()

    for path in sorted(search_iter, key=lambda item: item.as_posix().lower()):
        if path.is_file() and path.stem == sample_stem and path.suffix.lower() in suffixes:
            return path

    raise FileNotFoundError(f'Image for {sample_stem} was not found under {image_data_dir}')


def load_image_as_float(image_path, input_channels):
    """Load an image as channel-first float32 in the requested channel count."""
    image = img_as_float32(io.imread(image_path))

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    if image.ndim != 3:
        raise ValueError(f'Unsupported image shape for {image_path}: {image.shape}')

    if image.shape[2] == 4 and int(input_channels) in (1, 3):
        image = image[:, :, :3]

    if int(input_channels) == 1:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        return image[np.newaxis, :, :].astype(np.float32)

    if int(input_channels) == 3:
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] != 3:
            raise ValueError(f'Cannot convert {image_path} with {image.shape[2]} channels to 3 channels.')
        return np.moveaxis(image, -1, 0).astype(np.float32)

    raise ValueError(f'input_channels must be 1 or 3. Got: {input_channels}')


def resize_channel_first(image, image_size):
    """Resize a channel-first image."""
    target_height, target_width = map(int, image_size)
    channels = [cv2.resize(channel, (target_width, target_height), interpolation=cv2.INTER_AREA) for channel in image]
    return np.stack(channels, axis=0).astype(np.float32)


def scale_points(points, original_size, image_size):
    """Scale xy points from original image size to training image size."""
    original_height, original_width = original_size
    target_height, target_width = image_size
    scale_x = float(target_width) / float(original_width)
    scale_y = float(target_height) / float(original_height)
    return np.asarray([(float(x) * scale_x, float(y) * scale_y) for x, y in points], dtype=np.float32)


def create_heatmaps(points, image_size, sigma):
    """Create one Gaussian heatmap per landmark point."""
    height, width = map(int, image_size)
    yy, xx = np.mgrid[0:height, 0:width]
    heatmaps = np.zeros((len(points), height, width), dtype=np.float32)

    for point_index, (x, y) in enumerate(points):
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * float(sigma) ** 2))
        max_value = float(heatmap.max())
        heatmaps[point_index] = (heatmap / max_value if max_value > 0 else heatmap).astype(np.float32)

    return heatmaps


def heatmaps_to_points(heatmaps):
    """Convert heatmaps to xy points using the maximum response."""
    batch_size, num_points, height, width = heatmaps.shape
    flat_indices = torch.argmax(heatmaps.reshape(batch_size, num_points, height * width), dim=2)
    y = torch.div(flat_indices, width, rounding_mode='floor').float()
    x = (flat_indices % width).float()
    return torch.stack((x, y), dim=2)


def scale_points_to_original(points, original_sizes, image_size):
    """Scale predicted resized points back to original image coordinates."""
    target_height, target_width = map(float, image_size)
    original_height = original_sizes[:, 0].float().to(points.device)
    original_width = original_sizes[:, 1].float().to(points.device)
    scaled = points.clone()
    scaled[:, :, 0] = scaled[:, :, 0] * (original_width[:, None] / target_width)
    scaled[:, :, 1] = scaled[:, :, 1] * (original_height[:, None] / target_height)
    return scaled
