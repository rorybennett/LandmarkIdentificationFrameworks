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

REPETITION_DIR_PATTERN = re.compile(r'^repetition_(\d+)$')
TRAINING_LIST_PATTERN = re.compile(r'^training_f(\d+)\.txt$')
VALIDATION_LIST_PATTERN = re.compile(r'^val_f(\d+)\.txt$')
SPLIT_FILE_PREFIXES = {'training': 'training', 'validation': 'val'}
ALL_FOLD_NAME = 'all'
ALL_FOLD_FILE_NAMES = {'training': 'training_fall.txt', 'validation': 'val_fall.txt'}
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


def get_repetition_dir(fold_lists_path, repetition):
    """Return the directory containing one repetition's fold lists."""
    return Path(fold_lists_path) / f'repetition_{int(repetition)}'


def discover_repetition_numbers(fold_lists_path):
    """Return contiguous repetition numbers with identical fold counts."""
    fold_lists_path = Path(fold_lists_path)

    if not fold_lists_path.is_dir():
        raise ValueError(f'fold_lists_path does not exist or is not a directory: {fold_lists_path}')

    repetition_numbers = []

    for directory_path in fold_lists_path.iterdir():
        if directory_path.is_dir():
            match = REPETITION_DIR_PATTERN.fullmatch(directory_path.name)
            if match:
                repetition_numbers.append(int(match.group(1)))

    repetition_numbers = sorted(set(repetition_numbers))

    if not repetition_numbers:
        raise ValueError(f'No repetition_N directories found in {fold_lists_path}')

    expected = list(range(1, repetition_numbers[-1] + 1))

    if repetition_numbers != expected:
        raise ValueError(f'Repetition directories must be contiguous from repetition_1. Found {repetition_numbers}, expected {expected}')

    fold_number_sets = {repetition: discover_fold_numbers(fold_lists_path=fold_lists_path, repetition=repetition)
                        for repetition in repetition_numbers}
    reference_folds = fold_number_sets[repetition_numbers[0]]
    inconsistent = {repetition: folds for repetition, folds in fold_number_sets.items() if folds != reference_folds}

    if inconsistent:
        raise ValueError(f'Every repetition must contain identical contiguous fold numbers. Found: {fold_number_sets}')

    return repetition_numbers


def discover_fold_numbers(fold_lists_path, repetition):
    """Return contiguous fold numbers and validate training/validation file pairs."""
    repetition_dir = get_repetition_dir(fold_lists_path=fold_lists_path, repetition=repetition)

    if not repetition_dir.is_dir():
        raise ValueError(f'Repetition directory does not exist: {repetition_dir}')

    training_fold_numbers = []
    validation_fold_numbers = []

    for file_path in repetition_dir.iterdir():
        if file_path.is_file():
            training_match = TRAINING_LIST_PATTERN.fullmatch(file_path.name)
            validation_match = VALIDATION_LIST_PATTERN.fullmatch(file_path.name)

            if training_match:
                training_fold_numbers.append(int(training_match.group(1)))

            if validation_match:
                validation_fold_numbers.append(int(validation_match.group(1)))

    fold_numbers = sorted(set(training_fold_numbers))
    validation_fold_numbers = sorted(set(validation_fold_numbers))

    if not fold_numbers:
        raise ValueError(f'No training_fN.txt files found in {repetition_dir}')

    expected = list(range(1, fold_numbers[-1] + 1))

    if fold_numbers != expected:
        raise ValueError(f'Fold files in {repetition_dir} must be contiguous from training_f1.txt. Found {fold_numbers}, expected {expected}')

    if validation_fold_numbers != fold_numbers:
        raise ValueError(f'Training and validation fold numbers in {repetition_dir} must match exactly. '
                         f'Found training={fold_numbers}, validation={validation_fold_numbers}.')

    missing_files = []

    for fold_number in fold_numbers:
        for prefix in SPLIT_FILE_PREFIXES.values():
            fold_file = repetition_dir / f'{prefix}_f{fold_number}.txt'

            if not fold_file.is_file():
                missing_files.append(str(fold_file))

    if missing_files:
        missing_text = '\n'.join(missing_files)
        raise ValueError(f'Every fold must have training_fN.txt and val_fN.txt files. Missing files:\n{missing_text}')

    return fold_numbers


def canonical_split_name(value):
    """Return the canonical sample key used for fold-overlap checks."""
    return Path(str(value).split()[0]).stem


def normalise_fold(value):
    """Return a positive fold number or the canonical ``all`` fold name."""
    if str(value).strip().lower() == ALL_FOLD_NAME:
        return ALL_FOLD_NAME

    try:
        fold = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'fold must be a positive integer or "{ALL_FOLD_NAME}". Got: {value}') from error

    if fold < 1:
        raise ValueError(f'fold must be a positive integer or "{ALL_FOLD_NAME}". Got: {value}')

    return fold


def is_all_fold(value):
    """Return whether a fold value selects the all-data fold."""
    return normalise_fold(value) == ALL_FOLD_NAME


def get_split_file_path(fold_lists_path, repetition, split_name, fold):
    """Return one repetition/fold split-list path."""
    canonical_name = str(split_name).lower()

    if canonical_name not in SPLIT_FILE_PREFIXES:
        raise ValueError(f'Unknown split name: {split_name}. Expected one of {tuple(SPLIT_FILE_PREFIXES)}.')

    fold = normalise_fold(fold)
    file_name = ALL_FOLD_FILE_NAMES[canonical_name] if fold == ALL_FOLD_NAME else f'{SPLIT_FILE_PREFIXES[canonical_name]}_f{fold}.txt'
    return get_repetition_dir(fold_lists_path=fold_lists_path, repetition=repetition) / file_name


def read_split_names(fold_lists_path, repetition, split_name, fold):
    """Read sample names for one repetition and fold split."""
    split_file = get_split_file_path(fold_lists_path=fold_lists_path, repetition=repetition, split_name=split_name, fold=fold)

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


def validate_split_duplicates(split_name, names, repetition, fold):
    """Raise an error if a split file contains duplicate sample IDs."""
    seen = set()
    duplicates = set()

    for name in names:
        sample_key = canonical_split_name(name)

        if sample_key in seen:
            duplicates.add(sample_key)

        seen.add(sample_key)

    if duplicates:
        duplicate_text = ', '.join(sorted(duplicates, key=natural_key))
        raise ValueError(f'Repetition {repetition}, fold {fold} {split_name} split contains duplicate sample ID(s): {duplicate_text}')


def validate_fold_split_overlaps(fold_lists_path, repetition, fold):
    """Validate one fold's split relationship, including the intentional all-fold overlap."""
    fold = normalise_fold(fold)
    split_names = {split_name: read_split_names(fold_lists_path=fold_lists_path, repetition=repetition, split_name=split_name, fold=fold)
                   for split_name in SPLIT_FILE_PREFIXES}
    split_sets = {}

    for split_name, names in split_names.items():
        validate_split_duplicates(split_name=split_name, names=names, repetition=repetition, fold=fold)
        split_sets[split_name] = {canonical_split_name(name) for name in names}

    if fold == ALL_FOLD_NAME:
        if split_sets['training'] != split_sets['validation']:
            training_only = split_sets['training'] - split_sets['validation']
            validation_only = split_sets['validation'] - split_sets['training']
            raise ValueError(
                f'Repetition {repetition}, fold all requires training_fall.txt and val_fall.txt to contain the same sample IDs. '
                f'Training-only={sorted(training_only, key=natural_key)}, validation-only={sorted(validation_only, key=natural_key)}.'
            )
        return split_sets

    overlap = split_sets['training'] & split_sets['validation']

    if overlap:
        overlap_text = ', '.join(sorted(overlap, key=natural_key))
        training_file = get_split_file_path(fold_lists_path, repetition, 'training', fold)
        validation_file = get_split_file_path(fold_lists_path, repetition, 'validation', fold)
        raise ValueError(f'Repetition {repetition}, fold {fold} has overlapping sample ID(s) between {training_file.name} and '
                         f'{validation_file.name}: {overlap_text}')

    return split_sets


def validate_repeated_kfold_lists(fold_lists_path):
    """Validate the complete repeated k-fold collection."""
    repetition_numbers = discover_repetition_numbers(fold_lists_path)
    reference_sample_set = None
    fold_numbers = discover_fold_numbers(fold_lists_path=fold_lists_path, repetition=repetition_numbers[0])

    for repetition in repetition_numbers:
        repetition_sample_set = None
        validation_occurrences = []

        for fold in fold_numbers:
            split_sets = validate_fold_split_overlaps(fold_lists_path=fold_lists_path, repetition=repetition, fold=fold)
            fold_sample_set = split_sets['training'] | split_sets['validation']

            if repetition_sample_set is None:
                repetition_sample_set = fold_sample_set
            elif fold_sample_set != repetition_sample_set:
                raise ValueError(f'Repetition {repetition}, fold {fold} does not cover the same full dataset as the other folds.')

            if split_sets['training'] != fold_sample_set - split_sets['validation']:
                raise ValueError(f'Repetition {repetition}, fold {fold} training split is not the complement of its validation split.')

            validation_occurrences.extend(split_sets['validation'])

        if len(validation_occurrences) != len(set(validation_occurrences)):
            raise ValueError(f'Repetition {repetition} uses at least one sample as validation in more than one fold.')

        if set(validation_occurrences) != repetition_sample_set:
            raise ValueError(f'Repetition {repetition} does not use every sample as validation exactly once.')

        if reference_sample_set is None:
            reference_sample_set = repetition_sample_set
        elif repetition_sample_set != reference_sample_set:
            raise ValueError(f'Repetition {repetition} does not contain the same full dataset as repetition {repetition_numbers[0]}.')

    all_fold_presence = []

    for repetition in repetition_numbers:
        all_paths = [get_split_file_path(fold_lists_path, repetition, split_name, ALL_FOLD_NAME)
                     for split_name in SPLIT_FILE_PREFIXES]
        all_fold_presence.extend(path.is_file() for path in all_paths)

        if any(path.is_file() for path in all_paths) and not all(path.is_file() for path in all_paths):
            missing = [str(path) for path in all_paths if not path.is_file()]
            raise ValueError(f'Fold all requires both training_fall.txt and val_fall.txt. Missing: {missing}')

    if any(all_fold_presence):
        if not all(all_fold_presence):
            raise ValueError('Fold-all files must be present in every repetition or omitted from every repetition.')

        for repetition in repetition_numbers:
            split_sets = validate_fold_split_overlaps(fold_lists_path, repetition, ALL_FOLD_NAME)
            if split_sets['training'] != reference_sample_set:
                raise ValueError(f'Repetition {repetition}, fold all does not contain the same full dataset as the numbered folds.')

    return {'repetitions': repetition_numbers, 'folds': fold_numbers, 'sample_count': len(reference_sample_set)}


def resolve_image_path(image_data_dir, image_name, sample_stem, recursive=False, supported_suffixes=SUPPORTED_IMAGE_SUFFIXES):
    """Find the image file for one sample."""
    image_data_dir = Path(image_data_dir)
    candidates = [image_data_dir / image_name, image_data_dir / f'{sample_stem}{Path(image_name).suffix}']

    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate

    suffixes = tuple(suffix.lower() for suffix in supported_suffixes)
    search_iter = image_data_dir.rglob('*') if recursive else image_data_dir.iterdir()

    matches = [path for path in sorted(search_iter, key=lambda item: item.as_posix().lower())
               if path.is_file() and path.stem == sample_stem and path.suffix.lower() in suffixes]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        match_text = '\n'.join(str(path) for path in matches)
        raise ValueError(f'Multiple images matched sample {sample_stem} under {image_data_dir}:\n{match_text}')

    raise FileNotFoundError(f'Image for {sample_stem} was not found under {image_data_dir}')


def get_image_channel_count(image, image_path=None):
    """Return the number of source channels in an image."""
    path_text = f' for {image_path}' if image_path is not None else ''

    if image.ndim == 2:
        return 1

    if image.ndim == 3 and image.shape[2] in (1, 3, 4):
        return int(image.shape[2])

    raise ValueError(f'Unsupported image shape{path_text}: {image.shape}. Expected greyscale, RGB, or RGBA.')


def infer_image_channel_count(image_path):
    """Read one image and return its source channel count."""
    image = io.imread(image_path)
    return get_image_channel_count(image=image, image_path=image_path)


def get_image_size(image_path):
    """Read one image and return its height and width."""
    image = io.imread(image_path)

    if image.ndim < 2:
        raise ValueError(f'Unsupported image shape for {image_path}: {image.shape}.')

    return int(image.shape[0]), int(image.shape[1])


def validate_points_within_image(points, image_size, sample_name=None, image_path=None):
    """Validate that every xy landmark lies inside the corresponding image bounds."""
    height, width = map(int, image_size)
    points_array = np.asarray(points, dtype=np.float32)
    sample_text = f' for sample {sample_name}' if sample_name is not None else ''
    path_text = f' ({image_path})' if image_path is not None else ''

    if height < 1 or width < 1:
        raise ValueError(f'Invalid image size{sample_text}{path_text}: height={height}, width={width}.')

    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f'Landmarks{sample_text}{path_text} must have shape [N, 2]. Got: {points_array.shape}')

    for point_index, (x, y) in enumerate(points_array, start=1):
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f'Landmark {point_index}{sample_text}{path_text} contains a non-finite coordinate: ({x}, {y}).')

        if not (0.0 <= float(x) < width and 0.0 <= float(y) < height):
            raise ValueError(
                f'Landmark {point_index}{sample_text}{path_text} is outside image bounds: '
                f'point=({float(x)}, {float(y)}), valid x=[0, {width}), valid y=[0, {height}).'
            )

    return points_array


def validate_resolved_input_channels(input_channels):
    """Validate internally resolved model input channels."""
    if input_channels is None:
        raise ValueError('input_channels has not been resolved. Call the automatic channel resolver before creating the DataLoader.')

    input_channels = int(input_channels)

    if input_channels not in (1, 3, 4):
        raise ValueError(f'input_channels must resolve to 1, 3, or 4. Got: {input_channels}')

    return input_channels


def convert_channels_if_needed(image, input_channels, image_path=None):
    """Validate that an image matches the internally resolved channel count."""
    expected_channels = validate_resolved_input_channels(input_channels)
    actual_channels = get_image_channel_count(image=image, image_path=image_path)

    if actual_channels != expected_channels:
        raise ValueError(
            f'Image channel mismatch for {image_path}: expected {expected_channels} channel(s), '
            f'but found {actual_channels}. All train and validation images must have the same number of source channels.'
        )

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    return image


def load_image_as_float(image_path, input_channels):
    """Load an image as channel-first float32 in the requested channel count."""
    image = img_as_float32(io.imread(image_path))
    validate_image_value_range(image=image, image_path=image_path)
    image = convert_channels_if_needed(image=image, input_channels=input_channels, image_path=image_path)
    return np.moveaxis(image, -1, 0).astype(np.float32)


def validate_image_value_range(image, image_path=None):
    """Validate finite image values in the normalised 0 to 1 range."""
    path_text = f' for {image_path}' if image_path is not None else ''

    if not np.all(np.isfinite(image)):
        raise ValueError(f'Image{path_text} contains NaN or infinite values.')

    image_min = float(np.min(image))
    image_max = float(np.max(image))

    if image_min < 0.0 or image_max > 1.0:
        raise ValueError(f'Image{path_text} has values outside the supported 0 to 1 range after loading: min={image_min}, max={image_max}.')

    return image


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
