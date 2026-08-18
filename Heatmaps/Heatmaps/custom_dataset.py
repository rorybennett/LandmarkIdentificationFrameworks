"""
Dataset loader for full-image landmark heatmap regression.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap_transforms import get_default_heatmap_transforms
from .utils.annotation_utils import read_mark_list, resolve_mark_record, validate_annotation_point_count
from .utils.io_utils import create_heatmaps, get_image_size, get_split_file_path, load_image_as_float, natural_key, read_split_names, resize_channel_first, resolve_image_path, scale_points, validate_points_within_image


@dataclass
class HeatmapDatasetConfig:
    repetition: int
    fold: int
    split_name: str
    num_of_points: int
    fold_lists_path: Path
    mark_list_file: Path
    image_data_dir: Path
    image_size: tuple[int, int]
    heatmap_sigma: float
    input_channels: int | None = None
    recursive_image_search: bool = False
    oversampling_factor: int = 1


class HeatmapDataset(Dataset):
    """Load full images and generate target heatmaps on demand."""

    def __init__(self, config):
        self.config = config
        self.mark_records = read_mark_list(config.mark_list_file)
        self.records = self.build_records()
        self.oversampling_factor = self.resolve_oversampling_factor()
        self.oversampling_transform = get_default_heatmap_transforms() if self.oversampling_factor > 1 else None

    def __len__(self):
        return len(self.records) * self.oversampling_factor

    def __getitem__(self, index):
        original_index = int(index) % len(self.records)
        is_oversampled = int(index) >= len(self.records)
        record = self.records[original_index]
        image = load_image_as_float(record['image_path'], input_channels=self.config.input_channels)
        original_size = np.asarray(image.shape[1:3], dtype=np.int64)
        original_points = np.asarray(record['points'], dtype=np.float32)

        if is_oversampled and self.oversampling_transform is not None:
            image, original_points = self.oversampling_transform(image=image, points=original_points)

        image = resize_channel_first(image=image, image_size=self.config.image_size)
        heatmap_points = scale_points(points=original_points, original_size=original_size, image_size=self.config.image_size)
        heatmaps = create_heatmaps(points=heatmap_points, image_size=self.config.image_size, sigma=self.config.heatmap_sigma)

        return {'image': torch.from_numpy(image).float(), 'heatmaps': torch.from_numpy(heatmaps).float(), 'points_original': torch.from_numpy(original_points).float(), 'original_size': torch.from_numpy(original_size).long(), 'sample_name': record['sample_name'], 'image_path': str(record['image_path']), 'is_oversampled': bool(is_oversampled)}

    def resolve_oversampling_factor(self):
        """Return the active oversampling factor for this split."""
        factor = int(self.config.oversampling_factor)

        if factor < 1:
            raise ValueError(f'oversampling_factor must be at least 1. Got: {factor}')

        if self.config.split_name.lower() != 'training':
            return 1

        return factor

    def validate_all_records(self):
        """Validate the complete preprocessing path for every original record."""
        if self.config.input_channels is None:
            raise ValueError('input_channels must be resolved before validating dataset records.')

        image_height, image_width = map(int, self.config.image_size)
        expected_image_shape = (int(self.config.input_channels), image_height, image_width)
        expected_heatmap_shape = (int(self.config.num_of_points), image_height, image_width)

        for record_index in range(len(self.records)):
            sample = self[record_index]
            sample_name = sample['sample_name']

            if tuple(sample['image'].shape) != expected_image_shape:
                raise ValueError(f'Preprocessed image for {sample_name} has shape {tuple(sample["image"].shape)}, expected {expected_image_shape}.')

            if tuple(sample['heatmaps'].shape) != expected_heatmap_shape:
                raise ValueError(f'Target heatmaps for {sample_name} have shape {tuple(sample["heatmaps"].shape)}, expected {expected_heatmap_shape}.')

            if not torch.isfinite(sample['image']).all():
                raise ValueError(f'Preprocessed image for {sample_name} contains NaN or infinite values.')

            if not torch.isfinite(sample['heatmaps']).all():
                raise ValueError(f'Target heatmaps for {sample_name} contain NaN or infinite values.')

        return len(self.records)

    def build_records(self):
        """Build image and point records for this split."""
        split_names = read_split_names(fold_lists_path=self.config.fold_lists_path, repetition=self.config.repetition,
                                       split_name=self.config.split_name, fold=self.config.fold)
        split_file = get_split_file_path(fold_lists_path=self.config.fold_lists_path, repetition=self.config.repetition,
                                         split_name=self.config.split_name, fold=self.config.fold)
        records = []

        for sample_name in split_names:
            try:
                sample_stem, mark_record = resolve_mark_record(sample_name=sample_name, mark_records=self.mark_records)
            except KeyError as error:
                raise ValueError(
                    f'Dataset validation failed for repetition {self.config.repetition}, fold {self.config.fold}, {self.config.split_name} split: '
                    f"patient/sample {Path(str(sample_name)).stem!r}, listed in '{split_file}', has no annotation row in "
                    f"'{Path(self.config.mark_list_file)}'. Training cancelled; existing outputs were not removed."
                ) from error

            validate_annotation_point_count(mark_record=mark_record, expected_points=self.config.num_of_points, sample_name=sample_stem,
                                            mark_list_file=self.config.mark_list_file, repetition=self.config.repetition, fold=self.config.fold,
                                            split_name=self.config.split_name, training_context=True)
            image_path = resolve_image_path(image_data_dir=self.config.image_data_dir, image_name=mark_record['image_name'], sample_stem=sample_stem, recursive=self.config.recursive_image_search)
            image_size = get_image_size(image_path)
            points = validate_points_within_image(points=mark_record['points'], image_size=image_size, sample_name=sample_stem, image_path=image_path)
            records.append({'sample_name': sample_stem, 'image_path': image_path, 'points': points})

        records.sort(key=lambda item: natural_key(item['sample_name']))

        if not records:
            raise ValueError(f'No records found for repetition {self.config.repetition}, fold {self.config.fold}, {self.config.split_name} split.')

        return records
