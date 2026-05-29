"""
Dataset loader for full-image landmark heatmap regression.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap_transforms import get_default_heatmap_transforms
from .utils.io_utils import create_heatmaps, load_image_as_float, natural_key, read_mark_list, read_split_names, resize_channel_first, resolve_image_path, resolve_mark_record, scale_points


@dataclass
class HeatmapDatasetConfig:
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
        self.mark_records = read_mark_list(config.mark_list_file, expected_points=config.num_of_points)
        self.records = self.build_records()
        self.oversampling_factor = self.resolve_oversampling_factor()
        self.oversampling_transform = get_default_heatmap_transforms(num_of_points=config.num_of_points) if self.oversampling_factor > 1 else None

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

        if self.config.split_name.lower() != 'train':
            return 1

        return factor

    def build_records(self):
        """Build image and point records for this split."""
        split_names = read_split_names(fold_lists_path=self.config.fold_lists_path, split_name=self.config.split_name, fold=self.config.fold)
        records = []

        for sample_name in split_names:
            sample_stem, mark_record = resolve_mark_record(sample_name=sample_name, mark_records=self.mark_records)
            image_path = resolve_image_path(image_data_dir=self.config.image_data_dir, image_name=mark_record['image_name'], sample_stem=sample_stem, recursive=self.config.recursive_image_search)
            records.append({'sample_name': sample_stem, 'image_path': image_path, 'points': mark_record['points']})

        records.sort(key=lambda item: natural_key(item['sample_name']))

        if not records:
            raise ValueError(f'No records found for split {self.config.split_name} fold {self.config.fold}.')

        return records
