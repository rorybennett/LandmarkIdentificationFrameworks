"""
Dataset loader for quadruplet patch CSV files.

Each CSV row stores one sub-patch. Rows with the same patch_id are grouped into
one training sample with shape: [num_sub_patches, channels, height, width].
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage import io
from skimage.util import img_as_float32
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Load grouped multi-scale patch samples for the Quadruplet network."""

    def __init__(self, csv_file, num_sub_patches=4, transform=None):
        self.csv_file = Path(csv_file)
        self.num_sub_patches = num_sub_patches
        self.transform = transform

        self.csv_data = pd.read_csv(self.csv_file, header=None)
        self.patch_groups = self.create_patch_groups()
        self.input_channels = self.infer_input_channels()

    def __len__(self):
        """Return the number of grouped patch samples."""
        return len(self.patch_groups)

    def __getitem__(self, index):
        """Return one grouped multi-scale patch sample."""
        group = self.patch_groups[index]

        image = self.load_patch_group(group, expected_channels=self.input_channels)
        sample_name = group.iloc[0, 2]
        coordinates = group.iloc[0, 3:5].to_numpy(dtype=np.int32)
        labels = group.iloc[0, 5:].to_numpy(dtype=np.int64)

        sample = {
            'image': image,
            'sample_name': sample_name,
            'coordinates': coordinates,
            'labels': labels
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def create_patch_groups(self):
        """Group CSV rows by patch_id and validate each group."""
        groups = []

        for patch_id, group in self.csv_data.groupby(0, sort=False):
            if patch_id == 'none':
                continue

            if len(group) != self.num_sub_patches:
                raise ValueError(f'Patch group {patch_id} has {len(group)} rows, expected {self.num_sub_patches}.')

            groups.append(group.reset_index(drop=True))

        if not groups:
            raise ValueError(f'No valid patch groups found in {self.csv_file}.')

        return groups

    def infer_input_channels(self):
        """Infer the number of channels from the first patch group."""
        first_group = self.patch_groups[0]
        first_group_image = self.load_patch_group(first_group)

        return int(first_group_image.shape[1])

    @staticmethod
    def load_patch_group(group, expected_channels=None):
        """Load all sub-patches for one patch_id as a stacked NumPy array."""
        patches = []

        for _, row in group.iterrows():
            patch_path = row.iloc[1]
            patch_image = load_patch_image(patch_path)

            if expected_channels is not None and patch_image.shape[0] != expected_channels:
                raise ValueError(f'Patch {patch_path} has {patch_image.shape[0]} channels, expected {expected_channels}.')

            patches.append(patch_image)

        shapes = {tuple(patch.shape) for patch in patches}

        if len(shapes) != 1:
            raise ValueError(f'Patch group {group.iloc[0, 0]} contains inconsistent patch shapes: {sorted(shapes)}.')

        return np.stack(patches, axis=0).astype(np.float32)


def load_patch_image(patch_path):
    """Load one patch and return a channel-first float32 array."""
    patch_image = io.imread(patch_path)
    patch_image = img_as_float32(patch_image)

    if patch_image.ndim == 2:
        return patch_image[np.newaxis, :, :]

    if patch_image.ndim == 3:
        return np.moveaxis(patch_image, -1, 0)

    raise ValueError(f'Patch {patch_path} has unsupported image shape {patch_image.shape}.')


class ToTensor:
    """Convert NumPy arrays in a sample to PyTorch tensors."""

    def __call__(self, sample):
        return {
            'image': torch.from_numpy(sample['image']).float(),
            'sample_name': sample['sample_name'],
            'coordinates': torch.from_numpy(sample['coordinates']).long(),
            'labels': torch.from_numpy(sample['labels']).long()
        }
