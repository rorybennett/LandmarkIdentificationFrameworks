"""
Interactively verify one selected heatmap transform on randomly selected marked images.

Usage:
python -m Heatmaps.utils.verify_transforms /path/to/images /path/to/points.txt affine

Press SPACE to select a new random image and resample the selected transform.
"""

import argparse
import pprint
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.util import img_as_float32

try:
    from .. import heatmap_transforms as htf
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import heatmap_transforms as htf

POINT_PATTERN = re.compile(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)')
SUPPORTED_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
TRANSFORM_CHOICES = ('erasing', 'affine', 'flip', 'noise', 'blur', 'default')


def read_mark_rows(mark_list_file):
    """Read image names and landmark points from a mark-list file."""
    rows = []

    with open(mark_list_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            image_name = line.split()[0]
            points = [(float(x), float(y)) for x, y in POINT_PATTERN.findall(line)]

            if not points:
                raise ValueError(f'No points found on line {line_number}: {line}')

            rows.append({'image_name': image_name, 'points': np.asarray(points, dtype=np.float32)})

    if not rows:
        raise ValueError(f'No mark-list rows found in {mark_list_file}')

    return rows


def resolve_image_path(image_dir, image_name):
    """Find a marked image inside the supplied image directory."""
    image_dir = Path(image_dir)
    direct_path = image_dir / image_name

    if direct_path.is_file():
        return direct_path

    image_stem = Path(image_name).stem
    suffixes = tuple(suffix.lower() for suffix in SUPPORTED_IMAGE_SUFFIXES)

    for path in sorted(image_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.stem == image_stem and path.suffix.lower() in suffixes:
            return path

    raise FileNotFoundError(f'Could not find image {image_name} in {image_dir}')


def load_image(image_path):
    """Load an image as channel-first float32."""
    image = img_as_float32(io.imread(image_path))

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    if image.ndim != 3:
        raise ValueError(f'Unsupported image shape for {image_path}: {image.shape}')

    return np.moveaxis(image, -1, 0).astype(np.float32)


def make_transform(transform_name, num_points):
    """Create the selected transform, forcing stochastic gates on for visual checking."""
    transform_name = transform_name.lower()

    if transform_name == 'erasing':
        return htf.RandomErasing(probability=1.0)

    if transform_name == 'affine':
        return htf.RandomAffine()

    if transform_name == 'flip':
        return htf.RandomHorizontalFlip(probability=1.0, point_index_swaps=htf.get_default_horizontal_flip_swaps(num_points))

    if transform_name == 'noise':
        return htf.GaussianNoise()

    if transform_name == 'blur':
        return htf.GaussianBlur()

    if transform_name == 'default':
        return htf.get_default_heatmap_transforms(num_of_points=num_points)

    raise ValueError(f'Unknown transform: {transform_name}')


def collect_transform_params(transform):
    """Return sampled transform values recorded by heatmap_transforms.py."""
    if hasattr(transform, 'last_params'):
        return transform.last_params

    if hasattr(transform, 'transforms'):
        return [getattr(item, 'last_params', {'transform': item.__class__.__name__, 'params': 'not recorded'}) for item in transform.transforms]

    return getattr(transform, 'last_params', {'transform': transform.__class__.__name__, 'params': 'not recorded'})


def to_display_image(image):
    """Convert a channel-first image to a Matplotlib image."""
    image = np.clip(image, 0.0, 1.0)

    if image.shape[0] == 1:
        return image[0]

    return np.moveaxis(image[:3], 0, -1)


def draw_pair(axes, image_path, transform_name, original_image, original_points, transformed_image, transformed_points):
    """Draw the original and transformed image pair on the existing axes."""
    panels = (
        (axes[0], original_image, original_points, f'Original: {Path(image_path).name}'),
        (axes[1], transformed_image, transformed_points, f'Transformed: {transform_name}'),
    )

    for axis, image, points, title in panels:
        axis.clear()
        display_image = to_display_image(image)
        axis.imshow(display_image, cmap='gray' if display_image.ndim == 2 else None)
        axis.scatter(points[:, 0], points[:, 1], s=50, marker='x', c='limegreen')
        axis.set_title(title)
        axis.axis('off')


def print_transform_summary(image_path, points, transformed_points, transform_params):
    """Print selected sample and sampled transform values."""
    print('\n' + '=' * 80)
    print(f'Selected image: {image_path}')
    print(f'Transform module: {Path(htf.__file__).resolve()}')
    print('Original points:')
    pprint.pp(points.tolist())
    print('Transformed points:')
    pprint.pp(transformed_points.tolist())
    print('Transform values:')
    pprint.pp(transform_params)
    print('Press SPACE for another random image, or close the window to exit.', flush=True)


def choose_row(rows, previous_image_name=None):
    """Randomly choose a mark-list row, avoiding the previous image when possible."""
    if len(rows) == 1:
        return rows[0]

    candidates = [row for row in rows if row['image_name'] != previous_image_name]
    return random.choice(candidates or rows)


class TransformViewer:
    """Interactive Matplotlib viewer for repeatedly checking one transform type."""

    def __init__(self, image_dir, rows, transform_name):
        self.image_dir = Path(image_dir)
        self.rows = rows
        self.transform_name = transform_name
        self.current_image_name = None
        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def show_next(self):
        """Select a new random image, apply the selected transform, and update the viewer."""
        row = choose_row(self.rows, previous_image_name=self.current_image_name)
        self.current_image_name = row['image_name']
        image_path = resolve_image_path(self.image_dir, row['image_name'])
        image = load_image(image_path)
        points = row['points'].copy()
        transform = make_transform(self.transform_name, num_points=len(points))
        transformed_image, transformed_points = transform(image=image, points=points.copy())

        draw_pair(self.axes, image_path, self.transform_name, image, points, transformed_image, transformed_points)
        self.figure.suptitle('Press SPACE for a new random image')
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()
        print_transform_summary(image_path, points, transformed_points, collect_transform_params(transform))

    def on_key_press(self, event):
        """Handle keyboard input from the Matplotlib window."""
        if event.key == ' ':
            self.show_next()


def parse_args():
    """Parse the required command-line arguments."""
    parser = argparse.ArgumentParser(description='Interactively verify one heatmap transform on randomly selected marked images.')
    parser.add_argument('image_dir', type=Path, help='Directory containing the images.')
    parser.add_argument('mark_list_file', type=Path, help='Text file containing image names and landmark points.')
    parser.add_argument('transform', choices=TRANSFORM_CHOICES, help='Transform to apply.')
    return parser.parse_args()


def main():
    """Open the viewer and resample on SPACE."""
    args = parse_args()
    rows = read_mark_rows(args.mark_list_file)
    viewer = TransformViewer(image_dir=args.image_dir, rows=rows, transform_name=args.transform)
    viewer.show_next()
    plt.show()


if __name__ == '__main__':
    main()
