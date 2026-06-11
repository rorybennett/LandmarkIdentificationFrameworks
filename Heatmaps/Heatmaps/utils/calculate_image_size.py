"""
Calculate the average image size for a folder of training images.

Edit the path and switches below, then run from the repository root with:
python -m Heatmaps.utils.calculate_image_size
"""
from pathlib import Path

import numpy as np
from skimage import io

# ======================================================================================================================
# Paths
# ======================================================================================================================
IMAGE_DATA_DIR = Path(r'D:\Datasets\Heatmaps\OriginalData\TRANSVERSE')

# ======================================================================================================================
# Switches
# ======================================================================================================================
RECURSIVE_IMAGE_SEARCH = False
PRINT_EACH_IMAGE = False
SUPPORTED_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def find_images(image_data_dir, recursive=False, supported_suffixes=SUPPORTED_IMAGE_SUFFIXES):
    """Return supported image paths under the requested folder."""
    image_data_dir = Path(image_data_dir)

    if not image_data_dir.is_dir():
        raise ValueError(f'IMAGE_DATA_DIR does not exist or is not a directory: {image_data_dir}')

    suffixes = tuple(suffix.lower() for suffix in supported_suffixes)
    iterator = image_data_dir.rglob('*') if recursive else image_data_dir.iterdir()
    return sorted([path for path in iterator if path.is_file() and path.suffix.lower() in suffixes], key=lambda path: path.as_posix().lower())


def read_image_size(image_path):
    """Return one image size as height and width."""
    image = io.imread(image_path)

    if image.ndim < 2:
        raise ValueError(f'Unsupported image shape for {image_path}: {image.shape}')

    return int(image.shape[0]), int(image.shape[1])


def calculate_average_size(image_paths):
    """Calculate mean and rounded image sizes from a list of image paths."""
    sizes = [read_image_size(image_path) for image_path in image_paths]

    if not sizes:
        raise ValueError(f'No supported images found in {IMAGE_DATA_DIR}')

    heights = np.asarray([height for height, _ in sizes], dtype=np.float64)
    widths = np.asarray([width for _, width in sizes], dtype=np.float64)
    return sizes, float(np.mean(widths)), float(np.mean(heights)), int(round(float(np.mean(widths)))), int(round(float(np.mean(heights))))


def print_summary(image_paths, sizes, average_width, average_height, rounded_width, rounded_height):
    """Print the image-size summary."""
    if PRINT_EACH_IMAGE:
        for image_path, (height, width) in zip(image_paths, sizes):
            print(f'{image_path.name}: width={width}, height={height}')

    print(f'Image folder: {IMAGE_DATA_DIR}')
    print(f'Image count: {len(image_paths)}')
    print(f'Average width: {average_width:.2f}')
    print(f'Average height: {average_height:.2f}')
    print(f'Rounded width: {rounded_width}')
    print(f'Rounded height: {rounded_height}')
    print(f'Use with heatmaps-train as: --image-size {rounded_height} {rounded_width}')


def main():
    """Calculate and print average image dimensions."""
    image_paths = find_images(image_data_dir=IMAGE_DATA_DIR, recursive=RECURSIVE_IMAGE_SEARCH, supported_suffixes=SUPPORTED_IMAGE_SUFFIXES)
    sizes, average_width, average_height, rounded_width, rounded_height = calculate_average_size(image_paths)
    print_summary(image_paths=image_paths, sizes=sizes, average_width=average_width, average_height=average_height,
                  rounded_width=rounded_width, rounded_height=rounded_height)


if __name__ == '__main__':
    main()
