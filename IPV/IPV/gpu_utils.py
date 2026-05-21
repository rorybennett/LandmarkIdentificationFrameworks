import numpy as np
from numba import njit


@njit(cache=True)
def check_current_labels(labels_count, current_labels):
    """Check whether current labels are usable against the current label counts."""
    for point_index in range(len(current_labels)):
        current_label = current_labels[point_index]

        if current_label == 0:
            return False

        if labels_count[point_index][current_label] == labels_count[point_index][0]:
            return False

    return True


@njit(cache=True)
def check_labels(labels_count):
    """Check whether label counts are sufficiently balanced."""
    good_count = 0

    for point_index in range(len(labels_count)):
        good_count_for_point = 0
        reference_count = labels_count[point_index][0]

        for class_index in range(len(labels_count[point_index])):
            class_count = labels_count[point_index][class_index]

            if reference_count <= class_count <= reference_count * 2:
                good_count_for_point += 1

        if good_count_for_point == len(labels_count[point_index]):
            good_count += 1

    return good_count != len(labels_count)


@njit(cache=True)
def get_label(value, intervals):
    """Return the class index for a value given interval boundaries."""
    for interval_index in range(len(intervals)):
        lower_bound = intervals[interval_index][0]
        upper_bound = intervals[interval_index][1]

        if lower_bound <= value < upper_bound:
            return interval_index

    return -1


@njit(cache=True)
def get_angle(point_1, point_2):
    """Return the angle from point_1 to point_2 in degrees from 0 to 360."""
    angle = np.arctan2(point_2[1] - point_1[1], point_2[0] - point_1[0]) * 180 / np.pi

    if angle < 0:
        angle += 360

    return angle


def create_patch(image, x, y, patch_size):
    """Create a square patch centred on x, y with zero-padding outside the image."""
    x = int(x)
    y = int(y)
    half_patch = patch_size // 2

    if image.ndim == 2:
        patch = np.zeros((patch_size, patch_size), dtype=image.dtype)
    elif image.ndim == 3:
        patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
    else:
        raise ValueError(f'Image must be 2D or 3D, got shape {image.shape}.')

    row_start = y - half_patch
    row_end = row_start + patch_size
    col_start = x - half_patch
    col_end = col_start + patch_size

    source_row_start = max(row_start, 0)
    source_row_end = min(row_end, image.shape[0])
    source_col_start = max(col_start, 0)
    source_col_end = min(col_end, image.shape[1])

    if source_row_start >= source_row_end or source_col_start >= source_col_end:
        return patch

    patch_row_start = source_row_start - row_start
    patch_col_start = source_col_start - col_start
    patch_row_end = patch_row_start + (source_row_end - source_row_start)
    patch_col_end = patch_col_start + (source_col_end - source_col_start)

    patch[patch_row_start:patch_row_end, patch_col_start:patch_col_end, ...] = image[source_row_start:source_row_end, source_col_start:source_col_end, ...]

    return patch


@njit(cache=True)
def concat_patches(patches):
    """Concatenate four patches into a two-by-two grid."""
    top_row = np.concatenate((patches[0], patches[1]), axis=1)
    bottom_row = np.concatenate((patches[2], patches[3]), axis=1)

    return np.concatenate((top_row, bottom_row), axis=0)
