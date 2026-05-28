"""
Visualisation helpers for validation outputs.
"""

from pathlib import Path

import cv2
import numpy as np
from skimage import io
from skimage.util import img_as_ubyte

POINT_COLOURS = ((0, 0, 255), (255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (128, 0, 255), (255, 128, 0))
GROUND_TRUTH_COLOUR = (0, 255, 0)
PREDICTED_COLOUR = (0, 0, 255)


def load_display_image(image_path):
    """Load an image as BGR uint8 for OpenCV drawing."""
    image = io.imread(image_path)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        image = img_as_ubyte(image)

    return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)


def normalise_map(value_map):
    """Normalise a map into the 0 to 255 range."""
    value_map = np.asarray(value_map, dtype=np.float32)
    value_min = float(np.min(value_map))
    value_max = float(np.max(value_map))

    if value_max <= value_min:
        return np.zeros(value_map.shape, dtype=np.uint8)

    return ((value_map - value_min) / (value_max - value_min) * 255).astype(np.uint8)


def resize_heatmaps_to_display(heatmaps, display_shape):
    """Resize heatmaps from model resolution to display-image resolution."""
    heatmaps = np.asarray(heatmaps, dtype=np.float32)

    if heatmaps.ndim == 2:
        heatmaps = heatmaps[np.newaxis, :, :]

    if heatmaps.ndim != 3:
        raise ValueError(f'Expected heatmaps with shape [points, height, width], got {heatmaps.shape}.')

    display_height, display_width = int(display_shape[0]), int(display_shape[1])
    resized_heatmaps = []

    for heatmap in heatmaps:
        if heatmap.shape != (display_height, display_width):
            heatmap = cv2.resize(heatmap, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

        resized_heatmaps.append(heatmap.astype(np.float32))

    return np.stack(resized_heatmaps, axis=0)


def create_combined_heatmap_overlay(display_image, heatmaps, alpha=0.55):
    """Overlay all predicted heatmaps on one image."""
    heatmaps = resize_heatmaps_to_display(heatmaps=heatmaps, display_shape=display_image.shape)
    combined = np.max(heatmaps, axis=0)
    heatmap = cv2.applyColorMap(normalise_map(combined), cv2.COLORMAP_JET)

    if heatmap.shape != display_image.shape:
        raise ValueError(f'Heatmap overlay shape {heatmap.shape} does not match display image shape {display_image.shape}.')

    return cv2.addWeighted(display_image, 1.0 - float(alpha), heatmap, float(alpha), 0)


def create_endpoint_overlay(display_image, target_points, predicted_points):
    """Draw ground truth and predicted landmarks on one image."""
    overlay = display_image.copy()

    for point in target_points:
        cv2.drawMarker(overlay, (int(round(point[0])), int(round(point[1]))), GROUND_TRUTH_COLOUR, markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2, line_type=cv2.LINE_AA)

    for point in predicted_points:
        cv2.circle(overlay, (int(round(point[0])), int(round(point[1]))), 4, PREDICTED_COLOUR, thickness=-1, lineType=cv2.LINE_AA)

    return overlay


def save_validation_overlays(image_path, output_dir, output_stem, target_points, predicted_points, predicted_heatmaps):
    """Save heatmap and endpoint overlays for one validation image."""
    output_dir = Path(output_dir)
    heatmap_dir = output_dir / 'heatmap_overlays'
    endpoint_dir = output_dir / 'endpoint_overlays'
    heatmap_dir.mkdir(exist_ok=True, parents=True)
    endpoint_dir.mkdir(exist_ok=True, parents=True)

    display_image = load_display_image(image_path)
    heatmap_overlay = create_combined_heatmap_overlay(display_image=display_image, heatmaps=predicted_heatmaps)
    endpoint_overlay = create_endpoint_overlay(display_image=display_image, target_points=target_points, predicted_points=predicted_points)

    cv2.imwrite(str(heatmap_dir / f'{output_stem}_heatmap_overlay.png'), heatmap_overlay)
    cv2.imwrite(str(endpoint_dir / f'{output_stem}_endpoint_overlay.png'), endpoint_overlay)
