"""
Visualisation helpers for validation outputs.
"""

from pathlib import Path

import cv2
import numpy as np

GROUND_TRUTH_POINT_COLOUR = (0, 255, 0)
PREDICTED_POINT_COLOUR = (0, 0, 255)
POINT_MARKER_SIZE = 16
POINT_MARKER_THICKNESS = 2
HEATMAP_IMAGE_WEIGHT = 0.55
HEATMAP_COLOUR_WEIGHT = 0.90


def load_display_image(image_path):
    """Load one source image as BGR uint8 for OpenCV drawing."""
    display_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if display_image is None:
        raise FileNotFoundError(f'Could not read image for display: {image_path}')

    return display_image


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


def create_combined_heatmap_overlay(display_image, heatmaps, predicted_points=None):
    """Overlay all predicted heatmaps on one image and label predicted endpoints."""
    heatmaps = resize_heatmaps_to_display(heatmaps=heatmaps, display_shape=display_image.shape)
    combined = np.max(heatmaps, axis=0)
    heatmap = cv2.applyColorMap(normalise_map(combined), cv2.COLORMAP_JET)

    if heatmap.shape != display_image.shape:
        raise ValueError(f'Heatmap overlay shape {heatmap.shape} does not match display image shape {display_image.shape}.')

    overlay = cv2.addWeighted(display_image, HEATMAP_IMAGE_WEIGHT, heatmap, HEATMAP_COLOUR_WEIGHT, 0)

    if predicted_points is not None:
        draw_points(image=overlay, points=predicted_points, colour=PREDICTED_POINT_COLOUR, prefix='P')

    return overlay


def create_point_overlay(display_image, detected_points, ground_truth_points=None):
    """Create one image containing IPV-style predicted and optional ground-truth endpoints."""
    overlay = display_image.copy()

    if ground_truth_points is not None:
        draw_points(image=overlay, points=ground_truth_points, colour=GROUND_TRUTH_POINT_COLOUR, prefix='G')

    draw_points(image=overlay, points=detected_points, colour=PREDICTED_POINT_COLOUR, prefix='P')
    return overlay


def draw_points(image, points, colour, prefix):
    """Draw labelled endpoints onto an image."""
    for point_index, (x, y) in enumerate(points, start=1):
        centre = (int(round(x)), int(round(y)))
        cv2.drawMarker(image, centre, colour, markerType=cv2.MARKER_TILTED_CROSS, markerSize=POINT_MARKER_SIZE, thickness=POINT_MARKER_THICKNESS, line_type=cv2.LINE_AA)
        cv2.putText(image, f'{prefix}{point_index}', (centre[0] + 6, centre[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def save_validation_overlays(image_path, output_dir, output_stem, target_points, predicted_points, predicted_heatmaps):
    """Save heatmap and point overlays for one validation image."""
    output_dir = Path(output_dir)
    heatmap_dir = output_dir / 'heatmap_overlays'
    point_dir = output_dir / 'point_overlays'
    heatmap_dir.mkdir(exist_ok=True, parents=True)
    point_dir.mkdir(exist_ok=True, parents=True)

    display_image = load_display_image(image_path)
    heatmap_overlay = create_combined_heatmap_overlay(display_image=display_image, heatmaps=predicted_heatmaps, predicted_points=predicted_points)
    point_overlay = create_point_overlay(display_image=display_image, detected_points=predicted_points, ground_truth_points=target_points)

    cv2.imwrite(str(heatmap_dir / f'{output_stem}_validation_heatmap_overlay.png'), heatmap_overlay)
    cv2.imwrite(str(point_dir / f'{output_stem}_validation_points_overlay.png'), point_overlay)
