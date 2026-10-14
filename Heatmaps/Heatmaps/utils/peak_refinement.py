"""Inference-only DARK-style local log-heatmap Taylor refinement.

Research basis: Zhang et al., CVPR 2020, https://arxiv.org/abs/1910.06278.
This adaptation preserves the framework's pixel-centre letterboxing and adds
explicit fallbacks for borders, non-positive peaks and unstable local fits.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .io_utils import heatmaps_to_points, letterbox_geometry


@dataclass(frozen=True)
class PeakRefinementConfig:
    blur_kernel: int = 3
    blur_sigma: float = 0.0  # OpenCV derives sigma from the kernel when zero.
    max_offset: float = 1.0  # Pixels of the heatmap being refined.
    epsilon: float = 1e-10
    min_curvature: float = 1e-8
    max_condition: float = 1e6

    def __post_init__(self):
        if type(self.blur_kernel) is not int or self.blur_kernel < 1 or self.blur_kernel % 2 != 1:
            raise ValueError('blur_kernel must be a positive odd integer (1 disables smoothing).')
        for name in ('blur_sigma', 'max_offset', 'epsilon', 'min_curvature', 'max_condition'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name != 'blur_sigma' and value == 0):
                raise ValueError(f'{name} must be finite and {"non-negative" if name == "blur_sigma" else "positive"}.')
        if self.max_condition < 1:
            raise ValueError('max_condition must be at least 1.')


def refine_peak(content, peak, config):
    """Return a local offset and status, using only valid image-content pixels."""
    x, y = map(int, peak)
    if content[y, x] <= config.epsilon:
        return None, 'non_positive_peak'
    if float(np.ptp(content)) <= config.epsilon:
        return None, 'flat_heatmap'
    radius = 2 + config.blur_kernel // 2
    if min(x, y, content.shape[1] - 1 - x, content.shape[0] - 1 - y) < radius:
        return None, 'content_boundary'

    # Restrict smoothing to the neighbourhood needed for the derivative stencil.
    patch = content[y-radius:y+radius+1, x-radius:x+radius+1].astype(np.float64, copy=True)
    if config.blur_kernel > 1:
        patch = cv2.GaussianBlur(patch, (config.blur_kernel, config.blur_kernel), config.blur_sigma)
    if patch[radius, radius] <= config.epsilon:
        return None, 'non_positive_smoothed_peak'
    log_map = np.log(np.maximum(patch, config.epsilon))
    p = radius
    centre = log_map[p, p]
    gradient = np.array([(log_map[p, p+1] - log_map[p, p-1]) / 2,
                         (log_map[p+1, p] - log_map[p-1, p]) / 2])
    xx = (log_map[p, p+2] + log_map[p, p-2] - 2 * centre) / 4
    yy = (log_map[p+2, p] + log_map[p-2, p] - 2 * centre) / 4
    xy = (log_map[p+1, p+1] + log_map[p-1, p-1] - log_map[p+1, p-1] - log_map[p-1, p+1]) / 4
    hessian = np.array([[xx, xy], [xy, yy]])
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[-1] >= -config.min_curvature:
        return None, 'non_concave_or_flat_fit'
    if abs(eigenvalues[0] / eigenvalues[-1]) > config.max_condition:
        return None, 'ill_conditioned_fit'
    offset = np.linalg.solve(hessian, -gradient)
    if not np.isfinite(offset).all() or np.max(np.abs(offset)) > config.max_offset:
        return None, 'excessive_offset'
    return offset, 'refined'


def decode_peak_comparison(canvas_heatmaps, original_sizes, image_size, config=None, native_heatmaps=None):
    """Decode shared predictions; return canvas coordinates and per-point statuses.

    ``argmax`` calls the existing decoder unchanged. With native heatmaps, a
    ``native_argmax`` control isolates the change of resolution from refinement.
    Every rejected fit falls back to the original canvas argmax exactly.
    """
    config = config or PeakRefinementConfig()
    if canvas_heatmaps.ndim != 4 or not torch.isfinite(canvas_heatmaps).all():
        raise ValueError('Expected finite [batch, landmarks, height, width] heatmaps.')
    if len(original_sizes) != canvas_heatmaps.shape[0]:
        raise ValueError('Each heatmap batch item requires an original image size.')
    baseline = heatmaps_to_points(canvas_heatmaps, original_sizes, image_size)
    maps = canvas_heatmaps if native_heatmaps is None else native_heatmaps
    if maps.ndim != 4 or maps.shape[:2] != canvas_heatmaps.shape[:2] or not torch.isfinite(maps).all():
        raise ValueError('Native heatmaps must be finite and match the batch and landmark counts.')
    if min(maps.shape[-2:]) < 1:
        raise ValueError('Heatmaps must have non-empty spatial dimensions.')
    arrays = maps.detach().cpu().numpy().astype(np.float64)
    refined = baseline.clone()
    native_argmax = baseline.clone()
    statuses = []
    height, width = maps.shape[-2:]
    sx, sy = image_size / width, image_size / height
    # align_corners=False: preserve pixel centres when mapping the native grid.
    canvas_x = (np.arange(width) + 0.5) * sx - 0.5
    canvas_y = (np.arange(height) + 0.5) * sy - 0.5
    for batch_index, size in enumerate(original_sizes):
        rh, rw, top, left = letterbox_geometry(size, image_size)
        xs = np.flatnonzero((canvas_x >= left) & (canvas_x <= left + rw - 1))
        ys = np.flatnonzero((canvas_y >= top) & (canvas_y <= top + rh - 1))
        batch_status = []
        for landmark, heatmap in enumerate(arrays[batch_index]):
            if not len(xs) or not len(ys):
                batch_status.append('no_native_content_pixel')
                continue
            x0, x1, y0, y1 = xs[0], xs[-1] + 1, ys[0], ys[-1] + 1
            content = heatmap[y0:y1, x0:x1]
            y, x = np.unravel_index(np.argmax(content), content.shape)
            native_argmax[batch_index, landmark] = baseline.new_tensor([canvas_x[x+x0], canvas_y[y+y0]])
            offset, status = refine_peak(content, (x, y), config)
            batch_status.append(status)
            if offset is not None:
                mapped = [(x + x0 + offset[0] + 0.5) * sx - 0.5,
                          (y + y0 + offset[1] + 0.5) * sy - 0.5]
                if not (left <= mapped[0] <= left + rw - 1 and top <= mapped[1] <= top + rh - 1):
                    batch_status[-1] = 'outside_content'
                    continue
                refined[batch_index, landmark] = baseline.new_tensor(mapped)
        statuses.append(batch_status)
    points = {'argmax': baseline, 'dark': refined}
    if native_heatmaps is not None:
        points['native_argmax'] = native_argmax
    return points, statuses
