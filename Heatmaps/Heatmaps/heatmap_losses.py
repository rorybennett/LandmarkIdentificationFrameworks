"""Differentiable geometry evaluated in the pre-augmentation image frame."""

import math
import torch
import torch.nn.functional as F
from .utils.io_utils import scale_points_to_original


def heatmap_coordinates(heatmaps, valid_mask, temperature):
    values = heatmaps.float().flatten(2) / temperature
    mask = valid_mask.bool().expand_as(heatmaps).flatten(2)
    probabilities = values.masked_fill(~mask, float('-inf')).softmax(-1)
    height, width = heatmaps.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=heatmaps.device, dtype=torch.float32),
        torch.arange(width, device=heatmaps.device, dtype=torch.float32),
        indexing='ij',
    )
    return torch.stack(((probabilities * xx.flatten()).sum(-1), (probabilities * yy.flatten()).sum(-1)), -1)


def prostate_taus_terms(points, margin=0.02):
    """Points are normalised with one common scale per image; y increases downwards."""
    p1, p2, p3, p4 = points.unbind(1)
    vertical, horizontal = p3 - p1, p2 - p4
    vn = torch.linalg.vector_norm(vertical, dim=-1).clamp_min(1e-6)
    hn = torch.linalg.vector_norm(horizontal, dim=-1).clamp_min(1e-6)
    cosine = (vertical * horizontal).sum(-1) / (vn * hn)
    angle = F.relu(cosine.abs() - math.sin(math.radians(20))).square()
    # Orient each line normal to the original image's down/right directions.
    down = torch.stack((-horizontal[:, 1], horizontal[:, 0]), -1) / hn[:, None]
    down = down * torch.where(down[:, 1:2] < 0, -1.0, 1.0)
    right = torch.stack((vertical[:, 1], -vertical[:, 0]), -1) / vn[:, None]
    right = right * torch.where(right[:, 0:1] < 0, -1.0, 1.0)
    hc, vc = (p2 + p4) / 2, (p1 + p3) / 2
    distances = torch.stack(
        (
            -(p1 - hc).mul(down).sum(-1),
            (p3 - hc).mul(down).sum(-1),
            (p2 - vc).mul(right).sum(-1),
            -(p4 - vc).mul(right).sum(-1),
        ),
        -1,
    )
    sides = F.relu(margin - distances).square().mean(-1)
    # Collapsed axes must not satisfy the constraint through epsilon denominators.
    sides = sides + F.relu(2 * margin - vn).square() + F.relu(2 * margin - hn).square()
    return angle.mean(), sides.mean()


def constraint_terms(
    heatmaps,
    valid_mask,
    original_sizes,
    inverse_augmentation,
    temperature=0.05,
    margin=0.02,
    constraint_type="prostate_taus",
):
    expected, calculation = LANDMARK_CONSTRAINTS[constraint_type]
    if heatmaps.shape[1] != expected:
        raise ValueError(f'{constraint_type} requires {expected} landmarks.')
    points = heatmap_coordinates(heatmaps, valid_mask, temperature)
    points = scale_points_to_original(points, original_sizes, heatmaps.shape[-1])
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    points = torch.bmm(homogeneous, inverse_augmentation.float().transpose(1, 2))[..., :2]
    scale = torch.linalg.vector_norm(original_sizes.float(), dim=-1).clamp_min(1)
    points = points / scale[:, None, None]
    return calculation(points, margin)


def prostate_saus_terms(points, margin=0.02):
    return points.sum() * 0, F.relu(margin - (points[:, 1, 1] - points[:, 0, 1])).square().mean()


# Register (landmark count, callable(points, margin) -> (angle_term, side_term)).
# Coordinates are restored to the unaugmented image and scaled by its diagonal.
LANDMARK_CONSTRAINTS = {
    'prostate_taus': (4, prostate_taus_terms),
    'prostate-saus': (2, prostate_saus_terms),
}


from torch import nn


class WeightedMSELoss(nn.Module):
    """Apply stronger loss near landmark heatmap peaks."""

    def __init__(self, positive_weight=20.0):
        super().__init__()
        self.positive_weight = float(positive_weight)

    def forward(self, outputs, targets):
        weights = 1.0 + targets * self.positive_weight
        return weights * (outputs - targets) ** 2


HEATMAP_LOSSES = {
    'mse': lambda config: nn.MSELoss(reduction='none'),
    'weighted_mse': lambda config: WeightedMSELoss(config.positive_weight),
    'smooth_l1': lambda config: nn.SmoothL1Loss(reduction='none'),
    'bce_logits': lambda config: nn.BCEWithLogitsLoss(reduction='none'),
}


def build_heatmap_loss(config):
    try:
        return HEATMAP_LOSSES[config.loss_name](config)
    except KeyError as error:
        raise ValueError(f'Unknown heatmap loss: {config.loss_name}') from error
