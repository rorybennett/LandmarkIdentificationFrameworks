"""Differentiable geometry for IPV's distance/angle classification heads.

Unlike hard voting, expected polar offsets propagate gradients to both heads.
IPV angles run from landmark to patch centre, hence the negative displacement.
Each patch predicts all landmarks in the original, unaugmented image frame.
"""
import math
import torch
import torch.nn.functional as F

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

def prostate_saus_terms(points, margin=0.02):
    return points.sum() * 0, F.relu(margin - (points[:, 1, 1] - points[:, 0, 1])).square().mean()

LANDMARK_CONSTRAINTS = {
    'prostate_taus': (4, prostate_taus_terms),
    'prostate-saus': (2, prostate_saus_terms),
}


def expected_landmark_offsets(outputs, tasks_classes, original_sizes, temperature=1.0):
    """Expected offsets from patch centre, normalised by the image diagonal.

    Use distance-bin midpoints and circular (cos/sin) angle-bin midpoints, avoiding
    the discontinuity at 0/360 degrees. Translation cancels in all geometry terms.
    This is a per-patch surrogate, not the non-differentiable final voting result.
    """
    device = outputs[0].device
    scale = torch.linalg.vector_norm(original_sizes.to(device=device, dtype=torch.float32), dim=-1).clamp_min(1)
    radii = torch.tensor(tasks_classes[0], device=device, dtype=torch.float32).mean(-1)
    angles = torch.tensor(tasks_classes[1], device=device, dtype=torch.float32).mean(-1) * (math.pi / 180)
    directions = torch.stack((angles.cos(), angles.sin()), -1)
    points = []
    for index in range(0, len(outputs), 2):
        distance = (outputs[index].float() / temperature).softmax(-1) @ radii
        direction = (outputs[index + 1].float() / temperature).softmax(-1) @ directions
        points.append(-distance[:, None] * direction / scale[:, None])
    return torch.stack(points, 1)


def constraint_terms(outputs, tasks_classes, original_sizes, temperature=1.0,
                     margin=0.02, constraint_type='prostate_taus'):
    expected, calculation = LANDMARK_CONSTRAINTS[constraint_type]
    if len(outputs) != expected * 2:
        raise ValueError(f'{constraint_type} requires {expected} landmarks with distance and angle heads.')
    points = expected_landmark_offsets(outputs, tasks_classes, original_sizes, temperature)
    return calculation(points, margin)
