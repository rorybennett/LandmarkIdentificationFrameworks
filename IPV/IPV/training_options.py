"""Shared CLI validation for optional IPV training controls."""
import math
from .landmark_losses import LANDMARK_CONSTRAINTS

DEFAULTS = {
    'lr_plateau_patience': 5, 'lr_decay_epochs': 50, 'lr_min_factor': 0.01,
    'visualise_validation_progress_images': 0, 'visualise_validation_progress_epochs': 0,
    'landmark_constraint_loss': None, 'constraint_angle_weight': 0.01,
    'constraint_side_weight': 0.01, 'constraint_temperature': 1.0, 'constraint_margin': 0.02,
}


def normalise_constraint(value):
    return {'none': None, '': None, 'prostate-taus': 'prostate_taus',
            'prostate_saus': 'prostate-saus'}.get(value, value)


def add_training_arguments(parser):
    descriptions = {
        'lr_plateau_patience': 'Epochs without validation-loss improvement before reducing LR.',
        'lr_decay_epochs': 'Epochs over which cosine/linear LR decays to its floor.',
        'lr_min_factor': 'Minimum LR as a fraction of the initial LR for plateau/cosine/linear/exponential schedules.',
        'visualise_validation_progress_images': 'Fixed validation images to export per interval; 0 disables.',
        'visualise_validation_progress_epochs': 'Export both best checkpoints every N epochs; 0 disables.',
        'constraint_angle_weight': 'Weight of transverse near-perpendicular-axis penalty.',
        'constraint_side_weight': 'Weight of landmark ordering and axis-collapse penalties.',
        'constraint_temperature': 'Softmax temperature for expected polar offsets (IPV logits).',
        'constraint_margin': 'Ordering margin as a fraction of original image diagonal.',
    }
    for name, default in DEFAULTS.items():
        if name == 'landmark_constraint_loss':
            parser.add_argument('--landmark-constraint-loss', type=normalise_constraint,
                choices=(None, *LANDMARK_CONSTRAINTS), default=None,
                help='Optional geometry: none, prostate_taus (4 transverse points), prostate-saus (2 sagittal points).')
        else:
            parser.add_argument('--' + name.replace('_', '-'), type=type(default), default=default,
                                help=descriptions[name])


def validate_training_options(config, num_points):
    kind = normalise_constraint(config.landmark_constraint_loss)
    config.landmark_constraint_loss = kind
    if kind is not None:
        if kind not in LANDMARK_CONSTRAINTS:
            raise ValueError(f'Unknown landmark constraint loss: {kind}')
        if num_points != LANDMARK_CONSTRAINTS[kind][0]:
            raise ValueError(f'{kind} requires {LANDMARK_CONSTRAINTS[kind][0]} landmarks.')
    for name in ('lr_plateau_patience', 'visualise_validation_progress_images', 'visualise_validation_progress_epochs'):
        value = getattr(config, name)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a non-negative integer.')
    if not isinstance(config.lr_decay_epochs, int) or config.lr_decay_epochs < 1:
        raise ValueError('lr_decay_epochs must be a positive integer.')
    if config.lr_schedule == 'plateau' and not 0 < config.lr_gamma < 1:
        raise ValueError('lr_gamma must be between 0 and 1 for plateau scheduling.')
    if not math.isfinite(config.lr_min_factor) or not 0 <= config.lr_min_factor <= 1:
        raise ValueError('lr_min_factor must be between 0 and 1.')
    for name in ('constraint_angle_weight', 'constraint_side_weight', 'constraint_margin'):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and non-negative.')
    if not math.isfinite(config.constraint_temperature) or config.constraint_temperature <= 0:
        raise ValueError('constraint_temperature must be finite and positive.')
