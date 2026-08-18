"""
Heatmap landmark localisation package.
"""

import importlib

from .heatmap_transforms import get_default_heatmap_transforms
from .model_registry import build_heatmap_model, get_available_model_names
from .models import HRNetHeatmap, StackedHourglassHeatmap, UNetHeatmap, ViTPoseHeatmap, count_trainable_parameters

__version__ = "0.1.0"

_TRAINING_EXPORTS = {'HeatmapDataConfig', 'HeatmapModelConfig', 'TrainConfig', 'TrainModel'}


def __getattr__(name):
    """Load training exports lazily so utility viewers do not inherit the headless plotting backend."""
    if name in _TRAINING_EXPORTS:
        train_model_module = importlib.import_module('.train_model', __name__)
        value = getattr(train_model_module, name)
        globals()[name] = value
        return value

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = [
    '__version__',
    'UNetHeatmap',
    'HRNetHeatmap',
    'StackedHourglassHeatmap',
    'ViTPoseHeatmap',
    'count_trainable_parameters',
    'get_default_heatmap_transforms',
    'build_heatmap_model',
    'get_available_model_names',
    'HeatmapDataConfig',
    'HeatmapModelConfig',
    'TrainConfig',
    'TrainModel',
]
