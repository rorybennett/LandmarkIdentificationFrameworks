"""
Heatmap landmark localisation package.
"""

from .heatmap_transforms import get_default_heatmap_transforms
from .model_registry import build_heatmap_model, get_available_model_names
from .models import HRNetHeatmap, StackedHourglassHeatmap, UNetHeatmap, ViTPoseHeatmap, count_trainable_parameters
from .train_model import HeatmapDataConfig, HeatmapModelConfig, TrainConfig, TrainModel

__version__ = "0.1.0"

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
