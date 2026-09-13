"""Heatmap model implementations. Version 0.1 architecture contract."""

from .unet import UNetHeatmap
from .hrnet import HRNetHeatmap
from .stacked_hourglass import StackedHourglassHeatmap
from .vitpose import ViTPoseHeatmap
from .vit_medsam import MedSAMHeatmap
from .common import count_trainable_parameters, unpack_heatmap_output
