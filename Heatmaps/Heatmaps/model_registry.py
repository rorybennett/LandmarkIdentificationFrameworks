"""
Model registry for heatmap-regression networks.
"""

from .models import UNetHeatmap

AVAILABLE_MODELS = {
    'unet_basic': {
        'description': 'Configurable U-Net for dense landmark heatmap regression.',
        'supports_dropout': True,
        'supports_output_activation': True,
    },
}


def get_available_model_names():
    """Return available heatmap model names."""
    return tuple(AVAILABLE_MODELS.keys())


def print_available_models():
    """Print model names and descriptions."""
    for model_name, model_info in AVAILABLE_MODELS.items():
        print(f"{model_name}: {model_info['description']}")


def build_heatmap_model(network_name, num_of_points, input_channels, **kwargs):
    """Build a heatmap model from the registry."""
    network_name = str(network_name).lower()

    if network_name == 'unet_basic':
        return UNetHeatmap(num_of_points=num_of_points, input_channels=input_channels, **kwargs)

    raise ValueError(f'Unknown heatmap model: {network_name}')
