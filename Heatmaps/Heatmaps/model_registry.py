"""
Model registry for heatmap-regression networks.
"""

from .models import UNetHeatmap

AVAILABLE_MODELS = {
    'unet_basic': {
        'description': 'Configurable U-Net for dense landmark heatmap regression.',
        'module': 'Heatmaps.models',
        'class_name': 'UNetHeatmap',
    },
}


def get_available_model_names():
    """Return available heatmap model names."""
    return tuple(AVAILABLE_MODELS.keys())


def print_available_models():
    """Print model names and descriptions."""
    for model_name, model_info in AVAILABLE_MODELS.items():
        print(f"{model_name}: {model_info['description']}")


def get_model_registry_entry(network_name):
    """Return the registry entry for one model name."""
    network_name = str(network_name).lower()

    if network_name not in AVAILABLE_MODELS:
        raise ValueError(f'Unknown heatmap model: {network_name}')

    return dict(AVAILABLE_MODELS[network_name])


def build_heatmap_model(network_name, num_of_points, input_channels, **kwargs):
    """Build a heatmap model from the registry."""
    network_name = str(network_name).lower()

    get_model_registry_entry(network_name)

    if network_name == 'unet_basic':
        return UNetHeatmap(num_of_points=num_of_points, input_channels=input_channels, **kwargs)

    raise RuntimeError(f'No builder is registered for heatmap model: {network_name}')
