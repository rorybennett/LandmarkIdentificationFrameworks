"""
Model registry for heatmap-regression networks.
"""

from .models import HRNetHeatmap, StackedHourglassHeatmap, UNetHeatmap, ViTPoseHeatmap, MedSAMHeatmap


AVAILABLE_MODELS = {
    'vit-medsam': {
        'description': 'Pretrained MedSAM ViT-B with staged landmark fine-tuning.',
        'module': 'Heatmaps.models.vit_medsam', 'class_name': 'MedSAMHeatmap',
        'builder': MedSAMHeatmap,
        'config_fields': ('decoder_channels', 'output_activation', 'final_kernel_size', 'gradient_checkpointing'),
        'paper_url': 'https://www.nature.com/articles/s41467-024-44824-z',
    },
    'unet_basic': {
        'description': 'Configurable U-Net for dense landmark heatmap regression.',
        'module': 'Heatmaps.models.unet',
        'class_name': 'UNetHeatmap',
        'builder': UNetHeatmap,
        'config_fields': ('base_channels', 'depth', 'channel_multiplier', 'max_channels', 'normalisation', 'activation', 'dropout', 'upsampling',
                          'output_activation', 'padding_mode', 'final_kernel_size'),
        'paper_url': 'https://arxiv.org/abs/1505.04597',
    },
    'hrnet': {
        'description': 'High-resolution multi-branch network with repeated multi-scale fusion.',
        'module': 'Heatmaps.models.hrnet',
        'class_name': 'HRNetHeatmap',
        'builder': HRNetHeatmap,
        'config_fields': ('hrnet_width', 'hrnet_modules', 'hrnet_blocks', 'normalisation', 'activation', 'dropout', 'output_activation', 'padding_mode',
                          'final_kernel_size'),
        'paper_url': 'https://arxiv.org/abs/1902.09212',
    },
    'stacked_hourglass': {
        'description': 'Stacked bottom-up/top-down hourglass model with intermediate heatmap supervision.',
        'module': 'Heatmaps.models.stacked_hourglass',
        'class_name': 'StackedHourglassHeatmap',
        'builder': StackedHourglassHeatmap,
        'config_fields': ('hourglass_features', 'hourglass_stacks', 'hourglass_depth', 'hourglass_blocks', 'normalisation', 'activation', 'dropout',
                          'output_activation', 'padding_mode', 'final_kernel_size'),
        'paper_url': 'https://arxiv.org/abs/1603.06937',
    },
    'vitpose': {
        'description': 'ViTPose-pretrained ViT-B encoder with a landmark decoder.',
        'module': 'Heatmaps.models.vitpose', 'class_name': 'ViTPoseHeatmap', 'builder': ViTPoseHeatmap,
        'config_fields': ('decoder_channels','output_activation','final_kernel_size','gradient_checkpointing'),
        'paper_url': 'https://github.com/ViTAE-Transformer/ViTPose',
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
    """Return the public registry entry for one model name."""
    network_name = str(network_name).lower()

    if network_name not in AVAILABLE_MODELS:
        raise ValueError(f'Unknown heatmap model: {network_name}')

    return {key: value for key, value in AVAILABLE_MODELS[network_name].items() if key != 'builder'}


def get_model_config_fields(network_name):
    """Return configuration field names used by one model."""
    network_name = str(network_name).lower()

    if network_name not in AVAILABLE_MODELS:
        raise ValueError(f'Unknown heatmap model: {network_name}')

    return tuple(AVAILABLE_MODELS[network_name]['config_fields'])


def get_model_kwargs(network_name, config):
    """Extract architecture-specific constructor values from a config object or dictionary."""
    fields = get_model_config_fields(network_name)
    source = config if isinstance(config, dict) else vars(config)
    missing = [field for field in fields if field not in source]

    if missing:
        raise ValueError(f'Missing configuration field(s) for {network_name}: {missing}')

    return {field: source[field] for field in fields}


def build_heatmap_model(network_name, num_of_points, input_channels, image_size, **kwargs):
    """Build a heatmap model from the registry."""
    network_name = str(network_name).lower()

    if network_name not in AVAILABLE_MODELS:
        raise ValueError(f'Unknown heatmap model: {network_name}')

    from .utils.io_utils import validate_canvas_size
    image_size = validate_canvas_size(image_size)
    model_info = AVAILABLE_MODELS[network_name]
    model_kwargs = {field: kwargs[field] for field in model_info['config_fields']}

    if network_name in ('vitpose', 'vit-medsam'):
        model_kwargs['image_size'] = image_size

    if network_name in ('vitpose','vit-medsam'):
        model_kwargs.update(pretrained_checkpoint=kwargs.get('pretrained_checkpoint'), initialise_pretrained=kwargs.get('initialise_pretrained', True))
    return model_info['builder'](num_of_points=num_of_points, input_channels=input_channels, **model_kwargs)
