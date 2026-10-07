"""
Default heatmap package parameters.
"""

image_size = 512
heatmap_sigma = 8.0
supported_image_suffixes = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def get_training_defaults(network_name):
    """Return CLI defaults from the bundled normal-loss recommendation.

    Source run paths and pretrained hashes are provenance only: users still
    supply their own dataset paths and local pretrained checkpoint. Input
    channels are resolved by the existing greyscale preprocessing policy.
    """
    import json
    from importlib.resources import files

    reference = json.loads(files(__package__).joinpath('recommended_normal_loss_defaults.json').read_text(encoding='utf-8'))
    model = reference['models'][network_name]
    defaults = {}
    for section in ('data_config', 'model_config', 'train_config'):
        defaults.update(reference['common'].get(section, {}))
        defaults.update(model.get(section, {}))
    defaults.pop('input_channels')
    defaults['train_workers'] = defaults.pop('num_workers')
    return defaults
