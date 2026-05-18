"""
Model registry for available IPV network backbones.
"""

AVAILABLE_MODELS = {
    "resnet18_pretrained": {
        "description": "ResNet-18 branch using torchvision ImageNet pretrained weights.",
        "pretrained": True,
        "supports_frozen_stages": True,
        "supports_small_input_stem": True,
        "recommended_small_input_stem": False,
    },
    "resnet18_untrained": {
        "description": "ResNet-18 branch trained from scratch.",
        "pretrained": False,
        "supports_frozen_stages": False,
        "supports_small_input_stem": True,
        "recommended_small_input_stem": True,
    },
    "resnet34_pretrained": {
        "description": "ResNet-34 branch using torchvision ImageNet pretrained weights.",
        "pretrained": True,
        "supports_frozen_stages": True,
        "supports_small_input_stem": True,
        "recommended_small_input_stem": False,
    },
    "resnet34_untrained": {
        "description": "ResNet-34 branch trained from scratch.",
        "pretrained": False,
        "supports_frozen_stages": False,
        "supports_small_input_stem": True,
        "recommended_small_input_stem": True,
    },
    "resnet10_untrained": {
        "description": "Small custom ResNet branch with layer config [1, 1, 1, 1].",
        "pretrained": False,
        "supports_frozen_stages": False,
        "supports_small_input_stem": True,
        "recommended_small_input_stem": True,
    },
    "resnet14_untrained": {
        "description": "Small custom ResNet branch with layer config [1, 1, 2, 2].",
        "pretrained": False,
        "supports_frozen_stages": False,
        "supports_small_input_stem": True,
        "recommended_small_input_stem": True,
    },
    "small_cnn": {
        "description": "Lightweight CNN branch for 64x64 patch inputs.",
        "pretrained": False,
        "supports_frozen_stages": False,
        "supports_small_input_stem": False,
        "recommended_small_input_stem": False,
    },
}


def get_available_model_names():
    """Return available model names."""
    return tuple(AVAILABLE_MODELS.keys())


def print_available_models():
    """Print available model names and descriptions."""
    for model_name, model_info in AVAILABLE_MODELS.items():
        print(f"{model_name}: {model_info['description']}")