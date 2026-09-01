import importlib

from .model_registry import AVAILABLE_MODELS, get_available_model_names, print_available_models

__version__ = "0.1"


def __getattr__(name):
    """Load inference exports lazily so fold utilities do not require the ML runtime."""
    if name == 'LandmarkInferenceConfig':
        module = importlib.import_module('.utils.landmark_inference_utils', __name__)
        value = module.LandmarkInferenceConfig
        globals()[name] = value
        return value

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

__all__ = [
    "AVAILABLE_MODELS",
    "get_available_model_names",
    "print_available_models",
    "LandmarkInferenceConfig",
]
