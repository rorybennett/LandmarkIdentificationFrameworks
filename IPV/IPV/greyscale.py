"""Consistent RGB luminance conversion for model inputs."""
import numpy as np


def to_three_channel_greyscale(image):
    """Convert HWC RGB/RGBA (alpha ignored) or greyscale to three equal channels."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        grey = image
    elif image.ndim == 3 and image.shape[-1] == 1:
        grey = image[..., 0]
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        # Difference form preserves already equal channels exactly, including white.
        grey = image[..., 1] + np.float32(0.299) * (image[..., 0] - image[..., 1]) + np.float32(0.114) * (image[..., 2] - image[..., 1])
    else:
        raise ValueError(f'Unsupported image shape for greyscale conversion: {image.shape}')
    return np.repeat(np.clip(grey, 0, 1)[..., None], 3, axis=-1)
