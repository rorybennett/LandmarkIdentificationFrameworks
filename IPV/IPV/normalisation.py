"""Three-channel input normalisation utilities for IPV."""

import numpy as np
import torch


EXPECTED_NORMALISATION_CHANNELS = 3
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


class ChannelStatistics:
    """Accumulate population statistics without retaining image pixels."""

    def __init__(self, channels=EXPECTED_NORMALISATION_CHANNELS):
        self.channels = int(channels)
        self.count = 0
        self.mean = np.zeros(self.channels, dtype=np.float64)
        self.m2 = np.zeros(self.channels, dtype=np.float64)

    def update(self, image):
        """Add one channel-first image or batch of images to the estimate."""
        values = np.asarray(image, dtype=np.float64)

        if values.ndim < 3:
            raise ValueError(f'Expected channel-first image data with at least 3 dimensions, got shape {values.shape}.')

        channel_axis = values.ndim - 3

        if values.shape[channel_axis] != self.channels:
            raise ValueError(
                f'Input normalisation requires exactly {self.channels} channels, got shape {values.shape}.'
            )

        values = np.moveaxis(values, channel_axis, 0).reshape(self.channels, -1)
        batch_count = int(values.shape[1])

        if batch_count == 0:
            raise ValueError('Cannot calculate normalisation statistics from empty image data.')

        batch_mean = values.mean(axis=1)
        centred = values - batch_mean[:, np.newaxis]
        batch_m2 = np.sum(centred * centred, axis=1)
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * (batch_count / total_count)
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total_count
        self.count = total_count

    def finalise(self):
        """Return per-channel population mean and standard deviation."""
        if self.count == 0:
            raise ValueError('Cannot calculate normalisation statistics from an empty training set.')

        standard_deviation = np.sqrt(self.m2 / self.count)

        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(standard_deviation)):
            raise ValueError('Calculated normalisation statistics contain NaN or infinite values.')

        zero_variance_channels = np.flatnonzero(standard_deviation <= 0).tolist()

        if zero_variance_channels:
            raise ValueError(
                f'Cannot normalise constant training channel(s) {zero_variance_channels}; standard deviation must be positive.'
            )

        return self.mean.astype(float).tolist(), standard_deviation.astype(float).tolist()


def validate_normalisation_constants(mean, standard_deviation, channels=EXPECTED_NORMALISATION_CHANNELS):
    """Validate and return immutable per-channel normalisation constants."""
    mean = tuple(float(value) for value in mean)
    standard_deviation = tuple(float(value) for value in standard_deviation)

    if len(mean) != int(channels) or len(standard_deviation) != int(channels):
        raise ValueError(
            f'Normalisation mean and standard deviation must each contain exactly {channels} values; '
            f'got {len(mean)} and {len(standard_deviation)}.'
        )

    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(standard_deviation)):
        raise ValueError('Normalisation constants must be finite.')

    if any(value <= 0 for value in standard_deviation):
        raise ValueError('Every normalisation standard deviation must be greater than zero.')

    return mean, standard_deviation


def normalise_tensor(tensor, mean, standard_deviation):
    """Normalise a tensor whose third-last dimension is the channel dimension."""
    mean, standard_deviation = validate_normalisation_constants(mean, standard_deviation)

    if tensor.ndim < 3 or tensor.shape[-3] != EXPECTED_NORMALISATION_CHANNELS:
        raise ValueError(
            f'Input normalisation requires tensors with exactly {EXPECTED_NORMALISATION_CHANNELS} channels, '
            f'got shape {tuple(tensor.shape)}.'
        )

    shape = [1] * tensor.ndim
    shape[-3] = EXPECTED_NORMALISATION_CHANNELS
    mean_tensor = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).reshape(shape)
    std_tensor = torch.as_tensor(standard_deviation, dtype=tensor.dtype, device=tensor.device).reshape(shape)
    return (tensor - mean_tensor) / std_tensor
