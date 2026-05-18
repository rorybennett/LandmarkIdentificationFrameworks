"""
Dataset/task configuration.

Edit this file for values that define the dataset and prediction task.

Run-specific values such as fold, batch size, workers, learning rate, and paths
should come from terminal arguments.
"""

sub_patch_scales = [64, 128, 256, 512]

sampling_variances = (500, 10000)

distance_intervals = [
    (0, 15),
    (15, 25),
    (25, 40),
    (40, 60),
    (60, 85),
    (85, 115),
    (115, 150),
    (150, 190),
    (190, 235),
    (235, 285),
    (285, 340),
    (340, 400),
    (400, 465),
    (465, 535),
    (535, 610),
    (610, 690),
    (690, 775),
    (775, 865),
    (865, 2000),
]

angle_intervals = [
    (0, 45),
    (45, 90),
    (90, 135),
    (135, 180),
    (180, 225),
    (225, 270),
    (270, 315),
    (315, 360),
]
