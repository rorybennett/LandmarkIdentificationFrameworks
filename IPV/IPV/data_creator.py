import csv
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import distance as dist
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_float32, img_as_ubyte

from .gpu_utils import createPatch, getAngle, get_label

TRAIN_LIST_PATTERN = re.compile(r'^train_f(\d+)\.txt$')

GRID_PHASES = {'Val'}
SAMPLED_PHASES = {'Train'}

MIN_POINTS_PER_IMAGE = 1
MAX_POINTS_PER_IMAGE = 30
TASKS_PER_POINT = 2


@dataclass
class PatchJob:
    sample_index: int
    sample_name: str
    image_path: Path
    points: list
    phase: str
    grid_spacing: int
    sub_patch_scales: list
    distance_intervals: list
    angle_intervals: list
    patches_per_training_sample: int
    sampling_variances: tuple
    patch_save_path: Path
    image_save_path: Path
    part_csv_path: Path
    seed: int


class ProgressBar:
    """Simple terminal progress bar for phase-level patch creation."""

    def __init__(self, total, label, width=40):
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self.current = 0
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.render()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.write('\n')
        sys.stdout.flush()

    def update(self, increment=1):
        """Advance the progress bar."""
        self.current += increment
        self.render()

    def render(self):
        """Render the progress bar on one terminal line."""
        fraction = min(self.current / self.total, 1.0)
        filled = int(self.width * fraction)
        bar = '#' * filled + '-' * (self.width - filled)

        elapsed = time.time() - self.start_time if self.start_time else 0
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0

        message = (
            f'\r\t{self.label}: '
            f'[{bar}] '
            f'{self.current}/{self.total} '
            f'({fraction * 100:6.2f}%) '
            f'elapsed={self.format_seconds(elapsed)} '
            f'eta={self.format_seconds(remaining)}'
        )

        sys.stdout.write(message)
        sys.stdout.flush()

    @staticmethod
    def format_seconds(seconds):
        """Format seconds as HH:MM:SS."""
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def natural_key(value):
    """Sort strings naturally, so A2 comes before A10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(value))]


def discover_fold_numbers(fold_lists_path):
    """Return contiguous fold numbers discovered from train_fN.txt files."""
    fold_lists_path = Path(fold_lists_path)

    if not fold_lists_path.is_dir():
        raise ValueError(f'fold_lists_path does not exist or is not a directory: {fold_lists_path}')

    fold_numbers = []

    for file_path in fold_lists_path.iterdir():
        if not file_path.is_file():
            continue

        match = TRAIN_LIST_PATTERN.fullmatch(file_path.name)

        if match:
            fold_numbers.append(int(match.group(1)))

    fold_numbers = sorted(set(fold_numbers))

    if not fold_numbers:
        raise ValueError(f'No train_fN.txt files found in {fold_lists_path}')

    expected_fold_numbers = list(range(1, fold_numbers[-1] + 1))

    if fold_numbers != expected_fold_numbers:
        raise ValueError(f'Fold files must be contiguous from train_f1.txt. Found {fold_numbers}, expected {expected_fold_numbers}')

    val_file = fold_lists_path / 'val.txt'

    if not val_file.is_file():
        raise ValueError(f'Validation list file not found: {val_file}')

    return fold_numbers


class PatchCreator:
    """Create multi-scale patches centred on a single image location."""

    def __init__(self, image, sub_patch_scales):
        self.image = image
        self.sub_patch_scales = sub_patch_scales
        self.patch_size = sub_patch_scales[0]

    def create(self, x, y):
        """Create and resize all sub-patches for one centre point."""
        patches = []

        for scale in self.sub_patch_scales:
            patch = createPatch(self.image, x, y, scale)
            patch = resize_patch(patch, output_size=self.patch_size)
            patches.append(patch)

        return patches


def resize_patch(patch, output_size):
    """Resize a patch while preserving its original channel count."""
    if patch.ndim == 2:
        output_shape = (output_size, output_size)
    elif patch.ndim == 3:
        output_shape = (output_size, output_size, patch.shape[2])
    else:
        raise ValueError(f'Patch must be 2D or 3D, got shape {patch.shape}.')

    return resize(patch, output_shape, preserve_range=True, anti_aliasing=True)


def get_image_channel_count(image, image_path=None):
    """Return the channel count for a greyscale, RGB, or RGBA image array."""
    path_text = f' {image_path}' if image_path is not None else ''

    if image.ndim == 2:
        return 1

    if image.ndim == 3 and image.shape[2] in (1, 3, 4):
        return int(image.shape[2])

    raise ValueError(f'Image{path_text} has unsupported shape {image.shape}. Expected greyscale, RGB, or RGBA.')


def infer_image_channel_count(image_path):
    """Read one image and return its input channel count."""
    image = io.imread(image_path)

    return get_image_channel_count(image=image, image_path=image_path)


def load_patch_source_image(image_path):
    """Load a source image for patch extraction without changing its channel count."""
    image = io.imread(image_path)
    get_image_channel_count(image=image, image_path=image_path)

    return img_as_float32(image)


def load_display_image(image_path):
    """Load an image suitable for drawing and saving."""
    image = io.imread(image_path)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        image = img_as_ubyte(image)

    return np.ascontiguousarray(image)


def prepare_patch_for_saving(patch):
    """Convert a generated patch into a PNG-compatible array."""
    if patch.ndim == 3 and patch.shape[2] == 1:
        patch = patch[:, :, 0]

    if patch.ndim == 3 and patch.shape[2] not in (3, 4):
        raise ValueError(f'Cannot save patch with shape {patch.shape} as PNG. Use greyscale, RGB, or RGBA images.')

    return img_as_ubyte(patch)


def create_labels_for_point(points, x, y, distance_intervals, angle_intervals, sample_name=None):
    """Create distance and angle labels for one patch centre."""
    labels = []

    for point in points:
        pixel_distance = dist.euclidean(point, (x, y))
        angle = getAngle(point, (x, y))

        distance_class = get_checked_label(value=pixel_distance, intervals=distance_intervals, label_name='distance', sample_name=sample_name, point=point, x=x, y=y)

        angle_class = get_checked_label(value=angle, intervals=angle_intervals, label_name='angle', sample_name=sample_name, point=point, x=x, y=y)

        labels.extend([distance_class, angle_class])

    return labels


def get_checked_label(value, intervals, label_name, sample_name=None, point=None, x=None, y=None):
    """Return a class label and raise a clear error if the value is outside all intervals."""
    label = get_label(value, intervals)

    if label < 0:
        context = {
            'label_name': label_name,
            'value': float(value),
            'intervals': intervals,
            'sample_name': sample_name,
            'point': point,
            'patch_centre': (x, y)
        }

        raise ValueError(
            f'Invalid {label_name} label: value does not fall inside any configured interval. '
            f'Context: {context}'
        )

    return label


def create_grid_centres(image_shape, step):
    """Create grid-based patch centres."""
    height, width = image_shape[:2]

    for x in range(0, width, step):
        for y in range(0, height, step):
            yield int(x), int(y)


def wrap_point_to_image(x, y, width, height):
    """Wrap sampled coordinates back inside the image extent."""
    return int(round(x)) % width, int(round(y)) % height


def create_training_centres(points, image_shape, patches_per_training_sample, sampling_variances, rng):
    """Create unique random training patch centres around landmark locations."""
    height, width = image_shape[:2]
    groups = [(point, variance) for point in points for variance in sampling_variances]

    base_count = patches_per_training_sample // len(groups)
    remainder = patches_per_training_sample % len(groups)
    seen_centres = set()

    for group_index, (point, variance) in enumerate(groups):
        group_count = base_count + int(group_index < remainder)
        covariance = [[variance, 0], [0, variance]]
        created_count = 0
        attempt_count = 0
        max_attempts = max(group_count * 100, 1000)

        while created_count < group_count and attempt_count < max_attempts:
            attempt_count += 1
            x, y = rng.multivariate_normal(point, covariance).T
            centre = wrap_point_to_image(x=x, y=y, width=width, height=height)

            if centre in seen_centres:
                continue

            seen_centres.add(centre)
            created_count += 1
            yield centre

        if created_count < group_count:
            raise RuntimeError(f'Only created {created_count}/{group_count} unique centres for point {point} with variance {variance}.')


def create_patch_job(job):
    """Create all patch images and CSV rows for one sample."""
    rng = np.random.default_rng(job.seed)

    image = load_patch_source_image(job.image_path)
    display_image = load_display_image(job.image_path)
    patch_creator = PatchCreator(image, sub_patch_scales=job.sub_patch_scales)

    if job.phase in GRID_PHASES:
        centres = create_grid_centres(image_shape=image.shape, step=job.grid_spacing)
    elif job.phase in SAMPLED_PHASES:
        centres = create_training_centres(points=job.points, image_shape=image.shape, patches_per_training_sample=job.patches_per_training_sample,
                                          sampling_variances=job.sampling_variances, rng=rng)
    else:
        raise ValueError(f'Unknown phase for centre creation: {job.phase}')

    with open(job.part_csv_path, 'w', newline='', encoding='utf-8') as part_csv:
        csv_writer = csv.writer(part_csv)

        for x, y in centres:
            labels = create_labels_for_point(points=job.points, x=x, y=y, distance_intervals=job.distance_intervals, angle_intervals=job.angle_intervals,
                                             sample_name=job.sample_name)
            patches = patch_creator.create(x, y)
            patch_id = f'{job.sample_index}_{x}_{y}'

            for patch_index, patch in enumerate(patches, start=1):
                patch_path = job.patch_save_path / f'{patch_id}_{patch_index}.png'
                patch_image = prepare_patch_for_saving(patch)

                io.imsave(patch_path, patch_image, check_contrast=False)
                csv_writer.writerow([patch_id, patch_path.as_posix(), job.sample_name, x, y, *labels])

            cv2.circle(display_image, (x, y), 1, (255, 0, 0), 1)

    for point in job.points:
        cv2.drawMarker(display_image, point, (255, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2, line_type=cv2.LINE_AA)

    io.imsave(job.image_save_path / f'{job.sample_name}.png', display_image, check_contrast=False)

    return job.sample_index, job.part_csv_path


class DataCreator:
    """Create train and validation patch data for one landmark task."""

    def __init__(self,
                 distance_intervals,
                 angle_intervals,
                 subpatch_scales,
                 task_name=None,
                 data_save_path=None,
                 num_of_points=None,
                 patches_per_training_sample=None,
                 fold_lists_path=None,
                 mark_list_path=None,
                 image_data_path=None,
                 sampling_variances=(500, 10000),
                 num_workers=1,
                 random_seed=42,
                 keep_part_csvs=False):

        self.task_name = self.resolve_task_name(task_name=task_name)
        self.num_of_pts = self.validate_num_points(num_of_points)
        self.sampling_variances = self.validate_sampling_variances(sampling_variances)
        self.patches_per_training_sample = self.validate_patches_per_training_sample(patches_per_training_sample, self.num_of_pts, self.sampling_variances)
        self.validate_required_paths(data_save_path=data_save_path, fold_lists_path=fold_lists_path, mark_list_path=mark_list_path, image_data_path=image_data_path)
        self.validate_subpatch_scales(subpatch_scales)

        self.distance_intervals = distance_intervals
        self.angle_intervals = angle_intervals
        self.fold_numbers = discover_fold_numbers(fold_lists_path)
        self.folds = len(self.fold_numbers)
        self.sub_patch_scales = subpatch_scales
        self.data_save_path = Path(data_save_path)
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.keep_part_csvs = keep_part_csvs

        self.patch_size = subpatch_scales[0]
        self.fold_lists_path = Path(fold_lists_path)
        self.mark_list_path = Path(mark_list_path)
        self.image_data_path = Path(image_data_path)

        self.current_fold = None
        self.input_channels = None
        self.fold_list = []
        self.points_dict = {}
        self.paths_dict = {}

    @staticmethod
    def resolve_task_name(task_name=None):
        """Return the generic task name."""
        resolved_name = task_name

        if resolved_name is None:
            raise ValueError('task_name must be provided.')

        resolved_name = str(resolved_name).strip()

        if not resolved_name:
            raise ValueError('task_name cannot be empty.')

        return resolved_name

    @staticmethod
    def validate_num_points(num_of_points):
        """Validate the number of landmark points per image."""
        if num_of_points is None:
            raise ValueError('num_of_points must be provided.')

        num_of_points = int(num_of_points)

        if num_of_points < MIN_POINTS_PER_IMAGE or num_of_points > MAX_POINTS_PER_IMAGE:
            raise ValueError(f'num_of_points must be between {MIN_POINTS_PER_IMAGE} and {MAX_POINTS_PER_IMAGE}. Got: {num_of_points}')

        return num_of_points

    @staticmethod
    def validate_sampling_variances(sampling_variances):
        """Validate sampling variances and return them as a tuple."""
        if sampling_variances is None:
            raise ValueError('sampling_variances must be provided.')

        sampling_variances = tuple(sampling_variances)

        if not sampling_variances:
            raise ValueError('sampling_variances must contain at least one value.')

        for index, value in enumerate(sampling_variances):
            if not isinstance(value, (int, float)):
                raise ValueError(f'sampling_variances[{index}] must be numeric. Got: {value}')

            if value <= 0:
                raise ValueError(f'sampling_variances[{index}] must be greater than 0. Got: {value}')

        return sampling_variances

    @staticmethod
    def validate_patches_per_training_sample(patches_per_training_sample, num_of_points, sampling_variances):
        """Validate enough sampled patches exist for every point and variance group."""
        if patches_per_training_sample is None:
            raise ValueError('patches_per_training_sample must be provided.')

        patches_per_training_sample = int(patches_per_training_sample)

        if patches_per_training_sample < 1:
            raise ValueError('patches_per_training_sample must be at least 1.')

        minimum_patches = num_of_points * len(sampling_variances)

        if patches_per_training_sample < minimum_patches:
            raise ValueError(
                f'patches_per_training_sample must be at least {minimum_patches} '
                f'for {num_of_points} points and {len(sampling_variances)} sampling variances.'
            )

        return patches_per_training_sample

    @staticmethod
    def validate_required_paths(data_save_path, fold_lists_path, mark_list_path, image_data_path):
        """Check that required path arguments were supplied."""
        required_paths = {
            'data_save_path': data_save_path,
            'fold_lists_path': fold_lists_path,
            'mark_list_path': mark_list_path,
            'image_data_path': image_data_path
        }

        missing_paths = [name for name, value in required_paths.items() if value is None]

        if missing_paths:
            raise ValueError(f'Missing required path arguments: {missing_paths}')

    @staticmethod
    def validate_subpatch_scales(subpatch_scales):
        """Validate sub-patch scales before patch creation."""
        if subpatch_scales is None:
            raise ValueError('subpatch_scales must be provided.')

        if len(subpatch_scales) != 4:
            raise ValueError(f'Quadruplet data creation requires exactly 4 sub-patch scales. Got: {subpatch_scales}')

        if any(scale <= 0 for scale in subpatch_scales):
            raise ValueError(f'subpatch_scales must contain positive integers. Got: {subpatch_scales}')

        if sorted(subpatch_scales) != list(subpatch_scales):
            raise ValueError(f'subpatch_scales should be in ascending order because the first value is the output patch size. Got: {subpatch_scales}')

    def create(self, grid_spacing, current_fold):
        """Create training and validation data for the requested fold."""
        self.current_fold = current_fold

        if not self.fold_list:
            self.read_fold_lists()

        self.report_current_fold_input_channels()

        self.create_data(grid_spacing=grid_spacing, phase='Train')
        self.create_data(grid_spacing=grid_spacing, phase='Val')

    def read_fold_lists(self):
        """Read train and validation sample names for all folds."""
        self.fold_list = []

        for fold_index in self.fold_numbers:
            train_list = self.read_name_list(self.fold_lists_path / f'train_f{fold_index}.txt')
            val_list = self.read_name_list(self.fold_lists_path / 'val.txt')

            self.validate_fold_split(fold_index, train_list, val_list)
            self.fold_list.append([train_list, val_list])

        print(f'\tAll {self.folds} folds read...', flush=True)

    def read_points(self, phase):
        """Read image paths and landmark coordinates for the current fold and phase."""
        names_list = self.get_phase_names(phase)
        mark_records = self.read_mark_list()

        self.points_dict = {}
        self.paths_dict = {}

        for sample_name in names_list:
            if sample_name not in mark_records:
                raise KeyError(f'{sample_name} was found in the fold list but not in {self.mark_list_path}.')

            image_name, points = mark_records[sample_name]

            self.validate_mark_points(sample_name=sample_name, points=points)

            image_path = self.image_data_path / image_name

            if not image_path.is_file():
                raise FileNotFoundError(f'Image for {sample_name} was not found: {image_path}')

            self.paths_dict[sample_name] = image_path
            self.points_dict[sample_name] = points[:self.num_of_pts]

    def report_current_fold_input_channels(self):
        """Report and validate the input channel count for the current fold."""
        sample_names = self.get_current_fold_sample_names()
        mark_records = self.read_mark_list()
        channel_counts = {}

        for sample_name in sample_names:
            if sample_name not in mark_records:
                raise KeyError(f'{sample_name} was found in the fold list but not in {self.mark_list_path}.')

            image_name, _ = mark_records[sample_name]
            image_path = self.image_data_path / image_name

            if not image_path.is_file():
                raise FileNotFoundError(f'Image for {sample_name} was not found: {image_path}')

            input_channels = infer_image_channel_count(image_path)
            channel_counts.setdefault(input_channels, []).append(sample_name)

        if len(channel_counts) != 1:
            details = ', '.join(f'{channels} channels: {len(names)} images' for channels, names in sorted(channel_counts.items()))
            raise ValueError(f'Mixed input channel counts found for fold {self.current_fold}: {details}. Use one consistent channel count per dataset.')

        self.input_channels = next(iter(channel_counts))
        image_text = 'image' if len(sample_names) == 1 else 'images'
        print(f'	Input channels detected for fold {self.current_fold} {self.task_name}: {self.input_channels} ({len(sample_names)} {image_text}).', flush=True)

    def get_current_fold_sample_names(self):
        """Return all sample names used by the current fold."""
        sample_names = []

        for phase in ('Train', 'Val'):
            sample_names.extend(self.get_phase_names(phase))

        return sorted(set(sample_names), key=natural_key)

    def validate_mark_points(self, sample_name, points):
        """Validate landmark count for one mark-list row."""
        if len(points) < self.num_of_pts:
            raise ValueError(f'{sample_name} has {len(points)} points but {self.num_of_pts} are required.')

        if len(points) > MAX_POINTS_PER_IMAGE:
            raise ValueError(f'{sample_name} has {len(points)} points; the maximum supported number is {MAX_POINTS_PER_IMAGE}.')

    def create_data(self, grid_spacing, phase):
        """Create and save patches and labels for one phase."""
        self.read_points(phase)

        patch_save_path = self.data_save_path / f'{phase}_Patches_F{self.current_fold}'
        image_save_path = self.data_save_path / f'{phase}_Images_F{self.current_fold}'
        part_csv_dir = self.data_save_path / f'{phase}_csv_parts_F{self.current_fold}'
        csv_path = self.data_save_path / f'{phase}_f{self.current_fold}.csv'

        for output_dir in (patch_save_path, image_save_path):
            if output_dir.exists():
                shutil.rmtree(output_dir)

            output_dir.mkdir(exist_ok=True, parents=True)

        if part_csv_dir.exists():
            shutil.rmtree(part_csv_dir)

        part_csv_dir.mkdir(exist_ok=True, parents=True)

        jobs = self.create_patch_jobs(phase=phase, grid_spacing=grid_spacing, patch_save_path=patch_save_path, image_save_path=image_save_path, part_csv_dir=part_csv_dir)
        progress_label = f'{phase} phase patch creation'

        if self.num_workers <= 1:
            completed_parts = self.run_patch_jobs_serial(jobs, progress_label)
        else:
            completed_parts = self.run_patch_jobs_parallel(jobs, progress_label)

        self.merge_part_csvs(csv_path=csv_path, completed_parts=completed_parts)
        self.validate_patch_csv(csv_path=csv_path)

        if not self.keep_part_csvs:
            shutil.rmtree(part_csv_dir)

    def validate_patch_csv(self, csv_path):
        """Validate that each patch group contains one row per sub-patch scale and valid labels."""
        expected_rows = len(self.sub_patch_scales)
        expected_label_count = self.num_of_pts * TASKS_PER_POINT
        expected_columns = 5 + expected_label_count
        group_counts = {}

        with open(csv_path, 'r', newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:
                if not row:
                    continue

                if len(row) != expected_columns:
                    raise ValueError(f'{csv_path} has a row with {len(row)} columns, expected {expected_columns}. Bad row starts with: {row[:8]}')

                patch_id = row[0]
                group_counts[patch_id] = group_counts.get(patch_id, 0) + 1

                labels = [int(value) for value in row[5:]]

                for point_index in range(self.num_of_pts):
                    distance_label = labels[point_index * 2]
                    angle_label = labels[point_index * 2 + 1]

                    if not 0 <= distance_label < len(self.distance_intervals):
                        raise ValueError(f'{csv_path} contains invalid distance label {distance_label} for patch_id={patch_id}, point_index={point_index}.')

                    if not 0 <= angle_label < len(self.angle_intervals):
                        raise ValueError(f'{csv_path} contains invalid angle label {angle_label} for patch_id={patch_id}, point_index={point_index}.')

        bad_groups = [(patch_id, count)
                      for patch_id, count in group_counts.items()
                      if count != expected_rows]

        if bad_groups:
            examples = ', '.join([f'{patch_id}: {count}' for patch_id, count in bad_groups[:10]])
            raise ValueError(f'{csv_path} has invalid patch groups. Expected {expected_rows} rows per group. Examples: {examples}')

        print(f'\tValidated {csv_path.name}: {len(group_counts)} patch groups.', flush=True)

    def create_patch_jobs(self, phase, grid_spacing, patch_save_path, image_save_path, part_csv_dir):
        """Create one worker job per sample in natural alphabetical order."""
        jobs = []
        sample_names = sorted(self.points_dict.keys(), key=natural_key)

        for sample_index, sample_name in enumerate(sample_names, start=1):
            seed = self.random_seed + self.current_fold * 100000 + sample_index if self.random_seed is not None else None

            jobs.append(PatchJob(
                sample_index=sample_index,
                sample_name=sample_name,
                image_path=self.paths_dict[sample_name],
                points=self.points_dict[sample_name],
                phase=phase,
                grid_spacing=grid_spacing,
                sub_patch_scales=self.sub_patch_scales,
                distance_intervals=self.distance_intervals,
                angle_intervals=self.angle_intervals,
                patches_per_training_sample=self.patches_per_training_sample,
                sampling_variances=self.sampling_variances,
                patch_save_path=patch_save_path,
                image_save_path=image_save_path,
                part_csv_path=part_csv_dir / f'{sample_index}_{sample_name}.csv',
                seed=seed
            ))

        return jobs

    def run_patch_jobs_serial(self, jobs, progress_label):
        """Run sample-level patch creation in the current process."""
        completed_parts = []

        with ProgressBar(total=len(jobs), label=progress_label) as progress:
            for job in jobs:
                completed_parts.append(create_patch_job(job))
                progress.update()

        return completed_parts

    def run_patch_jobs_parallel(self, jobs, progress_label):
        """Run sample-level patch creation in separate processes."""
        completed_parts = []

        with ProgressBar(total=len(jobs), label=progress_label) as progress:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(create_patch_job, job): job for job in jobs}

                for future in as_completed(futures):
                    job = futures[future]

                    try:
                        completed_parts.append(future.result())
                        progress.update()
                    except Exception as error:
                        raise RuntimeError(f'Patch creation failed for {job.sample_name}') from error

        return completed_parts

    @staticmethod
    def merge_part_csvs(csv_path, completed_parts):
        """Merge per-sample temporary CSVs into the phase CSV."""
        completed_parts = sorted(completed_parts, key=lambda item: item[0])

        with open(csv_path, 'w', newline='', encoding='utf-8') as output_csv:
            writer = csv.writer(output_csv)

            for _, part_csv_path in completed_parts:
                with open(part_csv_path, 'r', newline='', encoding='utf-8') as part_csv:
                    reader = csv.reader(part_csv)
                    writer.writerows(reader)

    def get_phase_names(self, phase):
        """Return the sample names for a phase."""
        phase_indexes = {
            'Train': 0,
            'Val': 1
        }

        if phase not in phase_indexes:
            raise ValueError(f'Unknown phase: {phase}')

        return self.fold_list[self.current_fold - 1][phase_indexes[phase]]

    def read_mark_list(self):
        """Read landmark markings from the mark list file."""
        records = {}

        with open(self.mark_list_path, 'r', encoding='utf-8') as mark_file:
            for line in mark_file:
                line = line.strip()

                if not line:
                    continue

                image_name = line.split()[0]
                sample_name = Path(image_name).stem
                points = [(int(x), int(y)) for x, y in re.findall(r'\((-?\d+),\s*(-?\d+)\)', line)]

                records[sample_name] = (image_name, points)

        return records

    @staticmethod
    def read_name_list(path):
        """Read a text file containing one sample name per line in natural alphabetical order."""
        with open(path, 'r', encoding='utf-8') as name_file:
            names = [line.strip() for line in name_file if line.strip()]

        return sorted(names, key=natural_key)

    @staticmethod
    def validate_fold_split(fold_index, train_list, val_list):
        """Check that train and validation splits do not overlap."""
        train_set = set(train_list)
        val_set = set(val_list)

        overlap = train_set & val_set

        if overlap:
            raise ValueError(f'Fold {fold_index} has overlapping samples: {sorted(overlap)}')
