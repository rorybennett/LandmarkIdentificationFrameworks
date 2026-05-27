"""
Reusable IPV landmark inference, validation inference, and visualisation utilities.
"""
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_float32

from .patch_utils import create_patch
from .progress_bar import ProgressBar

POINT_PATTERN = re.compile(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)')
TASKS_PER_POINT = 2
PREDICTED_POINT_COLOUR = (0, 0, 255)
GROUND_TRUTH_POINT_COLOUR = (0, 255, 0)
POINT_MARKER_SIZE = 16
POINT_MARKER_THICKNESS = 2
HEATMAP_IMAGE_WEIGHT = 0.55
HEATMAP_COLOUR_WEIGHT = 0.65
VOTE_MAP_IMAGE_WEIGHT = 0.35
VOTE_MAP_COLOUR_WEIGHT = 0.75
SUPPORTED_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


@dataclass
class LandmarkInferenceConfig:
    output_dir: Path
    num_points: int
    sub_patch_scales: list
    distance_intervals: list
    angle_intervals: list
    grid_spacing: int
    input_channels: int
    task_name: str = ''
    fold: int | None = None
    data_save_path: Path | None = None
    mark_list_file: Path | None = None
    image_data_dir: Path | None = None
    batch_size: int = 2048
    smoothing_sigma: float = 7.0
    use_probability_weights: bool = True
    save_raw_vote_maps: bool = False
    checkpoint_path: Path | None = None
    checkpoint_type: str | None = None
    run_label: str = 'inference'


@dataclass
class LandmarkImageRecord:
    sample_name: str
    image_path: Path
    ground_truth_points: list | None = None


@dataclass
class LoadedInferenceCheckpoint:
    model: torch.nn.Module
    checkpoint: dict
    metadata: dict


def truncate_text(value, max_length):
    """Return text shortened for use inside terminal progress lines."""
    value = str(value)

    if len(value) <= max_length:
        return value

    return f'{value[:max_length - 3]}...'


class LandmarkImageInferer:
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = self.normalise_config(config)
        self.device = resolve_device(device) if device is not None else next(model.parameters()).device
        self.output_dirs = self.get_output_dirs()
        self.model.to(self.device)
        self.model.eval()
        self.prepare_output_dirs()

    @staticmethod
    def normalise_config(config):
        """Return config with path-like fields and numeric fields normalised."""
        config.output_dir = Path(config.output_dir)
        config.data_save_path = None if config.data_save_path is None else Path(config.data_save_path)
        config.mark_list_file = None if config.mark_list_file is None else Path(config.mark_list_file)
        config.image_data_dir = None if config.image_data_dir is None else Path(config.image_data_dir)
        config.checkpoint_path = None if config.checkpoint_path is None else Path(config.checkpoint_path)
        config.sub_patch_scales = [int(scale) for scale in config.sub_patch_scales]
        config.distance_intervals = [[float(lower), float(upper)] for lower, upper in config.distance_intervals]
        config.angle_intervals = [[float(lower), float(upper)] for lower, upper in config.angle_intervals]
        config.run_label = safe_file_stem(config.run_label)
        return config

    def prepare_output_dirs(self):
        """Create inference-output directories."""
        self.config.output_dir.mkdir(exist_ok=True, parents=True)

        for name, output_dir in self.output_dirs.items():
            if name == 'raw_vote_maps' and not self.config.save_raw_vote_maps:
                continue

            output_dir.mkdir(exist_ok=True, parents=True)

    def get_output_dirs(self):
        """Return per-output subdirectories."""
        return {
            'heatmap_overlays': self.config.output_dir / 'heatmap_overlays',
            'point_overlays': self.config.output_dir / 'point_overlays',
            'vote_maps': self.config.output_dir / 'vote_maps',
            'raw_vote_maps': self.config.output_dir / 'raw_vote_maps',
            'logs': self.config.output_dir / 'logs'
        }

    def infer_records(self, records):
        """Run inference for all image records and save combined summaries."""
        records = list(records)
        results = []
        start_time = time.perf_counter()
        progress_label = f'{self.config.run_label} inference'

        with ProgressBar(total=len(records), label=progress_label) as progress_bar:
            for record in records:
                progress_bar.set_status(record.sample_name)
                results.append(self.infer_record(record))
                progress_bar.update()

        self.save_combined_summaries(results)
        self.save_run_metadata(records=records, results=results, total_seconds=time.perf_counter() - start_time)
        return results

    def infer_record(self, record):
        """Run inference for one image and save per-image outputs."""
        record = LandmarkImageRecord(sample_name=record.sample_name, image_path=Path(record.image_path), ground_truth_points=record.ground_truth_points)
        image = load_input_image(record.image_path, input_channels=self.config.input_channels)
        display_image = load_display_image(record.image_path)
        centres = list(create_centres(image_shape=image.shape, step_size=self.config.grid_spacing))

        if not centres:
            raise ValueError(f'No inference centres were created for {record.image_path}. Check image size and grid spacing.')

        vote_inputs = self.create_empty_vote_inputs()

        with torch.inference_mode():
            for centre_batch in chunk_items(centres, self.config.batch_size):
                batch_tensor = create_batch_tensor(image=image, centres=centre_batch, sub_patch_scales=self.config.sub_patch_scales,
                                                   patch_size=self.config.sub_patch_scales[0], input_channels=self.config.input_channels)
                batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                outputs = self.model(batch_tensor)
                self.collect_vote_inputs(vote_inputs=vote_inputs, outputs=outputs)
                del batch_tensor, outputs

        vote_inputs = self.finalise_vote_inputs(vote_inputs)
        vote_maps = accumulate_vote_maps(centres=centres, vote_inputs=vote_inputs, image_shape=image.shape[:2], distance_intervals=self.config.distance_intervals,
                                         angle_intervals=self.config.angle_intervals, num_points=self.config.num_points)
        detected_points, peak_values, smoothed_vote_maps = detect_points(vote_maps=vote_maps, smoothing_sigma=self.config.smoothing_sigma)
        result = build_result(record=record, detected_points=detected_points, ground_truth_points=record.ground_truth_points, peak_values=peak_values,
                              num_centres=len(centres), grid_spacing=self.config.grid_spacing, checkpoint_type=self.config.checkpoint_type)
        output_stem = safe_file_stem(record.sample_name)
        self.save_visual_outputs(output_stem=output_stem, display_image=display_image, detected_points=detected_points, ground_truth_points=record.ground_truth_points,
                                 smoothed_vote_maps=smoothed_vote_maps)

        if self.config.save_raw_vote_maps:
            np.save(self.output_dirs['raw_vote_maps'] / f'{output_stem}_{self.config.run_label}_raw_vote_maps.npy', vote_maps)

        return result

    def create_empty_vote_inputs(self):
        """Create model-output containers for endpoint voting."""
        return [{'distance_classes': [], 'angle_classes': [], 'scores': []} for _ in range(self.config.num_points)]

    def collect_vote_inputs(self, vote_inputs, outputs):
        """Collect top-1 distance and angle predictions from one model batch."""
        expected_outputs = self.config.num_points * TASKS_PER_POINT

        if len(outputs) != expected_outputs:
            raise ValueError(f'Model produced {len(outputs)} output heads, expected {expected_outputs}.')

        probabilities = [torch.softmax(output, dim=1).detach().cpu().numpy() for output in outputs]
        predictions = [np.argmax(probability, axis=1).astype(np.int16) for probability in probabilities]
        confidence = [np.max(probability, axis=1).astype(np.float32) for probability in probabilities]

        for point_index in range(self.config.num_points):
            distance_head_index = point_index * TASKS_PER_POINT
            angle_head_index = distance_head_index + 1
            scores = confidence[distance_head_index] * confidence[angle_head_index] if self.config.use_probability_weights else np.ones_like(
                confidence[distance_head_index], dtype=np.float32)
            vote_inputs[point_index]['distance_classes'].append(predictions[distance_head_index])
            vote_inputs[point_index]['angle_classes'].append(predictions[angle_head_index])
            vote_inputs[point_index]['scores'].append(scores.astype(np.float32))

    @staticmethod
    def finalise_vote_inputs(vote_inputs):
        """Concatenate per-batch prediction arrays."""
        finalised_inputs = []

        for point_vote_input in vote_inputs:
            finalised_inputs.append({
                'distance_classes': np.concatenate(point_vote_input['distance_classes']),
                'angle_classes': np.concatenate(point_vote_input['angle_classes']),
                'scores': np.concatenate(point_vote_input['scores'])
            })

        return finalised_inputs

    def save_visual_outputs(self, output_stem, display_image, detected_points, ground_truth_points, smoothed_vote_maps):
        """Save heatmap and endpoint overlay images."""
        heatmap_overlay = create_combined_heatmap_overlay(display_image=display_image, smoothed_vote_maps=smoothed_vote_maps, detected_points=detected_points)
        point_overlay = create_point_overlay(display_image=display_image, detected_points=detected_points, ground_truth_points=ground_truth_points)
        cv2.imwrite(str(self.output_dirs['heatmap_overlays'] / f'{output_stem}_{self.config.run_label}_heatmap_overlay.png'), heatmap_overlay)
        cv2.imwrite(str(self.output_dirs['point_overlays'] / f'{output_stem}_{self.config.run_label}_points_overlay.png'), point_overlay)

        for point_index, vote_map in enumerate(smoothed_vote_maps, start=1):
            vote_overlay = create_single_vote_map_overlay(display_image=display_image, vote_map=vote_map)
            cv2.imwrite(str(self.output_dirs['vote_maps'] / f'{output_stem}_{self.config.run_label}_vote_map_p{point_index}.png'), vote_overlay)

    def save_combined_summaries(self, results):
        """Save combined inference summaries across images."""
        endpoint_rows = [row for result in results for row in result['endpoint_rows']]
        summary_rows = [result['summary'] for result in results]
        summary_prefix = build_summary_prefix(self.config.run_label)
        output_path = self.config.output_dir / f'{summary_prefix}_summary.xlsx'

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='image_summary', index=False)
            pd.DataFrame(endpoint_rows).to_excel(writer, sheet_name='endpoints', index=False)

    def save_run_metadata(self, records, results, total_seconds):
        """Save inference run metadata."""
        metadata = {
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'total_seconds': total_seconds,
            'image_count': len(records),
            'result_count': len(results),
            'config': asdict(self.config),
            'records': [{'sample_name': record.sample_name, 'image_path': Path(record.image_path).as_posix()} for record in records]
        }

        summary_prefix = build_summary_prefix(self.config.run_label)

        with open(self.output_dirs['logs'] / f'{summary_prefix}_run_metadata.json', 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4, default=str)


def build_summary_prefix(run_label):
    """Return the base filename used for combined summary outputs."""
    run_label = safe_file_stem(run_label)
    return run_label if run_label == 'inference' else f'{run_label}_inference'


def run_landmark_inference_for_records(model, config, records, device=None):
    """Run landmark inference from an already constructed model and explicit image records."""
    inferer = LandmarkImageInferer(model=model, config=config, device=device)
    return inferer.infer_records(records)


def run_validation_inference_for_trained_model(model, config, device=None):
    """Run validation inference from an already constructed and trained model."""
    config.run_label = 'validation'
    records = build_validation_records(config)
    inferer = LandmarkImageInferer(model=model, config=config, device=device)
    return inferer.infer_records(records)


def build_validation_records(config):
    """Build validation image records from the generated Val CSV and mark-list file."""
    config = LandmarkImageInferer.normalise_config(config)

    if config.data_save_path is None or config.fold is None:
        raise ValueError('Validation inference requires data_save_path and fold.')

    if config.mark_list_file is None or config.image_data_dir is None:
        raise ValueError('Validation inference requires mark_list_file and image_data_dir.')

    validation_csv_path = config.data_save_path / f'Val_f{config.fold}.csv'

    if not validation_csv_path.is_file():
        raise ValueError(f'Validation CSV does not exist: {validation_csv_path}')

    sample_names = read_validation_sample_names(validation_csv_path)
    mark_records = read_mark_list(config.mark_list_file, expected_points=config.num_points)
    records = []

    for sample_name in sample_names:
        if sample_name not in mark_records:
            raise KeyError(f'{sample_name} was found in {validation_csv_path}, but not in {config.mark_list_file}.')

        image_name, points = mark_records[sample_name]
        image_path = config.image_data_dir / image_name

        if not image_path.is_file():
            raise FileNotFoundError(f'Image for validation sample {sample_name} was not found: {image_path}')

        records.append(LandmarkImageRecord(sample_name=sample_name, image_path=image_path, ground_truth_points=points[:config.num_points]))

    return records


def build_image_records(input_path, num_points, mark_list_path=None, recursive=False, supported_suffixes=SUPPORTED_IMAGE_SUFFIXES):
    """Build inference image records from one image or a directory, with optional ground truth."""
    image_paths = find_images(input_path=input_path, recursive=recursive, supported_suffixes=supported_suffixes)
    mark_records = read_mark_list(mark_list_path, expected_points=num_points) if mark_list_path is not None else {}
    records = []

    for image_path in image_paths:
        sample_name = Path(image_path).stem
        ground_truth_points = mark_records.get(sample_name, (None, None))[1]
        records.append(LandmarkImageRecord(sample_name=sample_name, image_path=Path(image_path), ground_truth_points=ground_truth_points))

    return records


def find_images(input_path, recursive=False, supported_suffixes=SUPPORTED_IMAGE_SUFFIXES):
    """Return one image path or all supported image paths in a directory."""
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f'Input path does not exist: {input_path}')

    suffixes = tuple(suffix.lower() for suffix in supported_suffixes)
    iterator = input_path.rglob('*') if recursive else input_path.iterdir()
    return sorted([path for path in iterator if path.is_file() and path.suffix.lower() in suffixes], key=lambda path: path.as_posix().lower())


def read_validation_sample_names(validation_csv_path):
    """Read unique validation sample names from a generated Val CSV."""
    sample_names = []
    seen = set()

    with open(validation_csv_path, 'r', newline='', encoding='utf-8') as validation_csv:
        reader = csv.reader(validation_csv)

        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue

            if len(row) < 3:
                raise ValueError(f'Validation CSV row {row_number} in {validation_csv_path} does not contain a sample_name column.')

            sample_name = row[2]

            if sample_name not in seen:
                sample_names.append(sample_name)
                seen.add(sample_name)

    if not sample_names:
        raise ValueError(f'No validation samples were found in {validation_csv_path}.')

    return sample_names


def read_mark_list(mark_list_path, expected_points):
    """Read mark-list records keyed by sample stem."""
    mark_records = {}

    if mark_list_path is None:
        return mark_records

    with open(mark_list_path, 'r', encoding='utf-8') as mark_file:
        for line_number, line in enumerate(mark_file, start=1):
            line = line.strip()

            if not line:
                continue

            image_name = line.split()[0]
            sample_name = Path(image_name).stem
            points = [(float(x), float(y)) for x, y in POINT_PATTERN.findall(line)]

            if len(points) < int(expected_points):
                raise ValueError(f'Mark-list row {line_number} for {sample_name} has {len(points)} complete point(s), expected at least {expected_points}.')

            if sample_name in mark_records:
                raise ValueError(f'Duplicate sample name in mark list: {sample_name}.')

            mark_records[sample_name] = (image_name, points[:int(expected_points)])

    return mark_records


def load_input_image(image_path, input_channels):
    """Load one source image as float32 HWC and match the model channel count."""
    image = io.imread(image_path)
    image = img_as_float32(image)

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    if image.ndim != 3:
        raise ValueError(f'Unsupported image shape for {image_path}: {image.shape}')

    return np.ascontiguousarray(convert_channels_if_needed(image=image, input_channels=input_channels, image_path=image_path), dtype=np.float32)


def convert_channels_if_needed(image, input_channels, image_path):
    """Convert image channels when the source image differs from the trained model."""
    expected_channels = int(input_channels)
    actual_channels = int(image.shape[2])

    if actual_channels == expected_channels:
        return image

    if expected_channels == 1:
        source = image[:, :, :3] if actual_channels == 4 else image
        return cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis].astype(np.float32)

    if expected_channels == 3 and actual_channels == 1:
        return np.repeat(image, 3, axis=2)

    if expected_channels == 3 and actual_channels == 4:
        return image[:, :, :3]

    if expected_channels == 4 and actual_channels == 1:
        rgb_image = np.repeat(image, 3, axis=2)
        alpha = np.ones((*image.shape[:2], 1), dtype=np.float32)
        return np.concatenate([rgb_image, alpha], axis=2)

    if expected_channels == 4 and actual_channels == 3:
        alpha = np.ones((*image.shape[:2], 1), dtype=np.float32)
        return np.concatenate([image, alpha], axis=2)

    raise ValueError(f'Could not convert {image_path} from {actual_channels} to {expected_channels} channel(s).')


def load_display_image(image_path):
    """Load one source image as BGR uint8 for OpenCV drawing."""
    display_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if display_image is None:
        raise FileNotFoundError(f'Could not read image for display: {image_path}')

    return display_image


def create_batch_tensor(image, centres, sub_patch_scales, patch_size, input_channels):
    """Create one model-input batch tensor from full-image centre locations."""
    batch = np.empty((len(centres), len(sub_patch_scales), int(input_channels), int(patch_size), int(patch_size)), dtype=np.float32)

    for sample_index, (x, y) in enumerate(centres):
        batch[sample_index] = create_sample_tensor(image=image, x=x, y=y, sub_patch_scales=sub_patch_scales, patch_size=patch_size, input_channels=input_channels)

    return torch.from_numpy(batch).float()


def create_sample_tensor(image, x, y, sub_patch_scales, patch_size, input_channels):
    """Create one multi-scale sample tensor for a full-image centre."""
    sample = np.empty((len(sub_patch_scales), int(input_channels), int(patch_size), int(patch_size)), dtype=np.float32)

    for scale_index, scale in enumerate(sub_patch_scales):
        patch = create_patch(image=image, x=x, y=y, patch_size=int(scale))
        patch = resize_patch_for_model(patch=patch, patch_size=int(patch_size)) if int(scale) != int(patch_size) else patch.astype(np.float32)

        if patch.ndim == 2:
            patch = patch[:, :, np.newaxis]

        if patch.shape[2] != int(input_channels):
            raise ValueError(f'Patch generated at ({x}, {y}) has {patch.shape[2]} channels, expected {input_channels}.')

        sample[scale_index] = np.moveaxis(patch, -1, 0)

    return sample


def resize_patch_for_model(patch, patch_size):
    """Resize one patch using the same settings as data creation."""
    if patch.ndim == 2:
        output_shape = (int(patch_size), int(patch_size))
    elif patch.ndim == 3:
        output_shape = (int(patch_size), int(patch_size), patch.shape[2])
    else:
        raise ValueError(f'Patch must be 2D or 3D, got shape {patch.shape}.')

    return resize(patch, output_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)


def create_centres(image_shape, step_size):
    """Create full-image grid centres in the same order as data creation."""
    height, width = image_shape[:2]

    for x in range(0, width, int(step_size)):
        for y in range(0, height, int(step_size)):
            yield int(x), int(y)


def chunk_items(items, chunk_size):
    """Yield fixed-size chunks from a list."""
    for index in range(0, len(items), int(chunk_size)):
        yield items[index:index + int(chunk_size)]


def accumulate_vote_maps(centres, vote_inputs, image_shape, distance_intervals, angle_intervals, num_points):
    """Accumulate endpoint vote maps from distance-angle class predictions."""
    vote_maps = np.zeros((int(num_points), image_shape[0], image_shape[1]), dtype=np.float32)

    for point_index in range(int(num_points)):
        vote_maps[point_index] = accumulate_votes_for_point(centres=centres, distance_classes=vote_inputs[point_index]['distance_classes'],
                                                            angle_classes=vote_inputs[point_index]['angle_classes'], scores=vote_inputs[point_index]['scores'],
                                                            image_shape=image_shape, distance_intervals=distance_intervals, angle_intervals=angle_intervals)

    return vote_maps


def accumulate_votes_for_point(centres, distance_classes, angle_classes, scores, image_shape, distance_intervals, angle_intervals):
    """Accumulate votes for one endpoint."""
    vote_map = np.zeros(image_shape, dtype=np.float32)

    for centre, distance_class, angle_class, score in zip(centres, distance_classes, angle_classes, scores):
        distance_start, distance_end = distance_intervals[int(distance_class)]
        angle_start, angle_end = angle_intervals[int(angle_class)]
        radius = max(1, int(round((float(distance_start) + float(distance_end)) / 2)))
        thickness = max(1, int(round((float(distance_end) - float(distance_start)) / 2)))
        start_angle = (float(angle_start) + 180.0) % 360.0
        end_angle = (float(angle_end) + 180.0) % 360.0
        mask = np.zeros(image_shape, dtype=np.float32)
        draw_arc(mask=mask, centre=(int(centre[0]), int(centre[1])), radius=radius, start_angle=start_angle, end_angle=end_angle, value=1.0, thickness=thickness)
        vote_map += mask * float(score)

    return vote_map


def draw_arc(mask, centre, radius, start_angle, end_angle, value, thickness):
    """Draw one circular voting arc into a single-channel mask."""
    axes = (int(radius), int(radius))

    if start_angle < end_angle:
        cv2.ellipse(mask, centre, axes, 0, float(start_angle), float(end_angle), value, int(thickness))
        return

    cv2.ellipse(mask, centre, axes, 0, float(start_angle), 360.0, value, int(thickness))
    cv2.ellipse(mask, centre, axes, 0, 0.0, float(end_angle), value, int(thickness))


def detect_points(vote_maps, smoothing_sigma):
    """Locate endpoint maxima from smoothed vote maps."""
    detected_points = []
    peak_values = []
    smoothed_vote_maps = []

    for vote_map in vote_maps:
        smoothed_map = cv2.GaussianBlur(vote_map, (0, 0), sigmaX=float(smoothing_sigma), sigmaY=float(smoothing_sigma)) if float(smoothing_sigma) > 0 else vote_map
        _, max_value, _, max_location = cv2.minMaxLoc(smoothed_map)
        detected_points.append((int(max_location[0]), int(max_location[1])))
        peak_values.append(float(max_value))
        smoothed_vote_maps.append(smoothed_map)

    return detected_points, peak_values, np.asarray(smoothed_vote_maps, dtype=np.float32)


def create_combined_heatmap_overlay(display_image, smoothed_vote_maps, detected_points):
    """Create one image containing all endpoint heatmaps over the source image."""
    colour_layer = np.zeros_like(display_image, dtype=np.uint8)

    for vote_map in smoothed_vote_maps:
        normalised_map = normalise_vote_map(vote_map)
        coloured_map = cv2.applyColorMap(normalised_map, cv2.COLORMAP_JET)
        mask = normalised_map > 0
        colour_layer[mask] = np.maximum(colour_layer[mask], coloured_map[mask])

    overlay = cv2.addWeighted(display_image, HEATMAP_IMAGE_WEIGHT, colour_layer, HEATMAP_COLOUR_WEIGHT, 0)
    draw_points(image=overlay, points=detected_points, colour=PREDICTED_POINT_COLOUR, prefix='P')
    return overlay


def create_single_vote_map_overlay(display_image, vote_map):
    """Create a per-endpoint smoothed vote-map overlay."""
    normalised_map = normalise_vote_map(vote_map)
    coloured_map = cv2.applyColorMap(normalised_map, cv2.COLORMAP_JET)
    return cv2.addWeighted(display_image, VOTE_MAP_IMAGE_WEIGHT, coloured_map, VOTE_MAP_COLOUR_WEIGHT, 0)


def create_point_overlay(display_image, detected_points, ground_truth_points):
    """Create one image containing predicted and optional ground-truth endpoints."""
    overlay = display_image.copy()

    if ground_truth_points is not None:
        draw_points(image=overlay, points=ground_truth_points, colour=GROUND_TRUTH_POINT_COLOUR, prefix='G')

    draw_points(image=overlay, points=detected_points, colour=PREDICTED_POINT_COLOUR, prefix='P')
    return overlay


def normalise_vote_map(vote_map):
    """Normalise a vote map to uint8 for visual output."""
    if float(np.max(vote_map)) <= 0:
        return np.zeros(vote_map.shape, dtype=np.uint8)

    normalised_map = cv2.normalize(vote_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalised_map.astype(np.uint8)


def draw_points(image, points, colour, prefix):
    """Draw labelled endpoints onto an image."""
    for point_index, (x, y) in enumerate(points, start=1):
        centre = (int(round(x)), int(round(y)))
        cv2.drawMarker(image, centre, colour, markerType=cv2.MARKER_TILTED_CROSS, markerSize=POINT_MARKER_SIZE, thickness=POINT_MARKER_THICKNESS, line_type=cv2.LINE_AA)
        cv2.putText(image, f'{prefix}{point_index}', (centre[0] + 6, centre[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def build_result(record, detected_points, ground_truth_points, peak_values, num_centres, grid_spacing, checkpoint_type):
    """Build per-image endpoint and summary metrics."""
    endpoint_rows = build_endpoint_rows(record=record, detected_points=detected_points, ground_truth_points=ground_truth_points, peak_values=peak_values)
    point_errors = [row['point_error_px'] for row in endpoint_rows if row['point_error_px'] is not None]
    summary = {
        'sample_name': record.sample_name,
        'image_path': record.image_path.as_posix(),
        'checkpoint_type': checkpoint_type,
        'num_points': len(detected_points),
        'num_centres': int(num_centres),
        'grid_spacing': int(grid_spacing),
        'mean_point_error_px': float(np.mean(point_errors)) if point_errors else None,
        'median_point_error_px': float(np.median(point_errors)) if point_errors else None,
        'max_point_error_px': float(np.max(point_errors)) if point_errors else None
    }

    return {'summary': summary, 'endpoint_rows': endpoint_rows}

    return {'summary': summary, 'endpoint_rows': endpoint_rows}


def build_endpoint_rows(record, detected_points, ground_truth_points, peak_values):
    """Build endpoint metric rows for one image."""
    rows = []

    for point_index, ((pred_x, pred_y), peak_value) in enumerate(zip(detected_points, peak_values), start=1):
        gt_x, gt_y, point_error = get_point_error(detected_points=detected_points, ground_truth_points=ground_truth_points, point_index=point_index)
        rows.append({
            'sample_name': record.sample_name,
            'image_path': record.image_path.as_posix(),
            'point_index': point_index,
            'pred_x': pred_x,
            'pred_y': pred_y,
            'gt_x': gt_x,
            'gt_y': gt_y,
            'point_error_px': point_error,
            'vote_peak': peak_value
        })

    return rows


def get_point_error(detected_points, ground_truth_points, point_index):
    """Return ground-truth coordinates and point error for one endpoint."""
    if ground_truth_points is None:
        return None, None, None

    pred_x, pred_y = detected_points[point_index - 1]
    gt_x, gt_y = ground_truth_points[point_index - 1]
    point_error = float(np.hypot(float(pred_x) - float(gt_x), float(pred_y) - float(gt_y)))
    return float(gt_x), float(gt_y), point_error


def resolve_device(device='auto'):
    """Resolve the requested inference device."""
    if device is None or str(device).lower() == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return torch.device(device)


def load_checkpoint(checkpoint_path):
    """Load a checkpoint on CPU before model construction."""
    try:
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location='cpu')


def load_model_from_checkpoint(checkpoint_path, device='auto'):
    """Load a self-describing IPV checkpoint and return the model plus inference metadata."""
    from ..quadruplet import Quadruplet

    device = resolve_device(device)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = extract_inference_metadata_from_checkpoint(checkpoint)
    model = Quadruplet(**metadata['init_args'])
    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return LoadedInferenceCheckpoint(model=model, checkpoint=checkpoint, metadata=metadata)


def extract_state_dict(checkpoint):
    """Extract and normalise a state_dict from a current self-describing checkpoint."""
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError('Checkpoint must be a current IPV self-describing checkpoint containing a state_dict.')

    return {str(key).replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}


def extract_inference_metadata_from_checkpoint(checkpoint):
    """Extract model and inference settings written by TrainModel.save_checkpoint."""
    metadata = checkpoint.get('metadata') if isinstance(checkpoint, dict) else None

    if not isinstance(metadata, dict):
        raise ValueError('Checkpoint is missing the current IPV metadata block. Re-train or re-save the checkpoint with the current package.')

    require_metadata_schema(metadata)
    task_metadata = require_dict(metadata, 'task')
    model_metadata = require_dict(metadata, 'model')
    preprocessing_metadata = require_dict(metadata, 'preprocessing')
    inference_metadata = require_dict(metadata, 'inference')
    init_args = require_dict(model_metadata, 'init_args')
    tasks_classes = normalise_tasks_classes(init_args.get('tasks_classes'))
    task_names = list(task_metadata.get('task_names', []))

    if task_names != ['distance', 'angle']:
        raise ValueError(f"Expected checkpoint task_names ['distance', 'angle'], got {task_names}.")

    output_heads = normalise_output_heads(task_metadata.get('output_heads'), task_metadata.get('num_output_heads'))
    vote_accumulation = inference_metadata.get('vote_accumulation', {}) if isinstance(inference_metadata.get('vote_accumulation', {}), dict) else {}
    smoothing_sigma = vote_accumulation.get('smoothing_sigma')
    checkpoint_info = metadata.get('checkpoint', {}) if isinstance(metadata.get('checkpoint', {}), dict) else {}
    required_init_keys = ['num_of_points', 'tasks_classes', 'network_name', 'branch_features', 'frozen_stages', 'small_input_stem', 'input_channels']
    missing_init_keys = [key for key in required_init_keys if key not in init_args]

    if missing_init_keys:
        raise ValueError(f'Checkpoint model init_args are missing required key(s): {missing_init_keys}')

    return {
        'raw_checkpoint_metadata': json_safe_metadata(metadata),
        'init_args': {
            'num_of_points': int(init_args['num_of_points']),
            'tasks_classes': tasks_classes,
            'network_name': str(init_args['network_name']),
            'branch_features': int(init_args['branch_features']),
            'frozen_stages': int(init_args['frozen_stages']),
            'small_input_stem': bool(init_args['small_input_stem']),
            'input_channels': int(init_args['input_channels'])
        },
        'task_name': str(task_metadata.get('name') or ''),
        'task_names': task_names,
        'output_heads': output_heads,
        'num_points': int(task_metadata['num_points']),
        'distance_intervals': tasks_classes[0],
        'angle_intervals': tasks_classes[1],
        'sub_patch_scales': [int(scale) for scale in preprocessing_metadata['sub_patch_scales']],
        'patch_size': int(preprocessing_metadata['patch_size']),
        'num_sub_patches': int(preprocessing_metadata['num_sub_patches']),
        'input_channels': int(preprocessing_metadata['input_channels']),
        'grid_spacing': int(inference_metadata['grid_spacing']),
        'smoothing_sigma': float(smoothing_sigma) if smoothing_sigma is not None else 7.0,
        'checkpoint_type': checkpoint_info.get('type')
    }


def build_config_from_checkpoint_metadata(metadata, output_dir, batch_size=2048, grid_spacing=None, smoothing_sigma=None, use_probability_weights=True,
                                          save_raw_vote_maps=False, checkpoint_path=None, run_label='inference', dimension_point_map=None):
    """Create a LandmarkInferenceConfig from checkpoint metadata and runtime overrides."""
    return LandmarkInferenceConfig(
        output_dir=Path(output_dir),
        num_points=int(metadata['num_points']),
        sub_patch_scales=metadata['sub_patch_scales'],
        distance_intervals=metadata['distance_intervals'],
        angle_intervals=metadata['angle_intervals'],
        grid_spacing=int(grid_spacing) if grid_spacing is not None else int(metadata['grid_spacing']),
        input_channels=int(metadata['input_channels']),
        task_name=str(metadata.get('task_name') or ''),
        batch_size=int(batch_size),
        smoothing_sigma=float(smoothing_sigma) if smoothing_sigma is not None else float(metadata.get('smoothing_sigma', 7.0)),
        use_probability_weights=bool(use_probability_weights),
        save_raw_vote_maps=bool(save_raw_vote_maps),
        checkpoint_path=checkpoint_path,
        checkpoint_type=metadata.get('checkpoint_type'),
        run_label=run_label,
        dimension_point_map=dimension_point_map
    )


def require_metadata_schema(metadata):
    """Validate that the checkpoint uses the current metadata schema."""
    if metadata.get('schema') != 'ipv_checkpoint_metadata':
        raise ValueError(f"Unsupported checkpoint metadata schema: {metadata.get('schema')}")

    required_sections = ['task', 'model', 'preprocessing', 'inference']
    missing_sections = [section for section in required_sections if section not in metadata]

    if missing_sections:
        raise ValueError(f'Checkpoint metadata is missing required section(s): {missing_sections}')


def require_dict(parent, key):
    """Return a required dictionary section."""
    value = parent.get(key)

    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata section '{key}' must be a dictionary.")

    return value


def normalise_tasks_classes(tasks_classes):
    """Convert distance and angle intervals into numeric lists."""
    if not isinstance(tasks_classes, (list, tuple)) or len(tasks_classes) != 2:
        raise ValueError('Checkpoint model init_args must contain distance and angle tasks_classes.')

    return [normalise_intervals(tasks_classes[0]), normalise_intervals(tasks_classes[1])]


def normalise_intervals(intervals):
    """Convert interval pairs into float pairs."""
    if not isinstance(intervals, (list, tuple)) or not intervals:
        raise ValueError('Interval metadata must be a non-empty list.')

    return [[float(lower), float(upper)] for lower, upper in intervals]


def normalise_output_heads(output_heads, expected_head_count):
    """Validate and order output-head metadata."""
    if not isinstance(output_heads, list) or not output_heads:
        raise ValueError('Checkpoint task metadata must contain output_heads.')

    output_heads = sorted(output_heads, key=lambda item: int(item['head_index']))

    if expected_head_count is not None and len(output_heads) != int(expected_head_count):
        raise ValueError(f'Checkpoint has {len(output_heads)} output heads, expected {expected_head_count}.')

    for head_index, output_head in enumerate(output_heads):
        if int(output_head['head_index']) != head_index:
            raise ValueError('Output head indices must be contiguous from zero.')

        if str(output_head['task']) not in {'distance', 'angle'}:
            raise ValueError(f"Unsupported output-head task: {output_head['task']}")

    return [{'head_index': int(item['head_index']), 'point_index': int(item['point_index']), 'task': str(item['task'])} for item in output_heads]


def json_safe_metadata(metadata):
    """Return checkpoint metadata in a JSON-safe form."""
    try:
        json.dumps(metadata, default=str)
        return metadata
    except TypeError:
        return json.loads(json.dumps(metadata, default=str))


def safe_file_stem(value):
    """Return a path-safe file stem."""
    safe_value = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._-')

    if not safe_value:
        raise ValueError(f'Could not create a safe output name from: {value}')

    return safe_value
