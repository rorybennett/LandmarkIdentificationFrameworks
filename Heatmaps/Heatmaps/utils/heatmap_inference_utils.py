"""Reusable standalone inference utilities for trained heatmap landmark models."""

import csv
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from openpyxl import Workbook

from ..model_registry import build_heatmap_model
from ..models import unpack_heatmap_output
from .annotation_utils import read_mark_list, validate_annotation_point_count
from .io_utils import (heatmaps_to_points, load_image_as_float, resize_channel_first, safe_file_stem,
                       scale_points_to_original, validate_points_within_image)
from .progress_bar import ProgressBar
from .visualisation_utils import create_combined_heatmap_overlay, create_point_overlay, load_display_image

CHECKPOINT_METADATA_SCHEMA = 'heatmap_checkpoint_metadata'
SUPPORTED_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


@dataclass
class HeatmapInferenceConfig:
    """Runtime settings reconstructed from a self-describing heatmap checkpoint."""

    output_dir: Path
    num_points: int
    input_channels: int
    image_size: tuple[int, int]
    task_name: str = ''
    repetition: int | None = None
    fold: int | None = None
    checkpoint_path: Path | None = None
    checkpoint_type: str | None = None
    network_name: str | None = None
    batch_size: int = 1
    save_raw_heatmaps: bool = False
    clear_cuda_cache_between_batches: bool = True
    run_label: str = 'inference'
    checkpoint_metadata: dict | None = None


@dataclass
class HeatmapImageRecord:
    """One image and its optional original-coordinate ground truth."""

    sample_name: str
    image_path: Path
    ground_truth_points: list | None = None


@dataclass
class LoadedInferenceCheckpoint:
    """A reconstructed model together with its checkpoint and normalised metadata."""

    model: torch.nn.Module
    checkpoint: dict
    metadata: dict


class HeatmapImageInferer:
    """Run full-image inference and write comparison-ready results."""

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
        """Normalise and validate inference settings."""
        config.output_dir = Path(config.output_dir)
        config.checkpoint_path = None if config.checkpoint_path is None else Path(config.checkpoint_path)
        config.num_points = int(config.num_points)
        config.input_channels = int(config.input_channels)
        config.image_size = tuple(int(value) for value in config.image_size)
        config.repetition = None if config.repetition is None else int(config.repetition)
        config.fold = None if config.fold is None else int(config.fold)
        config.batch_size = int(config.batch_size)
        config.save_raw_heatmaps = bool(config.save_raw_heatmaps)
        config.clear_cuda_cache_between_batches = bool(config.clear_cuda_cache_between_batches)
        config.run_label = safe_file_stem(config.run_label)

        if config.num_points < 1:
            raise ValueError('num_points must be at least 1.')

        if config.input_channels not in (1, 3, 4):
            raise ValueError(f'input_channels must be 1, 3, or 4. Got: {config.input_channels}')

        if len(config.image_size) != 2 or min(config.image_size) < 1:
            raise ValueError(f'image_size must contain two positive values. Got: {config.image_size}')

        if config.batch_size < 1:
            raise ValueError('batch_size must be at least 1.')

        return config

    def get_output_dirs(self):
        """Return the visual, array, and log output directories."""
        prefix = build_summary_prefix(self.config.run_label)
        return {
            'heatmap_overlays': self.config.output_dir / f'{prefix}_heatmap_overlays',
            'point_overlays': self.config.output_dir / f'{prefix}_point_overlays',
            'raw_heatmaps': self.config.output_dir / f'{prefix}_raw_heatmaps',
            'logs': self.config.output_dir / f'{prefix}_logs',
        }

    def prepare_output_dirs(self):
        """Create only the directories required by the selected outputs."""
        self.config.output_dir.mkdir(exist_ok=True, parents=True)

        for name, output_dir in self.output_dirs.items():
            if name == 'raw_heatmaps' and not self.config.save_raw_heatmaps:
                continue

            output_dir.mkdir(exist_ok=True, parents=True)

    def infer_records(self, records):
        """Run inference for every record and save combined summaries."""
        records = list(records)

        if not records:
            raise ValueError('At least one image record is required for inference.')

        results = []
        batches = list(chunk_items(records, self.config.batch_size))

        with ProgressBar(total=len(records), label=f'{self.config.run_label} inference') as progress_bar:
            for batch_records in batches:
                progress_bar.set_status(', '.join(record.sample_name for record in batch_records[:2]))

                try:
                    results.extend(self.infer_batch(batch_records))
                finally:
                    if self.config.clear_cuda_cache_between_batches:
                        clear_device_memory(self.device)

                progress_bar.update(len(batch_records))

        output_paths = self.save_combined_summaries(results)
        self.save_run_metadata(records=records, results=results, output_paths=output_paths)
        return results

    def infer_batch(self, records):
        """Preprocess and infer one same-resolution model batch."""
        prepared = [self.prepare_record(record) for record in records]
        image_batch = torch.stack([item['image'] for item in prepared], dim=0).to(self.device, non_blocking=True)
        original_sizes = torch.as_tensor([item['original_size'] for item in prepared], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            model_output = self.model(image_batch)
            heatmaps, _ = unpack_heatmap_output(model_output)

        self.validate_heatmap_batch(heatmaps=heatmaps, expected_batch_size=len(records))
        resized_points = heatmaps_to_points(heatmaps)
        original_points = scale_points_to_original(points=resized_points, original_sizes=original_sizes,
                                                   image_size=self.config.image_size)
        heatmap_arrays = heatmaps.detach().cpu().numpy()
        point_arrays = original_points.detach().cpu().numpy()
        results = []

        for index, item in enumerate(prepared):
            result = build_result(record=item['record'], predicted_points=point_arrays[index],
                                  ground_truth_points=item['ground_truth_points'], original_size=item['original_size'],
                                  checkpoint_type=self.config.checkpoint_type, network_name=self.config.network_name,
                                  repetition=self.config.repetition, fold=self.config.fold)
            self.save_visual_outputs(record=item['record'], predicted_points=point_arrays[index],
                                     ground_truth_points=item['ground_truth_points'], heatmaps=heatmap_arrays[index])
            results.append(result)

        del image_batch, model_output, heatmaps, resized_points, original_points
        return results

    def prepare_record(self, record):
        """Load and validate one image using the training preprocessing contract."""
        record = HeatmapImageRecord(sample_name=str(record.sample_name), image_path=Path(record.image_path),
                                    ground_truth_points=record.ground_truth_points)
        image = load_inference_image_as_float(record.image_path, input_channels=self.config.input_channels)
        original_size = tuple(int(value) for value in image.shape[1:3])
        ground_truth_points = None

        if record.ground_truth_points is not None:
            ground_truth_points = validate_points_within_image(
                points=record.ground_truth_points, image_size=original_size,
                sample_name=record.sample_name, image_path=record.image_path,
            )

            if len(ground_truth_points) != self.config.num_points:
                raise ValueError(
                    f'Ground truth for {record.sample_name} contains {len(ground_truth_points)} point(s); '
                    f'exactly {self.config.num_points} are required.'
                )

        resized_image = resize_channel_first(image=image, image_size=self.config.image_size)
        return {
            'record': record,
            'image': torch.from_numpy(resized_image).float(),
            'original_size': original_size,
            'ground_truth_points': ground_truth_points,
        }

    def validate_heatmap_batch(self, heatmaps, expected_batch_size):
        """Validate the common final output contract for every registered model."""
        expected_shape = (int(expected_batch_size), self.config.num_points, *self.config.image_size)

        if not torch.is_tensor(heatmaps):
            raise TypeError('The model final heatmaps must be a tensor.')

        if tuple(heatmaps.shape) != expected_shape:
            raise ValueError(f'Model produced heatmaps with shape {tuple(heatmaps.shape)}, expected {expected_shape}.')

        if not torch.isfinite(heatmaps).all():
            raise ValueError('Model produced NaN or infinite heatmap values.')

    def save_visual_outputs(self, record, predicted_points, ground_truth_points, heatmaps):
        """Save the combined heatmap and endpoint overlays for one image."""
        display_image = load_display_image(record.image_path)
        output_stem = safe_file_stem(record.sample_name)
        heatmap_overlay = create_combined_heatmap_overlay(display_image=display_image, heatmaps=heatmaps,
                                                          predicted_points=predicted_points)
        point_overlay = create_point_overlay(display_image=display_image, detected_points=predicted_points,
                                             ground_truth_points=ground_truth_points)
        heatmap_path = self.output_dirs['heatmap_overlays'] / f'{output_stem}_{self.config.run_label}_heatmap_overlay.png'
        point_path = self.output_dirs['point_overlays'] / f'{output_stem}_{self.config.run_label}_points_overlay.png'

        if not cv2.imwrite(str(heatmap_path), heatmap_overlay):
            raise OSError(f'Could not write heatmap overlay: {heatmap_path}')

        if not cv2.imwrite(str(point_path), point_overlay):
            raise OSError(f'Could not write point overlay: {point_path}')

        if self.config.save_raw_heatmaps:
            np.save(self.output_dirs['raw_heatmaps'] / f'{output_stem}_{self.config.run_label}_heatmaps.npy', heatmaps)

    def save_combined_summaries(self, results):
        """Write image, endpoint, and wide prediction summaries."""
        summary_rows = [result['summary'] for result in results]
        endpoint_rows = [row for result in results for row in result['endpoint_rows']]
        prediction_rows = build_prediction_rows(summary_rows=summary_rows, endpoint_rows=endpoint_rows)
        prefix = build_summary_prefix(self.config.run_label)
        output_paths = {
            'summary_xlsx': self.config.output_dir / f'{prefix}_summary.xlsx',
            'image_summary_csv': self.config.output_dir / f'{prefix}_image_summary.csv',
            'endpoints_csv': self.config.output_dir / f'{prefix}_endpoints.csv',
            'predictions_csv': self.config.output_dir / f'{prefix}_predictions.csv',
        }
        write_summary_workbook(output_paths['summary_xlsx'], summary_rows=summary_rows, endpoint_rows=endpoint_rows)
        write_csv_rows(output_paths['image_summary_csv'], summary_rows)
        write_csv_rows(output_paths['endpoints_csv'], endpoint_rows)
        write_csv_rows(output_paths['predictions_csv'], prediction_rows)
        return output_paths

    def save_run_metadata(self, records, results, output_paths):
        """Record resolved inference settings and the exact checkpoint metadata."""
        config_metadata = asdict(self.config)
        checkpoint_metadata = config_metadata.pop('checkpoint_metadata')
        metadata = {
            'schema': 'heatmap_inference_run_metadata',
            'image_count': len(records),
            'result_count': len(results),
            'config': config_metadata,
            'output_paths': {name: str(path) for name, path in output_paths.items()},
            'records': [{'sample_name': record.sample_name, 'image_path': str(Path(record.image_path))} for record in records],
            'checkpoint_metadata': checkpoint_metadata,
        }
        metadata_path = self.output_dirs['logs'] / f'{build_summary_prefix(self.config.run_label)}_run_metadata.json'

        with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=4, default=str)


def run_heatmap_inference_for_records(model, config, records, device=None):
    """Run inference from an already reconstructed model and explicit image records."""
    return HeatmapImageInferer(model=model, config=config, device=device).infer_records(records)


def load_inference_image_as_float(image_path, input_channels):
    """Load an image, replicating one greyscale channel when the model expects RGB."""
    try:
        return load_image_as_float(image_path, input_channels=input_channels)
    except ValueError as channel_error:
        if int(input_channels) != 3:
            raise

        try:
            greyscale_image = load_image_as_float(image_path, input_channels=1)
        except ValueError:
            raise channel_error

        return np.repeat(greyscale_image, repeats=3, axis=0).astype(np.float32)


def build_image_records(input_path, num_points, mark_list_path=None, recursive=False,
                        supported_suffixes=SUPPORTED_IMAGE_SUFFIXES):
    """Build image records from one image or a directory, with optional ground truth."""
    image_paths = find_images(input_path=input_path, recursive=recursive, supported_suffixes=supported_suffixes)
    mark_records = read_mark_list(mark_list_path) if mark_list_path is not None else {}
    records = []
    seen_names = set()

    for image_path in image_paths:
        sample_name = image_path.stem

        if sample_name in seen_names:
            raise ValueError(f'Multiple input images share the sample name {sample_name!r}; output names would collide.')

        seen_names.add(sample_name)
        mark_record = mark_records.get(sample_name)
        ground_truth_points = None

        if mark_record is not None:
            validate_annotation_point_count(mark_record=mark_record, expected_points=num_points, sample_name=sample_name,
                                            mark_list_file=mark_list_path)
            ground_truth_points = mark_record['points']

        records.append(HeatmapImageRecord(sample_name=sample_name, image_path=image_path,
                                          ground_truth_points=ground_truth_points))

    return records


def find_images(input_path, recursive=False, supported_suffixes=SUPPORTED_IMAGE_SUFFIXES):
    """Return a supported image path or all supported images in a directory."""
    input_path = Path(input_path)
    suffixes = tuple(str(suffix).lower() for suffix in supported_suffixes)

    if input_path.is_file():
        if input_path.suffix.lower() not in suffixes:
            raise ValueError(f'Unsupported input image suffix: {input_path.suffix}')

        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f'Input path does not exist: {input_path}')

    iterator = input_path.rglob('*') if recursive else input_path.iterdir()
    return sorted((path for path in iterator if path.is_file() and path.suffix.lower() in suffixes),
                  key=lambda path: path.as_posix().lower())


def load_model_from_checkpoint(checkpoint_path, device='auto'):
    """Reconstruct any registered heatmap model from a self-describing checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_path}')

    device = resolve_device(device)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = extract_inference_metadata_from_checkpoint(checkpoint)
    init_args = dict(metadata['init_args'])
    init_args.pop('num_of_points', None)
    init_args.pop('input_channels', None)
    init_args.pop('image_size', None)
    model = build_heatmap_model(network_name=metadata['network_name'], num_of_points=metadata['num_points'],
                                input_channels=metadata['input_channels'], image_size=metadata['image_size'], **init_args)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.to(device)
    model.eval()
    return LoadedInferenceCheckpoint(model=model, checkpoint=checkpoint, metadata=metadata)


def load_checkpoint(checkpoint_path):
    """Load a checkpoint on CPU before model construction."""
    try:
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location='cpu')


def extract_state_dict(checkpoint):
    """Extract weights and remove an optional DataParallel prefix."""
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('state_dict'), dict):
        raise ValueError('Checkpoint must be a self-describing Heatmaps checkpoint containing a state_dict.')

    return {str(key).removeprefix('module.'): value for key, value in checkpoint['state_dict'].items()}


def extract_inference_metadata_from_checkpoint(checkpoint):
    """Validate and normalise reconstruction settings saved during training."""
    metadata = checkpoint.get('metadata') if isinstance(checkpoint, dict) else None

    if not isinstance(metadata, dict):
        raise ValueError('Checkpoint is missing Heatmaps metadata. Use a checkpoint saved by the current training pipeline.')

    if metadata.get('schema') != CHECKPOINT_METADATA_SCHEMA:
        raise ValueError(f"Unsupported checkpoint metadata schema: {metadata.get('schema')}")

    task = require_dict(metadata, 'task')
    model = require_dict(metadata, 'model')
    preprocessing = require_dict(metadata, 'preprocessing')
    inference = require_dict(metadata, 'inference')
    init_args = dict(require_dict(model, 'init_args'))
    image_size_metadata = require_dict(preprocessing, 'image_size')
    image_size = (int(image_size_metadata['height']), int(image_size_metadata['width']))
    num_points = int(task['num_points'])
    input_channels = int(preprocessing['input_channels'])
    network_name = str(model['registry_name'])
    required_init_args = {'num_of_points', 'input_channels'}
    missing_init_args = sorted(required_init_args - set(init_args))

    if missing_init_args:
        raise ValueError(f'Checkpoint model init_args are missing required key(s): {missing_init_args}')

    if int(init_args['num_of_points']) != num_points:
        raise ValueError('Checkpoint task and model metadata disagree about the number of landmarks.')

    if int(init_args['input_channels']) != input_channels:
        raise ValueError('Checkpoint preprocessing and model metadata disagree about input channels.')

    if inference.get('heatmap_to_point') != 'argmax' or not bool(inference.get('scale_back_to_original')):
        raise ValueError('Checkpoint records an unsupported heatmap inference convention.')

    checkpoint_info = metadata.get('checkpoint', {}) if isinstance(metadata.get('checkpoint'), dict) else {}
    data = metadata.get('data', {}) if isinstance(metadata.get('data'), dict) else {}
    return {
        'raw_checkpoint_metadata': json_safe_metadata(metadata),
        'init_args': init_args,
        'network_name': network_name,
        'num_points': num_points,
        'input_channels': input_channels,
        'image_size': image_size,
        'task_name': str(task.get('name') or ''),
        'repetition': data.get('repetition', task.get('repetition')),
        'fold': data.get('fold', task.get('fold')),
        'checkpoint_type': checkpoint_info.get('type', checkpoint.get('checkpoint_type')),
    }


def build_config_from_checkpoint_metadata(metadata, output_dir, batch_size=1, save_raw_heatmaps=False,
                                          clear_cuda_cache_between_batches=True, checkpoint_path=None,
                                          run_label='inference'):
    """Create runtime inference settings from checkpoint metadata and local overrides."""
    config = HeatmapInferenceConfig(
        output_dir=Path(output_dir),
        num_points=int(metadata['num_points']),
        input_channels=int(metadata['input_channels']),
        image_size=tuple(metadata['image_size']),
        task_name=str(metadata.get('task_name') or ''),
        repetition=metadata.get('repetition'),
        fold=metadata.get('fold'),
        checkpoint_path=None if checkpoint_path is None else Path(checkpoint_path),
        checkpoint_type=metadata.get('checkpoint_type'),
        network_name=metadata.get('network_name'),
        batch_size=int(batch_size),
        save_raw_heatmaps=bool(save_raw_heatmaps),
        clear_cuda_cache_between_batches=bool(clear_cuda_cache_between_batches),
        run_label=run_label,
        checkpoint_metadata=metadata.get('raw_checkpoint_metadata'),
    )
    return config


def build_result(record, predicted_points, ground_truth_points, original_size, checkpoint_type=None,
                 network_name=None, repetition=None, fold=None):
    """Build image- and endpoint-level output rows for one prediction."""
    predicted_points = np.asarray(predicted_points, dtype=np.float32)
    target_points = None if ground_truth_points is None else np.asarray(ground_truth_points, dtype=np.float32)
    errors = None if target_points is None else np.linalg.norm(predicted_points - target_points, axis=1)
    image_height, image_width = (int(value) for value in original_size)
    summary = {
        'dataset_split': 'inference',
        'repetition': repetition,
        'fold': fold,
        'sample_name': record.sample_name,
        'image_path': str(record.image_path),
        'image_height': image_height,
        'image_width': image_width,
        'num_points': len(predicted_points),
        'mean_error_px': None if errors is None else float(np.mean(errors)),
        'median_error_px': None if errors is None else float(np.median(errors)),
        'max_error_px': None if errors is None else float(np.max(errors)),
        'checkpoint_type': checkpoint_type,
        'network_name': network_name,
    }
    endpoint_rows = []

    for point_index, predicted in enumerate(predicted_points, start=1):
        target = None if target_points is None else target_points[point_index - 1]
        endpoint_rows.append({
            'dataset_split': 'inference',
            'repetition': repetition,
            'fold': fold,
            'sample_name': record.sample_name,
            'image_path': str(record.image_path),
            'point_index': point_index,
            'target_x': None if target is None else float(target[0]),
            'target_y': None if target is None else float(target[1]),
            'pred_x': float(predicted[0]),
            'pred_y': float(predicted[1]),
            'error_px': None if errors is None else float(errors[point_index - 1]),
            'checkpoint_type': checkpoint_type,
            'network_name': network_name,
        })

    return {'summary': summary, 'endpoint_rows': endpoint_rows}


def build_prediction_rows(summary_rows, endpoint_rows):
    """Build one comparison-ready wide prediction row per image."""
    endpoints_by_sample = {}

    for endpoint in endpoint_rows:
        endpoints_by_sample.setdefault(endpoint['sample_name'], []).append(endpoint)

    rows = []

    for summary in summary_rows:
        row = {
            'dataset_split': summary['dataset_split'],
            'repetition': summary['repetition'],
            'fold': summary['fold'],
            'sample_name': summary['sample_name'],
            'mean_error_px': summary['mean_error_px'],
        }

        for endpoint in sorted(endpoints_by_sample.get(summary['sample_name'], []), key=lambda item: item['point_index']):
            point_index = int(endpoint['point_index'])
            row[f'target_x{point_index}'] = endpoint['target_x']
            row[f'target_y{point_index}'] = endpoint['target_y']
            row[f'pred_x{point_index}'] = endpoint['pred_x']
            row[f'pred_y{point_index}'] = endpoint['pred_y']
            row[f'error_px{point_index}'] = endpoint['error_px']

        rows.append(row)

    return rows


def write_summary_workbook(output_path, summary_rows, endpoint_rows):
    """Write an IPV-compatible two-sheet Excel summary."""
    workbook = Workbook()
    image_sheet = workbook.active
    image_sheet.title = 'image_summary'
    write_rows_to_sheet(image_sheet, summary_rows)
    endpoint_sheet = workbook.create_sheet('endpoints')
    write_rows_to_sheet(endpoint_sheet, endpoint_rows)
    workbook.save(output_path)


def write_rows_to_sheet(sheet, rows):
    """Write dictionaries to a worksheet with stable field order."""
    if not rows:
        return

    headers = list(rows[0].keys())
    sheet.append(headers)

    for row in rows:
        sheet.append([row.get(header) for header in headers])


def write_csv_rows(output_path, rows):
    """Write dictionaries to CSV, preserving a stable first-row field order."""
    if not rows:
        return

    with open(output_path, 'w', encoding='utf-8', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_prefix(run_label):
    """Return the base name for inference outputs."""
    run_label = safe_file_stem(run_label)
    return run_label if run_label == 'inference' else f'{run_label}_inference'


def chunk_items(items, chunk_size):
    """Yield fixed-size lists from a sequence."""
    for start in range(0, len(items), int(chunk_size)):
        yield items[start:start + int(chunk_size)]


def resolve_device(device='auto'):
    """Resolve an explicit device or select CUDA when available."""
    if device is None or str(device).lower() == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return torch.device(device)


def clear_device_memory(device=None):
    """Release cached CUDA memory without unloading the model."""
    gc.collect()

    if device is None:
        return

    resolved_device = resolve_device(device)

    if resolved_device.type != 'cuda' or not torch.cuda.is_available():
        return

    torch.cuda.synchronize(resolved_device)
    torch.cuda.empty_cache()


def require_dict(parent, key):
    """Return a required dictionary-valued metadata section."""
    value = parent.get(key)

    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata section '{key}' must be a dictionary.")

    return value


def json_safe_metadata(metadata):
    """Return a detached JSON-safe copy of checkpoint metadata."""
    return json.loads(json.dumps(metadata, default=str))
