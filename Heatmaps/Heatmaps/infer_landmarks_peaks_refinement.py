"""Compare existing validation checkpoints with argmax and DARK-style decoding.

Edit the switches below, then run from the outer Heatmaps directory:
    python -m Heatmaps.infer_landmarks_peaks_refinement

No training takes place. Checkpoint preprocessing, weights and landmark order
are retained. The decoder settings are experimental, not established gains.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from openpyxl import Workbook

from .custom_dataset import HeatmapDataset, HeatmapDatasetConfig
from .models import unpack_heatmap_output
from .utils.heatmap_inference_utils import (
    HeatmapImageInferer,
    HeatmapImageRecord,
    build_config_from_checkpoint_metadata,
    build_prediction_rows,
    build_result,
    chunk_items,
    clear_device_memory,
    extract_inference_metadata_from_checkpoint,
    load_checkpoint,
    load_model_from_checkpoint,
    write_csv_rows,
    write_rows_to_sheet,
)
from .utils.io_utils import (
    get_split_file_path,
    is_all_fold,
    read_split_names,
    safe_file_stem,
    scale_points_to_original,
    validate_fold_split_overlaps,
    validate_split_duplicates,
)
from .utils.peak_refinement import PeakRefinementConfig, decode_peak_comparison

# Paths: each entry may be a checkpoint file or a run/results directory searched
# recursively. Include the sagittal and transverse run roots for the 30-model study.
MODEL_PATHS = [r"C:\Storage\Landmark\TR_hourglass\prostate_sagittal\stacked_hourglass_sigma12_weight10", r"C:\Storage\Landmark\TR_hourglass\prostate_transverse\stacked_hourglass_sigma12_weight10"]  # e.g. [r'C:\results\prostate_sagittal\run', r'C:\results\prostate_transverse\run']
OUTPUT_DIR = r"C:\Storage\GeneratedFiles\LandmarkIdentification\StackedHourglassCoordinateExtraction"  # A NEW directory, separate from the training results.
CHECKPOINT_FILENAME = "model_best_validation_pixel_error.pth"
EXPECTED_CHECKPOINT_COUNT = 30  # Set to None for another number or a single checkpoint.

# Empty paths reuse the checkpoint's saved locations. Override these when moving
# from the cluster. FOLD_LISTS_PATH contains repetition_N/val_fN.txt.
IMAGE_DATA_DIR = ""
FOLD_LISTS_PATH = r"C:\Storage\Datasets\LandmarkIdentification\OriginalIPVData\folds_network_study"
MARK_LIST_FILE = ""
# Per-task paths take precedence over the global paths above, for example:
# {'prostate_sagittal': {'image_data_dir': r'C:\data\SAGITTAL',
#                       'fold_lists_path': r'C:\data\folds',
#                       'mark_list_file': r'C:\data\sagittal_marks.txt'},
#  'prostate_transverse': {...}}
DATASET_OVERRIDES = {
    "prostate_sagittal": {
        "image_data_dir": r"C:\Storage\Datasets\LandmarkIdentification\OriginalIPVData\SAGITTAL",
        "mark_list_file": r"C:\Storage\Datasets\LandmarkIdentification\OriginalIPVData\doctors_resampled_sagittalMarkList.txt",
    },
    "prostate_transverse": {
        "image_data_dir": r"C:\Storage\Datasets\LandmarkIdentification\OriginalIPVData\TRANSVERSE",
        "mark_list_file": r"C:\Storage\Datasets\LandmarkIdentification\OriginalIPVData\doctors_resampled_transverseMarkList.txt",
    },
}

# Verify the initial study's saved training settings; these do not alter weights.
# Set to {} when comparing other models/configurations. Actual settings are logged.
REQUIRED_TRAINING_SETTINGS = {
    "network_name": "stacked_hourglass",
    "hourglass_stacks": 6,
    "heatmap_sigma": 12.0,
    "positive_weight": 10.0,
    "image_size": 512,
    "loss_name": "weighted_mse",
    "landmark_constraint_loss": None,
}

DEVICE = "auto"
BATCH_SIZE = 1
REFINE_NATIVE_HOURGLASS = True  # Capture the final head before bilinear enlargement.
BLUR_KERNEL = 3  # Odd; 1 disables smoothing. Units: decoding-grid pixels.
BLUR_SIGMA = 0.0  # Zero uses OpenCV's kernel-derived sigma.
MAX_REFINEMENT_OFFSET = 1.0  # Maximum absolute shift per axis in decoding-grid pixels.
ERROR_THRESHOLDS_PX = (5.0, 10.0, 20.0)
LARGE_ERROR_THRESHOLD_PX = 20.0
SAVE_OVERLAYS = True
SAVE_RAW_HEATMAPS = True
COMPARE_EXISTING_VALIDATION = True
REFERENCE_COORDINATE_TOLERANCE_PX = 0.05


@dataclass
class ComparisonConfig:
    model_paths: list
    output_dir: str | Path
    checkpoint_filename: str = "model_best_validation_pixel_error.pth"
    expected_count: int | None = None
    required_settings: dict = field(default_factory=dict)
    data_paths: dict = field(default_factory=dict)
    dataset_overrides: dict = field(default_factory=dict)
    device: str = "auto"
    batch_size: int = 1
    native_hourglass: bool = True
    refinement: PeakRefinementConfig = field(default_factory=PeakRefinementConfig)
    thresholds: tuple = (5.0, 10.0, 20.0)
    large_error_threshold: float = 20.0
    save_overlays: bool = False
    save_raw_heatmaps: bool = False
    compare_existing: bool = True
    reference_tolerance: float = 0.05


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str, allow_nan=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_settings(metadata):
    raw = metadata["raw_checkpoint_metadata"]
    values = {}
    for section in ("data_config", "train_config", "model_config"):
        values.update(raw.get("raw_configs", {}).get(section, {}))
    values.update(raw.get("training", {}).get("train_config", {}))
    values.update(metadata["init_args"])
    values.update(
        network_name=metadata["network_name"], image_size=metadata["image_size"]
    )
    if "heatmap_sigma" in raw["preprocessing"]:
        values["heatmap_sigma"] = raw["preprocessing"]["heatmap_sigma"]
    return values


def discover_checkpoints(config):
    """Find one explicitly selected checkpoint type and omit full-data folds."""
    if not config.model_paths or any(
        not str(path).strip() for path in config.model_paths
    ):
        raise ValueError(
            "Set MODEL_PATHS to checkpoint files or model run directories."
        )
    found = set()
    for value in config.model_paths:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            found.add(path)
        elif path.is_dir():
            found.update(path.rglob(config.checkpoint_filename))
        else:
            raise FileNotFoundError(f"Model path does not exist: {path}")
    return sorted(
        (
            path
            for path in found
            if not any(part.lower() == "fold_all" for part in path.parts)
        ),
        key=str,
    )


def validation_records(metadata, config):
    """Use the exact validation lists and the existing annotation/image resolver."""
    paths = dict(metadata["raw_checkpoint_metadata"].get("data", {}))
    paths.update({key: value for key, value in config.data_paths.items() if value})
    paths.update(
        {
            key: value
            for key, value in config.dataset_overrides.get(
                metadata["task_name"], {}
            ).items()
            if value
        }
    )
    for key in ("image_data_dir", "fold_lists_path", "mark_list_file"):
        if not paths.get(key):
            raise ValueError(f"Supply {key} for task {metadata['task_name']!r}.")
        paths[key] = Path(paths[key]).expanduser().resolve()
    repetition, fold = metadata["repetition"], metadata["fold"]
    names = read_split_names(paths["fold_lists_path"], repetition, "validation", fold)
    validate_split_duplicates("validation", names, repetition, fold)
    if get_split_file_path(
        paths["fold_lists_path"], repetition, "training", fold
    ).is_file():
        validate_fold_split_overlaps(paths["fold_lists_path"], repetition, fold)
    dataset = HeatmapDataset(
        HeatmapDatasetConfig(
            repetition=repetition,
            fold=fold,
            split_name="validation",
            num_of_points=metadata["num_points"],
            fold_lists_path=paths["fold_lists_path"],
            mark_list_file=paths["mark_list_file"],
            image_data_dir=paths["image_data_dir"],
            image_size=metadata["image_size"],
            heatmap_sigma=1.0,
            input_channels=metadata["input_channels"],
            enforce_greyscale=metadata["enforce_greyscale"],
            recursive_image_search=bool(paths.get("recursive_image_search", False)),
        )
    )  # Only records are used; no targets or augmentation are generated.
    records = [
        HeatmapImageRecord(item["sample_name"], item["image_path"], item["points"])
        for item in dataset.records
    ]
    split = get_split_file_path(
        paths["fold_lists_path"], repetition, "validation", fold
    )
    return records, {
        key: str(paths[key])
        for key in ("image_data_dir", "fold_lists_path", "mark_list_file")
    } | {
        "validation_list": str(split),
        "validation_list_sha256": file_sha256(split),
        "mark_list_sha256": file_sha256(paths["mark_list_file"]),
    }


def prepare_jobs(config):
    """Validate every checkpoint and split before creating experiment outputs."""
    jobs = []
    identities = set()
    for path in discover_checkpoints(config):
        payload = load_checkpoint(path)
        metadata = extract_inference_metadata_from_checkpoint(payload)
        del payload
        if metadata["fold"] is not None and is_all_fold(metadata["fold"]):
            continue
        if metadata["fold"] is None or metadata["repetition"] is None:
            raise ValueError(
                f"Validation inference requires repetition and fold metadata: {path}"
            )
        settings = training_settings(metadata)
        missing = {
            key: {"expected": value, "actual": settings.get(key, "<missing>")}
            for key, value in config.required_settings.items()
            if key not in settings or settings[key] != value
        }
        if missing:
            raise ValueError(
                f"Checkpoint does not match REQUIRED_TRAINING_SETTINGS: {path}\n{missing}"
            )
        records, inputs = validation_records(metadata, config)
        run_root = (
            path.parent.parent.parent
            if path.parent.name.startswith("fold_")
            else path.parent
        )
        run_id = (
            safe_file_stem(run_root.name)
            + "_"
            + hashlib.sha256(str(run_root).encode()).hexdigest()[:10]
        )
        series = "/".join(
            (
                safe_file_stem(metadata["task_name"]),
                metadata["network_name"],
                run_id,
                path.stem,
            )
        )
        identity = (series, metadata["repetition"], metadata["fold"])
        if identity in identities:
            raise ValueError(
                f"Duplicate checkpoint for the same run/repetition/fold: {path}"
            )
        identities.add(identity)
        relative = (
            Path(series)
            / f"repetition_{metadata['repetition']}"
            / f"fold_{metadata['fold']}"
        )
        jobs.append(
            dict(
                path=path,
                metadata=metadata,
                settings=settings,
                records=records,
                inputs=inputs,
                series=series,
                relative=relative,
            )
        )
    if not jobs:
        raise ValueError("No numbered validation-fold checkpoints were found.")
    if config.expected_count is not None and len(jobs) != config.expected_count:
        raise ValueError(
            f"Expected {config.expected_count} validation checkpoints, found {len(jobs)}. Check MODEL_PATHS or EXPECTED_CHECKPOINT_COUNT."
        )
    return jobs


def forward_for_comparison(model, image_batch, use_native):
    """Run once, optionally observing the hourglass final head without changing it."""
    captured = []
    handle = None
    if use_native:
        handle = model.heatmap_heads[-1].register_forward_hook(
            lambda _module, _args, output: captured.append(output)
        )
    try:
        with torch.inference_mode():
            canvas, _ = unpack_heatmap_output(model(image_batch))
            native = model.apply_output_activation(captured[0]) if use_native else None
    finally:
        if handle is not None:
            handle.remove()
    return canvas, native


def export_validation(output_dir, results):
    """Match the existing training validation CSV filenames, fields and worksheets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [result["summary"] for result in results]
    endpoints = [row for result in results for row in result["endpoint_rows"]]
    write_csv_rows(output_dir / "validation_image_summary.csv", images)
    write_csv_rows(output_dir / "validation_endpoints.csv", endpoints)
    write_csv_rows(
        output_dir / "validation_predictions.csv",
        build_prediction_rows(images, endpoints),
    )
    workbook = Workbook()
    workbook.active.title = "validation_image_summary"
    write_rows_to_sheet(workbook.active, images)
    write_rows_to_sheet(workbook.create_sheet("validation_endpoints"), endpoints)
    workbook.save(output_dir / "validation_summary.xlsx")


def error_summary(rows, config):
    errors = np.asarray([row["error_px"] for row in rows], dtype=float)
    large = errors > config.large_error_threshold
    result = {
        "endpoint_count": len(errors),
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
        "p90_error_px": float(np.percentile(errors, 90)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "max_error_px": float(errors.max()),
        "large_error_threshold_px": config.large_error_threshold,
        "large_error_count": int(large.sum()),
        "large_error_percent": float(large.mean() * 100),
        "large_error_mean_px": float(errors[large].mean()) if large.any() else None,
    }
    for threshold in config.thresholds:
        result[f"within_{threshold:g}px_percent"] = float(
            (errors <= threshold).mean() * 100
        )
    return result


def compare_saved_argmax(job, results, tolerance):
    """Audit rerun coordinates against matching saved validation, when available."""
    directory = {
        "best_validation_pixel_error": "validation_best_pixel_error",
        "best_validation_loss": "validation_best_loss",
    }.get(job["metadata"]["checkpoint_type"])
    reference = (
        None
        if directory is None
        else job["path"].parent / directory / "validation_endpoints.csv"
    )
    if reference is None or not reference.is_file():
        return {
            "status": "reference_unavailable",
            "path": str(reference) if reference else None,
        }
    with reference.open(encoding="utf-8-sig", newline="") as handle:
        saved = list(csv.DictReader(handle))
    key = lambda row: (row["sample_name"], int(row["point_index"]))
    expected = {key(row): row for row in saved}
    current = {key(row): row for result in results for row in result["endpoint_rows"]}
    if len(expected) != len(saved) or expected.keys() != current.keys():
        raise ValueError(
            f"Saved validation and rerun sample/landmark membership differ: {reference}"
        )
    differences = []
    for identity, row in current.items():
        old = expected[identity]
        if not np.allclose(
            [float(old["target_x"]), float(old["target_y"])],
            [row["target_x"], row["target_y"]],
            atol=1e-4,
            rtol=0,
        ):
            raise ValueError(
                f"Saved validation ground truth differs for {identity}: {reference}"
            )
        differences.append(
            np.linalg.norm(
                [
                    float(old["pred_x"]) - row["pred_x"],
                    float(old["pred_y"]) - row["pred_y"],
                ]
            )
        )
    maximum = float(max(differences))
    return {
        "status": "matched" if maximum <= tolerance else "coordinate_differences",
        "path": str(reference),
        "max_coordinate_difference_px": maximum,
        "tolerance_px": tolerance,
        "endpoints_outside_tolerance": int(
            sum(value > tolerance for value in differences)
        ),
    }


def evaluate_checkpoint(job, config, output_root):
    loaded = load_model_from_checkpoint(job["path"], device=config.device)
    use_native = (
        config.native_hourglass
        and loaded.metadata["network_name"] == "stacked_hourglass"
    )
    methods = ["argmax", "dark"] + (["native_argmax"] if use_native else [])
    output = output_root / job["relative"]
    runtime = build_config_from_checkpoint_metadata(
        loaded.metadata,
        output,
        batch_size=config.batch_size,
        checkpoint_path=job["path"],
    )
    inferer = HeatmapImageInferer(loaded.model, runtime, device=config.device)
    results = {method: [] for method in methods}
    paired = []
    grid_shape = None
    for batch_records in chunk_items(job["records"], config.batch_size):
        prepared = [inferer.prepare_record(record) for record in batch_records]
        images = torch.stack([item["image"] for item in prepared]).to(inferer.device)
        sizes = [item["original_size"] for item in prepared]
        canvas, native = forward_for_comparison(loaded.model, images, use_native)
        inferer.validate_heatmap_batch(canvas, len(batch_records))
        decoded, statuses = decode_peak_comparison(
            canvas, sizes, runtime.image_size, config.refinement, native
        )
        grid_shape = list((native if use_native else canvas).shape[-2:])
        coordinates = {
            method: scale_points_to_original(points, sizes, runtime.image_size)
            .cpu()
            .numpy()
            for method, points in decoded.items()
        }
        for index, item in enumerate(prepared):
            for method in methods:
                result = build_result(
                    item["record"],
                    coordinates[method][index],
                    item["ground_truth_points"],
                    item["original_size"],
                    runtime.checkpoint_type,
                    runtime.network_name,
                    runtime.repetition,
                    runtime.fold,
                )
                result["summary"]["dataset_split"] = "validation"
                for row in result["endpoint_rows"]:
                    row["dataset_split"] = "validation"
                results[method].append(result)
                if config.save_overlays:
                    original_dir = inferer.config.output_dir
                    inferer.config.output_dir = output / method
                    inferer.output_dirs = inferer.get_output_dirs()
                    inferer.prepare_output_dirs()
                    inferer.save_visual_outputs(
                        item["record"],
                        coordinates[method][index],
                        item["ground_truth_points"],
                        canvas[index].cpu().numpy(),
                    )
                    inferer.config.output_dir = original_dir
            for point_index, (baseline, refined) in enumerate(
                zip(
                    results["argmax"][-1]["endpoint_rows"],
                    results["dark"][-1]["endpoint_rows"],
                )
            ):
                paired.append(
                    {
                        "series": job["series"],
                        "task_name": runtime.task_name,
                        "network_name": runtime.network_name,
                        "checkpoint_path": str(job["path"]),
                        "repetition": runtime.repetition,
                        "fold": runtime.fold,
                        "sample_name": baseline["sample_name"],
                        "point_index": baseline["point_index"],
                        "target_x": baseline["target_x"],
                        "target_y": baseline["target_y"],
                        "argmax_x": baseline["pred_x"],
                        "argmax_y": baseline["pred_y"],
                        "dark_x": refined["pred_x"],
                        "dark_y": refined["pred_y"],
                        "argmax_error_px": baseline["error_px"],
                        "dark_error_px": refined["error_px"],
                        "error_change_px": refined["error_px"] - baseline["error_px"],
                        "coordinate_shift_px": float(
                            np.linalg.norm(
                                coordinates["dark"][index, point_index]
                                - coordinates["argmax"][index, point_index]
                            )
                        ),
                        "refinement_status": statuses[index][point_index],
                    }
                )
            if config.save_raw_heatmaps:
                raw_dir = output / "raw_heatmaps"
                raw_dir.mkdir(exist_ok=True)
                arrays = {"canvas": canvas[index].cpu().numpy()}
                if native is not None:
                    arrays["native"] = native[index].cpu().numpy()
                np.savez_compressed(
                    raw_dir / f"{safe_file_stem(item['record'].sample_name)}.npz",
                    **arrays,
                )
        del images, canvas, native, decoded
    audit = (
        compare_saved_argmax(job, results["argmax"], config.reference_tolerance)
        if config.compare_existing
        else {"status": "disabled"}
    )
    summaries = []
    for method, method_results in results.items():
        export_validation(output / method, method_results)
        endpoints = [
            row for result in method_results for row in result["endpoint_rows"]
        ]
        summaries.append(
            {
                "series": job["series"],
                "task_name": runtime.task_name,
                "network_name": runtime.network_name,
                "checkpoint_path": str(job["path"]),
                "repetition": runtime.repetition,
                "fold": runtime.fold,
                "decoder": method,
                "image_count": len(method_results),
                **error_summary(endpoints, config),
            }
        )
    write_csv_rows(output / "paired_endpoints.csv", paired)
    write_csv_rows(output / "decoder_summary.csv", summaries)
    write_json(
        output / "comparison_metadata.json",
        {
            "schema": "heatmap_peak_refinement_comparison",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": job["path"],
            "checkpoint_sha256": file_sha256(job["path"]),
            "checkpoint_metadata": loaded.metadata["raw_checkpoint_metadata"],
            "resolved_inputs": job["inputs"],
            "records": [
                {"sample_name": record.sample_name, "image_path": record.image_path}
                for record in job["records"]
            ],
            "refinement": asdict(config.refinement),
            "refinement_space": "native_hourglass" if use_native else "canvas",
            "decoding_grid_shape": grid_shape,
            "canvas_size": runtime.image_size,
            "native_mapping": "(coordinate + 0.5) * canvas_size / native_size - 0.5",
            "fallback": "existing_canvas_argmax",
            "refinement_status_counts": dict(
                Counter(row["refinement_status"] for row in paired)
            ),
            "saved_argmax_audit": audit,
            "torch_version": torch.__version__,
            "device": str(inferer.device),
            "settings": asdict(config),
        },
    )
    print(
        f"  {len(job['records'])} validation images; saved argmax audit: {audit['status']}",
        flush=True,
    )
    resolved_device = inferer.device
    del inferer, loaded
    clear_device_memory(resolved_device)
    return results, summaries, paired


def run_comparison(config):
    if not str(config.output_dir).strip():
        raise ValueError("Set OUTPUT_DIR to a new experiment directory.")
    if (
        config.batch_size < 1
        or not config.thresholds
        or any(not np.isfinite(t) or t <= 0 for t in config.thresholds)
    ):
        raise ValueError(
            "Use a positive batch size and positive finite error thresholds."
        )
    if (
        not np.isfinite(config.large_error_threshold)
        or config.large_error_threshold <= 0
    ):
        raise ValueError("large_error_threshold must be positive and finite.")
    if not np.isfinite(config.reference_tolerance) or config.reference_tolerance < 0:
        raise ValueError("reference_tolerance must be non-negative and finite.")
    output_root = Path(config.output_dir).expanduser().resolve()
    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        raise ValueError(
            f"OUTPUT_DIR must be new or empty; existing results are preserved: {output_root}"
        )
    jobs = prepare_jobs(config)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "settings": asdict(config),
        "checkpoint_count": len(jobs),
        "completed_checkpoints": [],
        "checkpoints": [
            {
                "path": job["path"],
                "series": job["series"],
                "inputs": job["inputs"],
                "training_settings": job["settings"],
            }
            for job in jobs
        ],
    }
    write_json(output_root / "experiment.json", manifest)
    fold_rows, paired_rows = [], []
    grouped = defaultdict(list)
    try:
        for index, job in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] {job['path']}", flush=True)
            results, summaries, paired = evaluate_checkpoint(job, config, output_root)
            fold_rows.extend(summaries)
            paired_rows.extend(paired)
            for method, method_results in results.items():
                grouped[(job["series"], method)].extend(
                    row for result in method_results for row in result["endpoint_rows"]
                )
            manifest["completed_checkpoints"].append(str(job["path"]))
            write_json(output_root / "experiment.json", manifest)
        aggregate = []
        for (series, method), endpoints in grouped.items():
            relevant = [
                row
                for row in fold_rows
                if row["series"] == series and row["decoder"] == method
            ]
            aggregate.append(
                {
                    "series": series,
                    "decoder": method,
                    "checkpoint_count": len(relevant),
                    "macro_fold_mean_error_px": float(
                        np.mean([row["mean_error_px"] for row in relevant])
                    ),
                    **error_summary(endpoints, config),
                }
            )
        changes = []
        for series in sorted({row["series"] for row in paired_rows}):
            group = [row for row in paired_rows if row["series"] == series]
            delta = np.asarray([row["error_change_px"] for row in group])
            baseline_large = [
                row
                for row in group
                if row["argmax_error_px"] > config.large_error_threshold
            ]
            changes.append(
                {
                    "series": series,
                    "endpoint_count": len(group),
                    "mean_error_change_px": float(delta.mean()),
                    "improved_count": int((delta < -1e-6).sum()),
                    "worsened_count": int((delta > 1e-6).sum()),
                    "unchanged_count": int((np.abs(delta) <= 1e-6).sum()),
                    "refined_count": sum(
                        row["refinement_status"] == "refined" for row in group
                    ),
                    "baseline_large_error_count": len(baseline_large),
                    "baseline_large_mean_change_px": float(
                        np.mean([row["error_change_px"] for row in baseline_large])
                    )
                    if baseline_large
                    else None,
                }
            )
        write_csv_rows(output_root / "fold_decoder_summary.csv", fold_rows)
        write_csv_rows(output_root / "aggregate_decoder_summary.csv", aggregate)
        write_csv_rows(output_root / "paired_endpoints.csv", paired_rows)
        write_csv_rows(output_root / "paired_change_summary.csv", changes)
        manifest["status"] = "complete"
        manifest["aggregation_note"] = (
            "Pooled endpoints include repeated evaluations across repetitions; they are not independent patients. Series/tasks are kept separate. Negative paired error change favours refinement."
        )
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output_root / "experiment.json", manifest)
    print(f"Comparison complete: {output_root}", flush=True)
    return fold_rows


def main():
    run_comparison(
        ComparisonConfig(
            model_paths=MODEL_PATHS,
            output_dir=OUTPUT_DIR,
            checkpoint_filename=CHECKPOINT_FILENAME,
            expected_count=EXPECTED_CHECKPOINT_COUNT,
            required_settings=REQUIRED_TRAINING_SETTINGS,
            data_paths={
                "image_data_dir": IMAGE_DATA_DIR,
                "fold_lists_path": FOLD_LISTS_PATH,
                "mark_list_file": MARK_LIST_FILE,
            },
            dataset_overrides=DATASET_OVERRIDES,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            native_hourglass=REFINE_NATIVE_HOURGLASS,
            refinement=PeakRefinementConfig(
                blur_kernel=BLUR_KERNEL,
                blur_sigma=BLUR_SIGMA,
                max_offset=MAX_REFINEMENT_OFFSET,
            ),
            thresholds=ERROR_THRESHOLDS_PX,
            large_error_threshold=LARGE_ERROR_THRESHOLD_PX,
            save_overlays=SAVE_OVERLAYS,
            save_raw_heatmaps=SAVE_RAW_HEATMAPS,
            compare_existing=COMPARE_EXISTING_VALIDATION,
            reference_tolerance=REFERENCE_COORDINATE_TOLERANCE_PX,
        )
    )


if __name__ == "__main__":
    main()
