"""
Example script for running landmark inference with any trained Heatmaps model.

Edit the path variables and switches below, then run from the repository root with:
python -m Heatmaps.infer_landmarks
"""
from .utils.heatmap_inference_utils import (build_config_from_checkpoint_metadata, build_image_records,
                                            load_model_from_checkpoint, run_heatmap_inference_for_records)

# =====================================================================================================================
# Paths
# =====================================================================================================================
MODEL_PATH = ''
INPUT_PATH = ''
OUTPUT_DIR = ''
GROUND_TRUTH_MARK_LIST_PATH = ''

# =====================================================================================================================
# Inference switches
# =====================================================================================================================
DEVICE = 'cuda'
BATCH_SIZE = 1
SAVE_RAW_HEATMAPS = True
CLEAR_CUDA_CACHE_BETWEEN_BATCHES = True
RECURSIVE_IMAGE_SEARCH = False
SUPPORTED_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
RUN_LABEL = 'inference'


def build_inference_config(checkpoint_metadata):
    """Build runtime settings from checkpoint metadata plus local overrides."""
    return build_config_from_checkpoint_metadata(
        metadata=checkpoint_metadata,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        save_raw_heatmaps=SAVE_RAW_HEATMAPS,
        clear_cuda_cache_between_batches=CLEAR_CUDA_CACHE_BETWEEN_BATCHES,
        checkpoint_path=MODEL_PATH,
        run_label=RUN_LABEL,
    )


def main():
    """Load a trained checkpoint, discover images, and run heatmap inference."""
    for name, value in (('MODEL_PATH', MODEL_PATH), ('INPUT_PATH', INPUT_PATH), ('OUTPUT_DIR', OUTPUT_DIR)):
        if not str(value).strip():
            raise ValueError(f'Set {name} before running inference.')

    loaded_checkpoint = load_model_from_checkpoint(checkpoint_path=MODEL_PATH, device=DEVICE)
    config = build_inference_config(loaded_checkpoint.metadata)
    records = build_image_records(
        input_path=INPUT_PATH,
        num_points=config.num_points,
        mark_list_path=GROUND_TRUTH_MARK_LIST_PATH or None,
        recursive=RECURSIVE_IMAGE_SEARCH,
        supported_suffixes=SUPPORTED_IMAGE_SUFFIXES,
    )

    if not records:
        raise ValueError(f'No supported images found at {INPUT_PATH}')

    print(f'Loaded {config.network_name} model from {MODEL_PATH}', flush=True)
    print(f'Found {len(records)} image(s). Outputs will be saved to {OUTPUT_DIR}', flush=True)
    run_heatmap_inference_for_records(model=loaded_checkpoint.model, config=config, records=records, device=DEVICE)
    print(f'Inference complete. Outputs saved to {OUTPUT_DIR}.', flush=True)


if __name__ == '__main__':
    main()
