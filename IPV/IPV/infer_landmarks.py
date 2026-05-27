"""
Example script for running IPV landmark inference on a user image or image folder.

Edit the path variables and switches below, then run from the repository root with:
python -m IPV.infer_landmarks
"""
from pathlib import Path

import cv2

from .utils.landmark_inference_utils import build_config_from_checkpoint_metadata, build_image_records, load_model_from_checkpoint, run_landmark_inference_for_records

# ======================================================================================================================
# Paths
# ======================================================================================================================
MODEL_PATH = Path(r'D:\Coding\Testing\IPV_SAVING\prostate_transverse\small_cnn_fs0_stemfalse_ppts200\model_f1_best.pth')
INPUT_PATH = Path(r'D:\Datasets\IPV\OriginalData\TRANSVERSE')
OUTPUT_DIR = Path(r'D:\Coding\Testing\InferenceResults')
GROUND_TRUTH_MARK_LIST_PATH = None

# ======================================================================================================================
# Inference switches
# ======================================================================================================================
DEVICE = 'auto'
BATCH_SIZE = 4096
GRID_SPACING_OVERRIDE = None
VOTE_SMOOTH_SIGMA_OVERRIDE = 3
USE_PROBABILITY_WEIGHTS = True
SAVE_RAW_VOTE_MAPS = False
CLEAR_CUDA_CACHE_BETWEEN_IMAGES = True
RECURSIVE_IMAGE_SEARCH = False
SUPPORTED_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
RUN_LABEL = 'inference'
PATCH_RESIZE_INTERPOLATION = cv2.INTER_AREA

# ======================================================================================================================
# Multiprocessing switches
# ======================================================================================================================
PARALLEL_PATCH_GENERATION = True
PATCH_WORKERS = 4
PATCH_CHUNKSIZE = 32
PARALLEL_VOTE_ACCUMULATION = True
VOTE_WORKERS = None
MULTIPROCESS_CONTEXT = 'spawn'


def build_inference_config(checkpoint_metadata):
    """Build the runtime config using checkpoint settings plus local overrides."""
    return build_config_from_checkpoint_metadata(metadata=checkpoint_metadata, output_dir=OUTPUT_DIR, batch_size=BATCH_SIZE, grid_spacing=GRID_SPACING_OVERRIDE,
                                                 smoothing_sigma=VOTE_SMOOTH_SIGMA_OVERRIDE, use_probability_weights=USE_PROBABILITY_WEIGHTS,
                                                 save_raw_vote_maps=SAVE_RAW_VOTE_MAPS, clear_cuda_cache_between_images=CLEAR_CUDA_CACHE_BETWEEN_IMAGES,
                                                 checkpoint_path=MODEL_PATH, run_label=RUN_LABEL,
                                                 parallel_patch_generation=PARALLEL_PATCH_GENERATION, patch_workers=PATCH_WORKERS,
                                                 patch_chunksize=PATCH_CHUNKSIZE, parallel_vote_accumulation=PARALLEL_VOTE_ACCUMULATION,
                                                 vote_workers=VOTE_WORKERS, multiprocess_context=MULTIPROCESS_CONTEXT,
                                                 patch_resize_interpolation=PATCH_RESIZE_INTERPOLATION)


def main():
    """Load a trained checkpoint, build image records, and run landmark inference."""
    loaded_checkpoint = load_model_from_checkpoint(checkpoint_path=MODEL_PATH, device=DEVICE)
    config = build_inference_config(loaded_checkpoint.metadata)
    records = build_image_records(input_path=INPUT_PATH, num_points=config.num_points, mark_list_path=GROUND_TRUTH_MARK_LIST_PATH,
                                  recursive=RECURSIVE_IMAGE_SEARCH, supported_suffixes=SUPPORTED_IMAGE_SUFFIXES)

    if not records:
        raise ValueError(f'No supported images found at {INPUT_PATH}')

    print(f'Loaded model from {MODEL_PATH}', flush=True)
    print(f'Found {len(records)} image(s). Outputs will be saved to {OUTPUT_DIR}', flush=True)
    run_landmark_inference_for_records(model=loaded_checkpoint.model, config=config, records=records, device=DEVICE)
    print(f'Inference complete. Outputs saved to {OUTPUT_DIR}.', flush=True)


if __name__ == '__main__':
    main()
