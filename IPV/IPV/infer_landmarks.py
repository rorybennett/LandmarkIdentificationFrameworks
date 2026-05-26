"""
Example script for running IPV landmark inference on a user image or image folder.

Edit the path variables and switches below, then run from the repository root with:
python -m IPV.infer_landmarks
"""
from pathlib import Path

from .utils.landmark_inference_utils import build_config_from_checkpoint_metadata, build_image_records, load_model_from_checkpoint, run_landmark_inference_for_records

# ======================================================================================================================
# Paths
# ======================================================================================================================
MODEL_PATH = Path(r'D:\Coding\Testing\IPV_SAVING\prostate_transverse\small_cnn_fs0_stemfalse_ppts200'
                  r'\model_f1_best.pth')
INPUT_PATH = Path(r'D:\Coding\Testing\Val_Images_F1')
OUTPUT_DIR = Path(r'D:\Coding\Testing\InferenceResults')
GROUND_TRUTH_MARK_LIST_PATH = None

# ======================================================================================================================
# Inference switches
# ======================================================================================================================
DEVICE = 'auto'
BATCH_SIZE = 4096
GRID_SPACING_OVERRIDE = 10
VOTE_SMOOTH_SIGMA_OVERRIDE = None
USE_PROBABILITY_WEIGHTS = True
SAVE_RAW_VOTE_MAPS = False
RECURSIVE_IMAGE_SEARCH = False
SUPPORTED_IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
RUN_LABEL = 'inference'

# ======================================================================================================================
# Optional dimension summaries
# ======================================================================================================================
DIMENSION_POINT_MAP = None


# Example for a non-prostate four-point task:
# DIMENSION_POINT_MAP = {'vertical': (1, 3), 'horizontal': (2, 4)}


def build_inference_config(checkpoint_metadata):
    """Build the runtime config using checkpoint settings plus local overrides."""
    return build_config_from_checkpoint_metadata(metadata=checkpoint_metadata, output_dir=OUTPUT_DIR, batch_size=BATCH_SIZE, grid_spacing=GRID_SPACING_OVERRIDE,
                                                 smoothing_sigma=VOTE_SMOOTH_SIGMA_OVERRIDE, use_probability_weights=USE_PROBABILITY_WEIGHTS,
                                                 save_raw_vote_maps=SAVE_RAW_VOTE_MAPS, checkpoint_path=MODEL_PATH, run_label=RUN_LABEL,
                                                 dimension_point_map=DIMENSION_POINT_MAP)


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
