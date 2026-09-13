# MedSAM landmark model

`vit-medsam` uses the bundled original MedSAM/SAM ViT-B image encoder and a newly trained landmark heatmap decoder. It shares the trainer, data processing, anatomical losses, schedulers and validation outputs described in [README.md](README.md).

## Installation and weights

Install this Heatmaps package and PyTorch for your HPC environment. **MedSAM and segment_anything do not need to be installed.** No prompts, prompt encoder, mask decoder or imports from the standalone ViTPose project are used.

Supply the original `medsam_vit_b.pth` with `--pretrained-checkpoint`. The strict loader extracts every `image_encoder.*` tensor and ignores prompt/mask decoder tensors. It rejects missing or mismatched encoder weights. Training cannot silently fall back to a random encoder. Trained landmark checkpoints contain all model parameters and reconstruction metadata; the original weights are unnecessary for inference or resume.

Run `run_vit_medsam.ps1` or `run_vit_medsam.sh` after editing paths and activating your environment. Default canvas: 512; Gaussian sigma: 8. The common entry point is `Heatmaps.heatmap_training_pipeline --network-name vit-medsam`.

## Encoder implementation

`Heatmaps/models/sam/image_encoder.py` and `common.py` retain the upstream implementation, copyright headers and ViTDet reference. [MedSAM's source](https://github.com/bowang-lab/MedSAM/tree/main/segment_anything/modeling) is based on [Segment Anything](https://github.com/facebookresearch/segment-anything). Apache-2.0 licence, notice and source URLs/hashes are bundled alongside the files and included in package builds.

The encoder retains 768 channels, 12 blocks, 12 attention heads, 16-pixel patches, window size 14, global attention in blocks 2/5/8/11, relative positional attention and the original 256-channel neck. No timm or ViTDet package is required: its referenced attention operations are present in the bundled code.

Any integer canvas >=16 is accepted, fixed throughout a run. Internal right/bottom padding aligns to 16; decoded heatmaps are cropped back. Importing the original 1024 encoder interpolates the 64x64 absolute position grid and global relative positions to the configured grid; local window positions retain their original dimensions. Saved landmark weights reload strictly at their recorded canvas.

Greyscale is replicated to three channels, resized with aspect ratio preserved, then min-max scaled within image content. Padding remains zero. Dataset standardisation is disabled.

## Training and outputs

The decoder trains until validation pixel error plateaus and the minimum decoder epoch count is met. Joint training restores the best pixel checkpoint, resets optimiser/scheduler state, and enables the selected final encoder blocks plus pretrained neck; selecting all 12 also enables patch and position parameters. The maximum epoch budget applies throughout.

The shared trainer supports optional TAUS/SAUS geometry losses, transformed-coordinate handling and fixed validation progress images. At the end, `validation_best_loss/` and `validation_best_pixel_error/` contain matching CSV/workbook/overlay schemas with their own checkpoint metadata. See the README for every control and resume behaviour.
