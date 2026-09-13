# ViTPose encoder attribution

The local ViTPose-compatible encoder in vitpose.py follows the architecture and checkpoint naming in https://github.com/ViTAE-Transformer/ViTPose/blob/main/mmpose/models/backbones/vit.py (ViTPose contributors; Apache License 2.0, reproduced in LICENSE_VITPOSE).

The adaptation removes the mmpose/mmcv/timm installation requirements, replaces the pose head with the shared landmark decoder, adds a trainable projection and configured positional-grid interpolation, and integrates staged fine-tuning. The standard ViTPose-B COCO 256x192 backbone contract is retained; this is not the complete original pose estimation framework.

Bundled MedSAM/SAM source has its separate upstream notices and licence in sam/.
