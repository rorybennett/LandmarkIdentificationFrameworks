"""Fixed validation examples evaluated against both best checkpoints during training."""

import json
import random
from pathlib import Path
import torch
from .models import unpack_heatmap_output
from .utils.io_utils import heatmaps_to_points, scale_points_to_original, safe_file_stem
from .utils.visualisation_utils import save_validation_overlays


class ValidationProgress:
    def __init__(self, dataset, count, interval, seed, output_dir):
        self.dataset = dataset
        self.interval = interval
        self.output_dir = Path(output_dir)
        self.indices = (
            random.Random(seed).sample(range(len(dataset)), min(count, len(dataset)))
            if count > 0 and interval > 0
            else []
        )
        self.sample_names = [dataset.records[index]['sample_name'] for index in self.indices]

    def render(self, trainer, model, epoch):
        if not self.indices or epoch % self.interval:
            return
        state = trainer.clone_state_dict_to_cpu(model.state_dict())
        rng = trainer.capture_rng_state()
        was_training = model.training
        try:
            model.eval()
            with torch.inference_mode():
                for kind in ('best_validation_loss', 'best_validation_pixel_error'):
                    payload = trainer.load_checkpoint_state(model, trainer.get_checkpoint_path(kind))
                    output = self.output_dir / f'epoch_{epoch:04d}' / kind
                    output.mkdir(parents=True, exist_ok=True)
                    for index in self.indices:
                        sample = self.dataset[index]
                        image = sample['image'].unsqueeze(0).to(trainer.device)
                        sizes = sample['original_size'].unsqueeze(0).to(trainer.device)
                        heatmaps, _ = unpack_heatmap_output(model(image))
                        points = heatmaps_to_points(heatmaps, sizes, trainer.data_config.image_size)
                        points = scale_points_to_original(points, sizes, trainer.data_config.image_size)
                        save_validation_overlays(
                            image_path=Path(sample['image_path']),
                            output_dir=output,
                            output_stem=safe_file_stem(sample['sample_name']),
                            target_points=sample['points_original'].numpy(),
                            predicted_points=points[0].cpu().numpy(),
                            predicted_heatmaps=heatmaps[0].cpu().numpy(),
                        )
                    (output / 'selection.json').write_text(
                        json.dumps(
                            {
                                'observed_epoch': epoch,
                                'checkpoint_epoch': payload['epoch'],
                                'checkpoint_type': kind,
                                'sample_names': self.sample_names,
                            },
                            indent=2,
                        )
                    )
        finally:
            model.load_state_dict(state, strict=True)
            model.train(was_training)
            trainer.restore_rng_state(rng)
