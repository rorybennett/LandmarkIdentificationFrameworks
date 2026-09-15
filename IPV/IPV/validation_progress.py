"""Fixed full-image validation examples from both best checkpoints."""
import json
import random
from .utils.landmark_inference_utils import build_validation_records, run_landmark_inference_for_records


class ValidationProgress:
    def __init__(self, trainer):
        self.interval = trainer.train_config.visualise_validation_progress_epochs
        count = trainer.train_config.visualise_validation_progress_images
        self.records = []
        if count > 0 and self.interval > 0:
            config = trainer.build_validation_inference_config(trainer.output_path / 'validation_progress')
            records = sorted(build_validation_records(config), key=lambda record: record.sample_name)
            self.records = random.Random(trainer.train_config.random_seed).sample(records, min(count, len(records)))
        self.sample_names = [record.sample_name for record in self.records]

    def render(self, trainer, model, epoch):
        if not self.records or epoch % self.interval:
            return
        state = trainer.clone_state_dict_to_cpu(model.state_dict())
        rng = trainer.capture_rng_state()
        generator_states = trainer.capture_data_loader_generator_states()
        was_training = model.training
        try:
            for kind in ('best_validation_loss', 'best_validation_pixel_error'):
                checkpoint_path = trainer.get_checkpoint_path(kind)
                checkpoint = trainer.load_checkpoint_state(model, checkpoint_path)
                output_dir = trainer.output_path / 'validation_progress' / f'epoch_{epoch:04d}' / kind
                config = trainer.build_validation_inference_config(output_dir, checkpoint_path, kind, checkpoint)
                config.save_raw_vote_maps = False
                run_landmark_inference_for_records(model, config, self.records, device=trainer.device)
                (output_dir / 'selection.json').write_text(json.dumps({
                    'observed_epoch': epoch, 'checkpoint_epoch': checkpoint['epoch'],
                    'checkpoint_type': kind, 'sample_names': self.sample_names,
                }, indent=2), encoding='utf-8')
        finally:
            model.load_state_dict(state, strict=True)
            model.train(was_training)
            trainer.restore_rng_state(rng)
            trainer.restore_data_loader_generator_states(generator_states)
