"""Plateau control, scheduler trajectories and best-weight restoration."""
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from Heatmaps.train_model import TrainModel
from medsam_helpers import configs, pretrained, encoder_double, limited_threads


def test_plateau_requires_patience_and_minimum(tmp_path, pretrained):
    data, config, model = configs(tmp_path, pretrained)
    trainer = TrainModel(data, replace(config, max_training_epochs=10, freeze_encoder_epochs=5, decoder_plateau_patience=2), model, tmp_path/'run', device='cpu')
    for epoch, error in enumerate([10, 9, 9, 8, 8], 1):
        trainer.update_decoder_plateau(epoch, error)
        assert not trainer.stage_control['pending_transition']
    trainer.update_decoder_plateau(6, 8)
    assert trainer.stage_control['pending_transition']


@pytest.mark.parametrize('schedule', ['none', 'step', 'plateau', 'linear', 'cosine', 'exponential'])
def test_scheduler_decreases_and_resumes(tmp_path, pretrained, schedule):
    data, config, model = configs(tmp_path, pretrained)
    trainer = TrainModel(data, replace(config, visualise_validation_progress_images=1, visualise_validation_progress_epochs=2, lr_schedule=schedule, lr_decay_epochs=4, lr_plateau_patience=0), model, tmp_path/'run', device='cpu')
    def build():
        optimiser = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.01)
        return optimiser, trainer.build_scheduler(optimiser)
    opt, scheduler = build()
    rates=[]
    for epoch in range(7):
        opt.step()
        if scheduler is not None:
            scheduler.step(10.) if schedule == 'plateau' else scheduler.step()
        rates.append(opt.param_groups[0]['lr'])
        if epoch == 2:
            other_opt, other = build()
            other_opt.load_state_dict(opt.state_dict())
            if other is not None:
                other.load_state_dict(scheduler.state_dict())
        elif epoch > 2:
            other_opt.step()
            if other is not None:
                other.step(10.) if schedule == 'plateau' else other.step()
            assert other_opt.param_groups[0]['lr'] == pytest.approx(rates[-1])
    assert all(a >= b for a,b in zip(rates, rates[1:]))
    assert schedule == 'none' or rates[-1] < .01
    if schedule in ('linear', 'cosine'):
        assert rates[-1] == pytest.approx(.0001)


def test_transition_restores_best_decoder_and_resets_optimiser(tmp_path, pretrained):
    data, config, model = configs(tmp_path, pretrained)
    trainer = TrainModel(data, replace(config, decoder_plateau_patience=1, save_validation_predictions=False), model, tmp_path/'run', device='cpu')
    original = trainer.train_epoch
    calls = 0
    def observe(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            assert not kwargs['optimiser'].state
            for name, value in kwargs['model'].state_dict().items():
                torch.testing.assert_close(value.cpu(), trainer.best_pixel_checkpoint['state_dict'][name], rtol=0, atol=0)
            assert kwargs['optimiser'].param_groups[0]['lr'] == pytest.approx(config.learning_rate * .5)
            assert kwargs['model'].image_encoder.neck.weight.requires_grad
        return original(**kwargs)
    with patch.object(trainer, 'train_epoch', side_effect=observe), patch.object(trainer, 'validate', side_effect=[{'loss':1.,'error_px':10.}, {'loss':.9,'error_px':11.}, {'loss':.8,'error_px':12.}]):
        trainer.train()
    assert calls == 3
    assert trainer.best_pixel_checkpoint['epoch'] == 1

@pytest.mark.parametrize('schedule', ['plateau', 'cosine', 'step'])
@pytest.mark.parametrize('interrupt_at', [3, 4])
def test_resume_matches_uninterrupted_across_unfreeze(tmp_path, pretrained, schedule, interrupt_at):
    data, train, model = configs(tmp_path, pretrained)
    train = replace(train, save_validation_predictions=False, max_training_epochs=4, decoder_plateau_patience=1, lr_schedule=schedule)
    reference = TrainModel(replace(data), replace(train), model, tmp_path / 'reference', device=torch.device('cpu'))
    with patch.object(reference, 'validate', return_value={'loss': 1., 'error_px': 10.}):
        reference.train()
    interrupted = TrainModel(replace(data), replace(train), model, tmp_path / 'resumed', device=torch.device('cpu'))
    original_epoch = interrupted.train_epoch
    calls = 0
    def interrupt(**kwargs):
        nonlocal calls
        calls += 1
        if calls == interrupt_at:
            raise KeyboardInterrupt()
        return original_epoch(**kwargs)
    with patch.object(interrupted, 'train_epoch', side_effect=interrupt), patch.object(interrupted, 'validate', return_value={'loss': 1., 'error_px': 10.}):
        with pytest.raises(KeyboardInterrupt):
            interrupted.train()
    snapshot = torch.load(tmp_path / 'resumed/model_last_epoch.pth', weights_only=False)
    original_weights = torch.load(pretrained, weights_only=True)
    for name, value in snapshot['state_dict'].items():
        if interrupt_at == 3 and name.startswith('image_encoder.'):
            torch.testing.assert_close(value, original_weights[name], rtol=0, atol=0)
    pretrained.unlink()
    resumed = TrainModel(replace(data), replace(train), model, tmp_path / 'resumed', device=torch.device('cpu'), resume_training=True)
    with patch.object(resumed, 'validate', return_value={'loss': 1., 'error_px': 10.}):
        resumed.train()
    expected = torch.load(tmp_path / 'reference/model_last_epoch.pth', weights_only=False)
    actual = torch.load(tmp_path / 'resumed/model_last_epoch.pth', weights_only=False)
    for name, value in expected['state_dict'].items():
        torch.testing.assert_close(value, actual['state_dict'][name], rtol=0, atol=0)
    for name in ('training_loss', 'validation_loss', 'training_error_px', 'validation_error_px', 'lr', 'lr_2', 'training_loss_component_heatmap'):
        assert expected['training_state']['history'][name] == actual['training_state']['history'][name]
    assert not torch.equal(actual['state_dict']['image_encoder.neck.weight'], original_weights['image_encoder.neck.weight'])
    groups = actual['optimiser_state_dict']['param_groups']
    assert groups[0]['lr'] == pytest.approx(groups[1]['lr'] * 5)


    assert actual['training_state']['stage_control']['joint_start_epoch'] == 3
