"""Loss contribution accounting, history compatibility and diagnostic plots."""
from types import SimpleNamespace
from unittest.mock import patch
import csv
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from Heatmaps.train_model import TrainModel


def trainer(tmp_path):
    value = TrainModel.__new__(TrainModel)
    value.output_path = tmp_path
    value.model_config = SimpleNamespace(auxiliary_loss_weight=0.6)
    value.train_config = SimpleNamespace(landmark_constraint_loss=None,
        constraint_angle_weight=2., constraint_side_weight=3.,
        constraint_temperature=.05, constraint_margin=.02)
    return value


@pytest.mark.parametrize('constraint,angle_weight,side_weight,expected', [
    (None, 2., 3., set()),
    ('prostate_taus', 2., 3., {'constraint_angle', 'constraint_side'}),
    ('prostate_taus', 0., 3., {'constraint_side'}),
    ('prostate-saus', 2., 3., {'constraint_side'}),
    ('prostate_taus', 0., 0., set()),
])
def test_weighted_components_and_gradients(tmp_path, constraint, angle_weight, side_weight, expected):
    value = trainer(tmp_path)
    value.train_config.landmark_constraint_loss = constraint
    value.train_config.constraint_angle_weight = angle_weight
    value.train_config.constraint_side_weight = side_weight
    output = torch.ones(2, 2, 4, 4, requires_grad=True)
    auxiliary = [output * 2, output * 3]
    target = torch.zeros_like(output)
    mask = torch.ones_like(output, dtype=torch.bool)
    batch = {'original_size': torch.tensor([[4, 4], [4, 4]]),
             'inverse_augmentation': torch.eye(3).repeat(2, 1, 1)}
    angle = output.sum() * (0 if constraint == 'prostate-saus' else .01)
    side = output.sum() * .02
    with patch('Heatmaps.heatmap_losses.constraint_terms', return_value=(angle, side)):
        loss, components = value.calculate_model_loss(output, auxiliary, target,
            torch.nn.MSELoss(reduction='none'), mask, batch, return_components=True)
    assert set(components) == {'heatmap', 'auxiliary_1', 'auxiliary_2'} | expected
    torch.testing.assert_close(loss, sum(components.values()))
    assert components['auxiliary_1'].item() == pytest.approx(1.2)
    assert components['auxiliary_2'].item() == pytest.approx(2.7)
    loss.backward()
    assert torch.isfinite(output.grad).all()
    value.model_config.auxiliary_loss_weight = 0
    value.train_config.landmark_constraint_loss = None
    _, disabled = value.calculate_model_loss(output, auxiliary, target,
        torch.nn.MSELoss(reduction='none'), mask, return_components=True)
    assert set(disabled) == {'heatmap'}


def append(history, epoch, components=False, rates=None):
    metrics = {'loss': 2., 'error_px': 3.}
    if components:
        metrics['loss_component_heatmap'] = 2.
    TrainModel.update_history(history, epoch, 'start', 'end', rates[0] if rates else .01, metrics, metrics,
                              1., 1., 2., epoch_lrs=rates)


def test_legacy_history_csv_and_missing_epochs(tmp_path):
    value = trainer(tmp_path)
    history = value.empty_history()
    append(history, 1)
    value.validate_history(history, 1)
    append(history, 2, True, [.01, .001])
    assert history['lr_2'] == [None, .001]
    assert history['training_loss_component_heatmap'] == [None, 2.]
    value.validate_history(history, 2)
    value.write_history_log(history)
    with value.get_log_path().open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['lr_2'] == '' and float(rows[1]['lr_2']) == .001
    history['lr_2'][1] = np.inf
    with pytest.raises(ValueError, match='non-finite'):
        value.validate_history(history, 2)
    history['lr_2'].pop()
    with pytest.raises(ValueError, match='inconsistent lengths'):
        value.validate_history(history, 2)


@pytest.mark.parametrize('rates,axis_count', [([.01], 1), ([.01, .001], 2)])
def test_plot_axes_and_only_recorded_components(tmp_path, rates, axis_count):
    value = trainer(tmp_path)
    history = value.empty_history()
    append(history, 1, True, rates)
    append(history, 2, True, [rate / 2 for rate in rates])
    captured = []
    from matplotlib.figure import Figure
    save = Figure.savefig
    def capture(figure, path, **kwargs):
        captured.append((path.name, figure))
        save(figure, path, **kwargs)
    with patch.object(Figure, 'savefig', capture):
        value.save_history_plot(history)
    plots = dict(captured)
    assert set(plots) == {'training_validation_plot.png', 'individual_loss_plot.png', 'learning_rate_plot.png'}
    combined = plots['training_validation_plot.png']
    assert len(combined.axes) == 2
    assert sum(len(axis.lines) for axis in combined.axes) == 4
    losses = plots['individual_loss_plot.png'].axes[0].lines
    assert len(losses) == 2
    assert [line.get_linestyle() for line in losses] == ['-', '--']
    rates_plot = plots['learning_rate_plot.png']
    assert len(rates_plot.axes) == axis_count
    for index, axis in enumerate(rates_plot.axes):
        assert axis.lines[0].get_linestyle() == ('-' if index == 0 else '--')
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [rates[index], rates[index] / 2])
    for name in plots:
        assert (tmp_path / name).stat().st_size > 1000
    plt.close('all')
