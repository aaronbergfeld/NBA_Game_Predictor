import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from collections import OrderedDict
from src.model_definitions import BrierScoreLoss, class_specific_ece, train_test_split


def test_brier_score_loss():
    loss_fn = BrierScoreLoss()
    y_pred = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    y_true = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = loss_fn(y_pred, y_true)
    assert abs(loss.item() - 0.2) < 1e-6


def test_class_specific_ece_perfect():
    y_pred = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = np.array([0, 1])
    assert class_specific_ece(y_true, y_pred, n_classes=2, n_bins=2) == 0.0


def test_train_test_split():
    seasons = OrderedDict([(2020, [1, 2]), (2021, [3, 4]), (2022, [5, 6])])
    train, test = train_test_split(seasons, 2021)
    assert train == [1, 2]
    assert test == [3, 4]
