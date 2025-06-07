import pytest

pytest.importorskip("pandas")
pytest.importorskip("torch")

from src.simulation import uniform_betting, kelly_criterion


def test_uniform_betting_threshold():
    rule = uniform_betting(1000, bankroll_fraction=0.1, threshold=0.2)
    # Probability close to 0.5 should result in no bet
    assert rule(1000, 2.0, 0.55) == 0
    # Otherwise constant fraction of initial bankroll
    assert rule(1000, 2.0, 0.8) == 100


def test_kelly_criterion():
    rule = kelly_criterion(kelly_fraction=0.2, threshold=0.1)
    # Probability difference below threshold
    assert rule(500, 2.0, 0.55) == 0
    # Positive expected value bet
    expected = 500 * 0.2 * (2.0 * 0.6 - (1 - 0.6)) / 2.0
    assert abs(rule(500, 2.0, 0.6) - expected) < 1e-6
