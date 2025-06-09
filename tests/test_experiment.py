import os
from pathlib import Path

import pytest

from pasna_analysis import Config, Experiment

VALID_DIR = Path("./tests/assets/data/20250210")


@pytest.fixture
def exp():
    return Experiment(VALID_DIR)


@pytest.fixture
def config():
    return Config(VALID_DIR)


@pytest.fixture(scope="session", autouse=True)
def clear_pd_params():
    yield
    pd_params_path = VALID_DIR.joinpath("peak_detection_params.json")
    if pd_params_path.exists():
        os.remove(pd_params_path)


def test_can_create_experiment(exp):
    assert exp is not None


def test_can_skip_peaks_before_first_peak_threshold(exp):
    # emb4 has first peak before 30 min and should be excluded
    assert exp.embryos is not None
    assert len(exp.embryos) == 2


def test_can_exclude_embryos(config):
    to_exclude = [1]
    config.update_params({"exp_params": {"to_exclude": to_exclude}})
    exp = Experiment(VALID_DIR, config)

    assert len(exp.embryos) == 1


def test_ignores_embryos_not_in_experiment(config):
    # VALID_DIR only contains emb1, emb3, and emb4
    to_exclude = [15]

    config.update_params({"exp_params": {"to_exclude": to_exclude}})
    exp = Experiment(VALID_DIR, config)

    assert len(exp.embryos) == 2


def test_can_use_kwargs():
    expected_dff_strategy = "local_minima"
    exp = Experiment(VALID_DIR, dff_strategy=expected_dff_strategy)

    pd_params = exp.config.get_pd_params()
    actual_strategy = pd_params.get("dff_strategy", None)

    assert actual_strategy == expected_dff_strategy


def test_can_use_all_valid_kwargs():
    expected_has_transients = True
    expected_to_exclude = [1]
    expected_dff_strategy = "local_minima"
    expected_first_peak_threshold = 35

    exp = Experiment(
        VALID_DIR,
        dff_strategy=expected_dff_strategy,
        has_transients=expected_has_transients,
        to_exclude=expected_to_exclude,
        first_peak_threshold=expected_first_peak_threshold,
    )

    pd_params = exp.config.get_pd_params()
    exp_params = exp.config.get_exp_params()
    actual_strategy = pd_params.get("dff_strategy", None)
    actual_to_exclude = exp_params.get("to_exclude", None)
    actual_has_transients = exp_params.get("has_transients", None)
    actual_first_peak_threshold = exp_params.get("first_peak_threshold", None)

    assert actual_strategy == expected_dff_strategy
    assert actual_to_exclude == expected_to_exclude
    assert actual_has_transients == expected_has_transients
    assert actual_first_peak_threshold == expected_first_peak_threshold


def test_ignores_invalid_kwargs(capsys):
    Experiment(VALID_DIR, invalid_kwarg="invalid")
    captured = capsys.readouterr()

    assert "WARN" in captured.out
