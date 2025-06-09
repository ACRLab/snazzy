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
