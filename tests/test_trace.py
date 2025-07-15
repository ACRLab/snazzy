from pathlib import Path

import numpy as np
import pytest

from pasna_analysis import BaselineStrategies, Config, Trace

VALID_DIR = Path("./tests/assets/data/20250210")


@pytest.fixture(scope="function")
def config():
    return Config(VALID_DIR)


@pytest.fixture
def activity():
    csv_path = Path.joinpath(VALID_DIR, "activity", "emb1.csv")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data


def test_baseline_with_average_n_values():
    arr = [1, 2, 3, 4, 0, 1, 0, 7, 6]
    baseline = Trace.average_n_lowest_window(arr, window_size=5, n_lowest=2)

    # the expected values here assume that the padding mode is 'reflect'
    expected = np.array([1.5, 1.5, 0.5, 0.5, 0, 0, 0, 0.5, 0])

    assert len(baseline) == len(arr)

    assert np.allclose(baseline, expected)


def test_baseline_average_bottom_n_requires_odd_window():
    arr = [1, 2, 3, 4, 0, 1, 0, 7, 6]

    invalid_window_size = 6

    with pytest.raises(ValueError):
        Trace.average_n_lowest_window(arr, window_size=invalid_window_size, n_lowest=2)


def test_baseline_average_bottom_n_checks_window_size():
    arr = [1, 2, 3, 4, 0, 1, 0, 7, 6]

    invalid_window_size = len(arr) + 1

    with pytest.raises(ValueError):
        Trace.average_n_lowest_window(arr, window_size=invalid_window_size, n_lowest=2)


def test_use_manual_trim_idx_if_present(config: Config, activity):
    expected_trim_idx = 2000
    config.update_params({"embryos": {"emb1": {"manual_trim_idx": expected_trim_idx}}})
    trace = Trace("emb1", activity, config)

    calculated_trim_idx = trace.trim_idx

    assert calculated_trim_idx == expected_trim_idx


def test_trim_idx_is_last_point_when_not_hatches(config: Config, activity):
    # use a high zscore to make sure we won't reach this threshold
    high_zscore = 5
    config.update_params({"pd_params": {"trim_zscore": high_zscore}})
    trace = Trace("emb1", activity, config)

    calculated_trim_idx = trace.trim_idx
    expected_trim_idx = len(trace.struct)

    assert calculated_trim_idx == expected_trim_idx


def test_raises_when_unknown_baseline_strategy(config: Config, activity):
    baseline_strategy = "invalid_baseline"
    # make sure this strategy is not in the listed BaselineStrategies
    with pytest.raises(ValueError):
        BaselineStrategies(baseline_strategy)

    config.update_params({"pd_params": {"dff_strategy": baseline_strategy}})

    with pytest.raises(ValueError):
        Trace("emb1", activity, config)
