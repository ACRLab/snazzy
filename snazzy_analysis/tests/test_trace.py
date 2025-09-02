from pathlib import Path

import numpy as np
import pytest

from snazzy_analysis import BaselineStrategies, Config, Trace

VALID_DIR = Path(__file__).parent.joinpath("assets", "data", "20250210")


def read_from_csv(csv_path):
    return np.loadtxt(csv_path, delimiter=",", skiprows=1)


@pytest.fixture(scope="function")
def config():
    return Config(VALID_DIR)


@pytest.fixture
def activity():
    csv_path = Path.joinpath(VALID_DIR, "activity", "emb1.csv")
    return read_from_csv(csv_path)


def test_can_access_trace_props(config, activity):
    trace = Trace("emb1", activity, config)
    assert trace.peak_idxes is not None
    assert trace.peak_times is not None
    assert trace.peak_intervals is not None
    assert trace.peak_amplitudes is not None
    assert trace.peak_bounds_indices is not None
    assert trace.peak_bounds_times is not None
    assert trace.peak_durations is not None
    assert trace.peak_rise_times is not None
    assert trace.peak_decay_times is not None
    assert trace.peak_aucs is not None
    assert trace.rms is not None


def test_can_process_trace_without_peaks(config):
    csv_path = Path.joinpath(VALID_DIR, "activity", "emb22.csv")
    activity = read_from_csv(csv_path)

    trace = Trace("emb22", activity, config)
    assert trace.peak_idxes is not None
    assert trace.peak_times is not None
    assert trace.peak_intervals is not None
    assert trace.peak_amplitudes is not None
    assert trace.peak_bounds_indices is not None
    assert trace.peak_bounds_times is not None
    assert trace.peak_durations is not None
    assert trace.peak_rise_times is not None
    assert trace.peak_decay_times is not None
    assert trace.peak_aucs is not None
    assert trace.rms is not None


def test_can_create_dsna_trace_and_calculate_dsna(config, activity):
    config.update_params({"exp_params": {"has_dsna": True}})
    trace = Trace("emb1", activity, config)
    assert trace

    dsna_start = trace.get_dsna_start(freq=0.02)
    assert dsna_start


def test_baseline_with_average_n_values():
    arr = [1, 2, 3, 4, 0, 1, 0, 7, 6]
    baseline = Trace.average_n_lowest_window(arr, window_size=5, n_lowest=2)

    # the expected values here assume that the padding mode is 'reflect'
    expected = np.array([1.5, 1.5, 0.5, 0.5, 0, 0, 0, 0.5, 0])

    assert len(baseline) == len(arr)

    assert np.allclose(baseline, expected)


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
