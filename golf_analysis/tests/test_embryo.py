from pathlib import Path

import numpy as np
import pytest

from golf_analysis import Config, Experiment

VALID_DIR = Path("./tests/assets/data/20250210")


@pytest.fixture
def emb():
    to_exclude = [3, 4]
    config = Config(VALID_DIR)
    config.update_params({"exp_params": {"to_exclude": [to_exclude]}})
    exp = Experiment(VALID_DIR, config)
    return exp.get_embryo("emb1")


@pytest.fixture
def bin_params():
    # number of bins, first bin, bin width
    return (40, 1, 0.2)


def test_time_bins_represent_bin_lower_bound(emb, bin_params):
    n_bins, first_bin, bin_width = bin_params
    last_bin = first_bin + n_bins * bin_width

    bins = np.arange(first_bin, last_bin, bin_width)

    _, idx_offset = emb.get_time_bins(bins)

    initial_dt = emb.lin_developmental_time[0]
    assert bins[idx_offset] <= initial_dt


def test_time_bins_have_correct_idx_offset(emb, bin_params):
    n_bins, first_bin, bin_width = bin_params
    last_bin = first_bin + n_bins * bin_width

    bins = np.arange(first_bin, last_bin, bin_width)

    _, idx_offset = emb.get_time_bins(bins)

    initial_dt = emb.lin_developmental_time[0]
    assert bins[idx_offset] <= initial_dt < bins[idx_offset + 1]


def test_time_bins_ignores_extra_bins(emb, bin_params):
    n_bins, first_bin, bin_width = bin_params
    last_bin = first_bin + n_bins * bin_width

    bins = np.arange(first_bin, last_bin, bin_width)

    # 'emb1' dev time starts at 1.9863 and ends at 2.7193
    # it should have bins that match the dev time range from 1.8 to 2.6
    time_bins, _ = emb.get_time_bins(bins)

    assert len(time_bins) < len(bins)
    assert len(time_bins) == 6
