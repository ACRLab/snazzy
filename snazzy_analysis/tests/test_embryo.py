from pathlib import Path

import numpy as np
import pytest

from snazzy_analysis import Config, Dataset

VALID_DIR = Path(__file__).parent.joinpath("assets", "data", "20250210")


@pytest.fixture
def emb():
    to_exclude = ["emb3", "emb4"]
    config = Config(VALID_DIR)
    config.update_params({"exp_params": {"to_exclude": to_exclude}})
    dataset = Dataset(VALID_DIR, config)
    return dataset.get_embryo("emb1")


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


def test_can_estimate_time_from_dev_time(emb):
    dev_time = emb.developmental_time()

    time_from_dt = emb.get_time_from_DT(dev_time)

    assert type(time_from_dt) == np.ndarray

    # for a single dev_time value, the function returns a single timepoint:
    single_timepoint_from_dt = emb.get_time_from_DT(2)

    assert type(single_timepoint_from_dt) == float
