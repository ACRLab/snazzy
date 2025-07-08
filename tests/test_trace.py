import numpy as np
import pytest

from pasna_analysis import Trace


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
