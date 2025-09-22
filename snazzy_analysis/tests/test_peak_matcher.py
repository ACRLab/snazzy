from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from snazzy_analysis.gui import PeakMatcher

BASE_DIR = Path(__file__).parent.joinpath("images")


@pytest.fixture
def trace():
    sig = np.zeros(shape=(60,))
    sig[10] = 1.2
    sig[20] = 1.2
    sig[11:15] = 1
    sig[50:53] = 1

    mock_obj = Mock()
    type(mock_obj).dff = property(lambda self: sig)
    type(mock_obj).peak_idxes = property(lambda self: np.array([10, 50]))
    return mock_obj


def test_add_peak_on_position_when_peak_is_clicked(trace):
    click_pos = 10

    pm = PeakMatcher()
    new_peak, new_arr, to_remove = pm.add_peak(click_pos, trace, [], 2)

    assert new_peak == click_pos
    assert np.array_equal(new_arr, trace.peak_idxes)
    assert to_remove is None


def test_add_peak_on_local_peak_when_clicked_near_it(trace):
    click_pos = 9
    # there's a peak at index 10, that point is inside wlen and will be selected
    expected_pos = trace.peak_idxes[0]

    pm = PeakMatcher()
    new_peak, new_arr, to_remove = pm.add_peak(click_pos, trace, [], 2)

    assert new_peak == expected_pos
    assert np.array_equal(new_arr, trace.peak_idxes)
    assert to_remove is None


def test_add_peak_udpates_to_remove_list(trace):
    click_pos = 20
    # there's a removed peak at index 20, a click there reverts the to_remove index
    expected_pos = 20
    to_remove = [20]

    pm = PeakMatcher()
    new_peak, new_arr, to_remove = pm.add_peak(click_pos, trace, to_remove, 2)

    assert new_peak == expected_pos
    assert np.array_equal(new_arr, np.array([10, 20, 50]))
    assert to_remove == []


def test_remove_peak_has_no_effect_when_clicked_outside_peak_range(trace):
    click_pos = 0
    # there are no peaks closeby, the peak index array should be the same

    pm = PeakMatcher()
    removed, new_arr, to_add, filtered_peak_widths = pm.remove_peak(
        click_pos, trace, [], {}
    )

    assert removed == []
    assert np.array_equal(new_arr, trace.peak_idxes)
    assert to_add is None
    assert filtered_peak_widths == {}


def test_remove_peak_when_clicked_close_to_peak(trace):
    click_pos = 9

    pm = PeakMatcher()
    removed, new_arr, to_add, filtered_peak_widths = pm.remove_peak(
        click_pos, trace, [], {}
    )

    assert removed == [10]
    assert not np.array_equal(new_arr, trace.peak_idxes)
    assert new_arr.size == trace.peak_idxes.size - 1
    assert to_add is None
    assert filtered_peak_widths == {}


def test_remove_peak_also_updates_manually_added_peaks(trace):
    click_pos = 11
    manual_add = [16, 8, 44]

    pm = PeakMatcher()
    removed, new_arr, to_add, filtered_peak_widths = pm.remove_peak(
        click_pos, trace, manual_add, {}, wlen=5
    )

    assert removed == [10]
    assert to_add == [44]
    assert filtered_peak_widths == {}


def test_remove_peak_also_updates_peak_widths(trace):
    click_pos = 11
    peak_widths = {"10": [9, 13], "50": [40, 52]}

    pm = PeakMatcher()
    removed, new_arr, to_add, filtered_peak_widths = pm.remove_peak(
        click_pos, trace, [], peak_widths
    )

    assert removed == [10]
    assert not np.array_equal(new_arr, trace.peak_idxes)
    assert new_arr.size == trace.peak_idxes.size - 1
    assert to_add is None
    assert "10" not in filtered_peak_widths
    assert filtered_peak_widths


def test_remove_peak_removes_all_peaks_within_range(trace):
    click_pos = 11
    peak_widths = {"10": [9, 13], "50": [40, 52]}

    pm = PeakMatcher()
    removed, new_arr, to_add, filtered_peak_widths = pm.remove_peak(
        click_pos, trace, [], peak_widths
    )

    assert removed == [10]
    assert not np.array_equal(new_arr, trace.peak_idxes)
    assert new_arr.size == trace.peak_idxes.size - 1
    assert to_add is None
    assert "10" not in filtered_peak_widths
    assert filtered_peak_widths
