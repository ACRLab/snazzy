import numpy as np
import pytest

from pasna_analysis import TracePhases


def test_finds_right_index_in_dist_matrix():
    dist_matrix = np.random.random((10, 10))
    thres = 1
    k = 5
    dist_matrix[k, :] += 1

    actual_index = TracePhases.segment_distance_matrix_forward(dist_matrix, thres)
    expected_index = k - 1

    assert actual_index == expected_index


def test_finds_right_index_in_last_row():
    dist_matrix = np.random.random((10, 10))
    thres = 1
    k = 9
    dist_matrix[k, :] += 1

    actual_index = TracePhases.segment_distance_matrix_forward(dist_matrix, thres)
    expected_index = k - 1

    assert actual_index == expected_index


def test_when_all_dists_below_thres_return_last_index():
    dist_matrix = np.random.random((10, 10))
    thres = 2

    actual_index = TracePhases.segment_distance_matrix_forward(dist_matrix, thres)
    expected_index = len(dist_matrix) - 1

    assert actual_index == expected_index


def test_when_reversing_and_all_dists_below_thres_return_last_index():
    dist_matrix = np.random.random((10, 10))
    thres = 2

    actual_index = TracePhases.segment_distance_matrix_reverse(dist_matrix, thres)
    expected_index = 9

    assert actual_index == expected_index


def test_finds_right_index_in_dist_matrix_reverse():
    dist_matrix = np.random.random((10, 10))
    thres = 1
    k = 5
    dist_matrix[k, :] += 1

    actual_index = TracePhases.segment_distance_matrix_reverse(dist_matrix, thres)
    expected_index = k + 1

    assert actual_index == expected_index


def test_finds_right_index_if_dist_matrix_size_1():
    dist_matrix = np.random.random(1)
    thres = 0

    actual_index = TracePhases.segment_distance_matrix_forward(dist_matrix, thres)
    expected_index = 0

    assert actual_index == expected_index


def test_apply_thres_raises_if_empty_dist_matrix():
    dist_matrix = np.array([])
    thres = 0

    with pytest.raises(ValueError):
        TracePhases.segment_distance_matrix_forward(dist_matrix, thres)
