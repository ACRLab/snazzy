import numpy as np

import pytest

from pasna_analysis import TracePhases


def test_finds_right_index_in_dist_matrix():
    dist_matrix = np.random.random((10, 10))
    thres = 1
    k = 5
    dist_matrix[k, :] += 1

    assert TracePhases.apply_threshold_to_matrix(dist_matrix, thres) == k - 1


def test_finds_right_index_in_dist_matrix_reverse():
    dist_matrix = np.random.random((10, 10))
    thres = 1
    k = 5
    dist_matrix[k, :] += 1

    assert TracePhases.apply_threshold_to_matrix(dist_matrix, thres, reverse=True) == k


def test_finds_right_index_if_dist_matrix_size_1():
    dist_matrix = np.random.random(1)
    thres = 0
    expected_index = 0

    assert TracePhases.apply_threshold_to_matrix(dist_matrix, thres) == expected_index


def test_apply_thres_raises_if_empty_dist_matrix():
    dist_matrix = np.array([])
    thres = 0

    with pytest.raises(ValueError):
        TracePhases.apply_threshold_to_matrix(dist_matrix, thres)
