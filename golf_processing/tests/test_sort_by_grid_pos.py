import random

import pytest

from golf_processing import slice_img

RANDOM_SEED = 19680801


@pytest.fixture(scope="function")
def grid_3x3_bboxes():
    return [
        [100, 200, 100, 200],
        [400, 500, 100, 200],
        [700, 800, 100, 200],
        [100, 200, 400, 500],
        [400, 500, 400, 500],
        [700, 800, 400, 500],
        [100, 200, 700, 800],
        [400, 500, 700, 800],
        [700, 800, 700, 800],
    ]


def shuffle_arr(arr):
    random.seed(RANDOM_SEED)
    random.shuffle(arr)


def test_sorts_a_full_grid(grid_3x3_bboxes):
    expected = {i: b for i, b in enumerate(grid_3x3_bboxes, 1)}

    shuffle_arr(grid_3x3_bboxes)

    calc_boundaries = slice_img.sort_by_grid_pos(grid_3x3_bboxes, 3)

    assert expected == calc_boundaries


def test_sorts_a_grid_with_one_missing_embryo(grid_3x3_bboxes):
    """Removed embryo that would take pos 9."""
    bboxes = grid_3x3_bboxes[:-1]
    expected = {i: b for i, b in enumerate(bboxes, 1)}

    shuffle_arr(bboxes)

    calc_boundaries = slice_img.sort_by_grid_pos(bboxes, 3)
    assert expected == calc_boundaries


def test_sorts_a_grid_with_several_missing_embryos(grid_3x3_bboxes):
    """Removed embryos at index 0, 5, 6, and 7."""
    keeping_idxs = [1, 2, 3, 4, 7]
    bboxes = [grid_3x3_bboxes[i] for i in keeping_idxs]

    expected = {i: bbox for i, bbox in enumerate(bboxes, start=1)}

    shuffle_arr(bboxes)

    calc_boundaries = slice_img.sort_by_grid_pos(bboxes, 3)

    assert expected == calc_boundaries


def test_sorts_a_grid_with_empty_column(grid_3x3_bboxes):
    """Removed entire middle column (pos 4, 5, 6)."""
    pos = [0, 1, 2, 6, 7, 8]
    bboxes = [grid_3x3_bboxes[p] for p in pos]

    expected = {i: bbox for i, bbox in enumerate(bboxes, start=1)}

    shuffle_arr(bboxes)

    calc_boundaries = slice_img.sort_by_grid_pos(bboxes, 2)

    assert expected == calc_boundaries
