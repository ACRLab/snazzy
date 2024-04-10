from pasnascope import slice_img
import random


def test_sorts_a_full_grid():
    sorted_boundaries = [
        [100, 200, 100, 200], [400, 500, 100, 200], [700, 800, 100, 200],
        [100, 200, 400, 500], [400, 500, 400, 500], [700, 800, 400, 500],
        [100, 200, 700, 800], [400, 500, 700, 800], [700, 800, 700, 800]]
    boundaries = sorted_boundaries[:]
    random.seed(19680801)
    random.shuffle(boundaries)
    calc_boundaries = slice_img.sort_by_grid_pos(boundaries, 3)

    assert sorted_boundaries == calc_boundaries


def test_sorts_a_grid_with_one_missing_embryo():
    '''Removed embryo at row 3 column 3, from the full grid.'''
    sorted_boundaries = [
        [100, 200, 100, 200], [400, 500, 100, 200], [700, 800, 100, 200],
        [100, 200, 400, 500], [400, 500, 400, 500], [700, 800, 400, 500],
        [100, 200, 700, 800], [400, 500, 700, 800]]

    boundaries = sorted_boundaries[:]
    random.seed(19680801)
    random.shuffle(boundaries)
    calc_boundaries = slice_img.sort_by_grid_pos(boundaries, 3)

    assert sorted_boundaries == calc_boundaries


def test_sorts_a_grid_with_several_missing_embryos():
    '''Removed embryos 11, 23, 31, 33, first digit row, second digit column.'''
    sorted_boundaries = [
        [400, 500, 100, 200], [700, 800, 100, 200], [100, 200, 400, 500],
        [400, 500, 400, 500], [400, 500, 700, 800]]

    boundaries = sorted_boundaries[:]
    random.seed(19680801)
    random.shuffle(boundaries)
    calc_boundaries = slice_img.sort_by_grid_pos(boundaries, 3)

    assert sorted_boundaries == calc_boundaries


def test_sorts_a_grid_with_empty_column():
    '''Removed entire middle column from a 3x3 grid.'''
    sorted_boundaries = [
        [100, 200, 100, 200], [400, 500, 100, 200], [700, 800, 100, 200],
        [100, 200, 700, 800], [400, 500, 700, 800], [700, 800, 700, 800]]

    boundaries = sorted_boundaries[:]
    random.seed(19680801)
    random.shuffle(boundaries)
    calc_boundaries = slice_img.sort_by_grid_pos(boundaries, 3)

    assert sorted_boundaries == calc_boundaries
