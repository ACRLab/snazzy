from pasnascope import slice_img
import random


def test_sorts_a_full_grid():
    bboxes = [
        [100, 200, 100, 200], [400, 500, 100, 200], [700, 800, 100, 200],
        [100, 200, 400, 500], [400, 500, 400, 500], [700, 800, 400, 500],
        [100, 200, 700, 800], [400, 500, 700, 800], [700, 800, 700, 800]]
    expected = {i: b for i, b in enumerate(bboxes, 1)}
    random.seed(19680801)
    random.shuffle(bboxes)
    calc_boundaries = slice_img.sort_by_grid_pos(bboxes, 3)

    assert expected == calc_boundaries


def test_sorts_a_grid_with_one_missing_embryo():
    '''Removed embryo that would take pos 9.'''
    bboxes = [
        [100, 200, 100, 200], [400, 500, 100, 200], [700, 800, 100, 200],
        [100, 200, 400, 500], [400, 500, 400, 500], [700, 800, 400, 500],
        [100, 200, 700, 800], [400, 500, 700, 800]]
    expected = {i: b for i, b in enumerate(bboxes, 1)}
    random.seed(19680801)
    random.shuffle(bboxes)
    calc_boundaries = slice_img.sort_by_grid_pos(bboxes, 3)
    assert expected == calc_boundaries


def test_sorts_a_grid_with_several_missing_embryos():
    '''Removed embryos that would take pos 1, 6, 7, and 9.'''
    bboxes = [
        [400, 500, 100, 200], [700, 800, 100, 200], [100, 200, 400, 500],
        [400, 500, 400, 500], [400, 500, 700, 800]]
    expected = {1: [400, 500, 100, 200], 2: [700, 800, 100, 200], 3: [
        100, 200, 400, 500], 4: [400, 500, 400, 500],  5: [400, 500, 700, 800], }
    random.seed(19680801)
    random.shuffle(bboxes)
    calc_boundaries = slice_img.sort_by_grid_pos(bboxes, 3)

    assert expected == calc_boundaries


def test_sorts_a_grid_with_empty_column():
    '''Removed entire middle column (pos 4, 5, 6).'''
    bboxes = [
        [100, 200, 100, 200], [400, 500, 100, 200], [700, 800, 100, 200],
        [100, 200, 700, 800], [400, 500, 700, 800], [700, 800, 700, 800]]
    expected = {i: b for i, b in enumerate(bboxes, 1)}
    random.seed(19680801)
    random.shuffle(bboxes)
    calc_boundaries = slice_img.sort_by_grid_pos(bboxes, 2)

    assert expected == calc_boundaries
