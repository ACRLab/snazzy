from pathlib import Path

from snazzy_processing import slice_img

FIRST_FRAMES = Path(__file__).parent.joinpath("images", "first_frames.tif")


def test_can_convert_from_bbox_to_rect_coords():
    rect_coords = [100, 100, 50, 80]
    x, y, h, w = rect_coords
    bbox_coords = [x, x + w, y, y + h]

    calculated_rect_coords = slice_img.boundary_to_rect_coords(bbox_coords)

    assert calculated_rect_coords == rect_coords


def test_can_mark_neighbors():
    coords = slice_img.calculate_slice_coordinates(FIRST_FRAMES)
    number_of_embryos = 4
    assert len(coords) == number_of_embryos


def test_increase_bbox_does_not_mutate_coords():
    coords = {1: [30, 80, 30, 80], 2: [100, 300, 50, 250]}
    orig_coords = coords.copy()
    shape = (300, 300)

    expanded_bbox = slice_img.increase_bbox(coords, 10, 10, shape)

    assert expanded_bbox is not coords
    assert coords == orig_coords


def test_increase_bbox_clips_to_shape():
    """Adding padding should not result in negative or OutOfBounds indices."""
    coords = {1: [2, 200, 5, 490]}
    expected_coords = {1: [0, 210, 0, 500]}
    shape = (300, 500)

    actual = slice_img.increase_bbox(coords, w=20, h=20, shape=shape)
    assert actual == expected_coords


def test_increase_bbox_when_resulting_bbox_inside():
    coords = {1: [10, 200, 50, 490]}
    expected_coords = {1: [0, 210, 40, 500]}
    shape = (600, 600)

    actual = slice_img.increase_bbox(coords, w=20, h=20, shape=shape)
    assert actual == expected_coords
