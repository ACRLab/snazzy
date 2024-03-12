import numpy as np
from pasnascope import roi


def test_works_with_binary_image():
    '''Assumes opening is done with an `octagon(3,3)` as strucural element.'''
    img = np.load('./tests/images/rectangle.npy')
    opened_img = np.load(
        './tests/images/rectangle-opened.npy').astype(np.bool_)
    expected_roi = ~opened_img
    img_roi = roi.get_single_roi(img)

    assert np.array_equal(img_roi, expected_roi)


def test_returns_none_for_empty_images():
    img = np.zeros((50, 50))
    img_roi = roi.get_single_roi(img)

    assert img_roi is None


def test_selects_largest_bin_region():
    img = np.load('./tests/images/two-regions.npy')
    img_roi = roi.get_single_roi(img)
    expected_roi = np.load('./tests/images/two-regions-roi.npy')

    assert np.array_equal(img_roi, expected_roi)
