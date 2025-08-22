from pathlib import Path

import numpy as np
import pytest

from golf_processing import roi

EMBRYO_IMG = Path(__file__).parent.joinpath("images", "embryo.npy")


@pytest.fixture
def single_roi_img():
    """A 2D nparray with a rectangular region representing an embryo."""
    img = np.random.randint(0, 20, (50, 50))
    img[10:40, 10:45] = np.random.randint(155, 175, (30, 35))

    return img


@pytest.fixture
def img_3D():
    """A 3D nparray with a rectangular region representing an embryo.

    Represents a sequence of images over time."""
    img = np.random.randint(0, 20, (50, 50, 50))
    img[:, 10:40, 10:45] = np.random.randint(155, 175, (30, 35))

    return img


def test_roi_shape_matches_img_shape_without_downsamping():
    img = np.load(EMBRYO_IMG)
    img_roi = roi.get_roi(img, window=1)

    assert img_roi.shape == img.shape


def test_roi_shape_using_default_downsampling():
    img = np.load(EMBRYO_IMG)
    downsampling_factor = 10
    img_roi = roi.get_roi(img)

    assert img_roi.shape == (img.shape[0] // downsampling_factor, *img.shape[1:])


def test_roi_shape_for_given_downsampling():
    img = np.load(EMBRYO_IMG)
    downsampling_factor = 10
    img_roi = roi.get_roi(img, window=downsampling_factor)

    assert img_roi.shape == (img.shape[0] // downsampling_factor, *img.shape[1:])


def test_when_img_not_2D_then_get_single_roi_raises():
    img = np.random.randint(0, 255, 50)

    # img.shape == (50,) so it should raise
    with pytest.raises(ValueError):
        roi.get_single_roi(img)


def test_when_img_not_3D_then_get_rois_raises(single_roi_img):
    with pytest.raises(ValueError):
        roi.get_roi(single_roi_img)


def test_roi_covers_small_holes(single_roi_img):
    hole_mask = np.zeros((50, 50), dtype=np.bool_)
    hole_mask[15:20, 20:25] = True

    # create a hole in the single_roi_img
    single_roi_img[hole_mask] = np.random.randint(0, 20, 25)

    img_roi = roi.get_single_roi(single_roi_img)

    assert img_roi is not None
    # points inside the ROI are marked as False
    assert np.sum(img_roi[hole_mask]) == 0


def test_can_find_contours(img_3D):
    window = 10
    contour = roi.get_contours(img_3D, window=window)

    assert contour is not None
    assert len(contour) == img_3D.shape[0] // window
