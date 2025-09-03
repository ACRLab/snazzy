from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from snazzy_processing import centerline

BASE_DIR = Path(__file__).parent.joinpath("images")


@pytest.fixture
def emb_img():
    return np.load(BASE_DIR / "embryo.npy")[0]


@pytest.fixture
def emb_img_3d():
    return np.load(BASE_DIR / "embryo.npy")


def test_binarize_does_not_change_input_image(emb_img):
    before = emb_img.copy()
    _ = centerline.binarize(emb_img)

    assert np.array_equal(before, emb_img)


def test_binarize_raises_if_not_2D_image(emb_img_3d):
    with pytest.raises(ValueError):
        centerline.binarize(emb_img_3d)


def test_binarize_raises_if_threshold_method_not_supported(emb_img):
    with pytest.raises(ValueError):
        centerline.binarize(emb_img, threshold_method="unsupported_method")


def test_binarize_regular_otsu(emb_img):
    bin_img = centerline.binarize(emb_img, threshold_method="otsu")

    assert ((bin_img == 0) | (bin_img == 1)).all()


def test_cannot_measure_centerline_on_3D_image(emb_img_3d):
    with pytest.raises(ValueError):
        centerline.centerline_dist(emb_img_3d)


def test_cannot_measure_centerline_when_ransac_does_not_converge(emb_img):
    # arrange three points as a triangle, so RANSAC won't converge
    # because it won't reach the stop_score
    radius = 50
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    circle_points = np.column_stack((x, y))

    with patch(
        "snazzy_processing.centerline.get_DT_maxima", return_value=circle_points
    ):
        calculated_dist = centerline.centerline_dist(emb_img)
        expected_dist = 0

        assert calculated_dist == expected_dist


@pytest.mark.parametrize(
    "DT_points",
    [np.array([]), np.array([[1, 1]]), np.array([0, 0]), np.array([[1, 1], [1, 2]])],
)
def test_cannot_measure_centerline_without_DT_maxima(emb_img, DT_points):
    with patch("snazzy_processing.centerline.get_DT_maxima", return_value=DT_points):
        calculated_dist = centerline.centerline_dist(emb_img)

        assert calculated_dist == 0


def test_creates_centerline_mask():
    num_points = 20
    y_val = 5
    # generate a horizontal line, as a nparray of [x, y] points
    x = np.arange(0, num_points, step=1)
    y = np.ones(shape=(num_points)) * y_val
    points = np.column_stack((y, x))

    # DT maxima will match the line itself
    model = centerline.apply_ransac(points)

    img = np.ones(shape=(20, 20))

    mask = centerline.centerline_mask(img.shape, model.predict)

    # for a horizontal line, only values at y_val will be preserved
    # the rest is set to zero by the mask
    line_values = np.sum(mask[y_val, :])
    assert line_values == num_points
    assert np.sum(mask) - line_values == 0


def test_measure_centerline():
    pixel_width = 1.6
    num_points = 20
    # generate a 2D nparray representing an image with a horizontal line
    img = np.zeros(shape=(30, 30))
    img[5, 5 : 5 + num_points] = 1

    dist = centerline.measure_length(img, pixel_width)

    assert dist == (num_points - 1) * pixel_width


def test_measure_centerline_raises_if_not_2D(emb_img_3d):
    with pytest.raises(ValueError):
        centerline.measure_length(emb_img_3d, pixel_width=1.6)
