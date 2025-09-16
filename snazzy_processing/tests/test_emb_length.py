from pathlib import Path

import numpy as np
import pytest
from tifffile import imread


from snazzy_processing import full_embryo_length

BASE_DIR = Path(__file__).parent.joinpath("images")
EMB_PATH = BASE_DIR.joinpath("embryo_movies", "emb1-ch2.tif")


@pytest.fixture
def avg_emb_img():
    return imread(BASE_DIR / "average-100-frames-ch2-embryo.tif")


def test_binarize_does_not_change_input_image(avg_emb_img):
    original_img = avg_emb_img.copy()

    _ = full_embryo_length.binarize(avg_emb_img)

    assert np.array_equal(original_img, avg_emb_img)


def test_binarize_low_exposure_does_not_change_input_image(avg_emb_img):
    original_img = avg_emb_img.copy()

    _ = full_embryo_length.binarize_low_embryo_background(avg_emb_img)

    assert np.array_equal(original_img, avg_emb_img)


def test_read_image_returns_2d_image():
    processed_img = full_embryo_length.read_and_preprocess_image(EMB_PATH)

    assert processed_img.ndim == 2


def test_can_measure_img():
    emb_len = full_embryo_length.measure(EMB_PATH)

    assert emb_len > 0


def test_can_measure_img_low_exposure_overestimates_if_not_needed():
    """If the background has lower signal than the embryo, which is the regular case,
    using the `low_non_VNC=True` will overestimate the length and shoud not be used."""
    emb_len = full_embryo_length.measure(EMB_PATH)
    emb_len_over = full_embryo_length.measure(EMB_PATH, low_non_VNC=True)

    assert emb_len_over > emb_len
