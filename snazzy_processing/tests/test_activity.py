from pathlib import Path

import numpy as np
import pytest
from tifffile import imread

from snazzy_processing import activity

BASE_DIR = Path(__file__).parent.joinpath("images")


@pytest.fixture
def emb_img():
    return np.load(BASE_DIR / "embryo.npy")


@pytest.fixture
def mask_2d():
    return np.load(BASE_DIR / "mask-2d-embryo.npy")


@pytest.fixture
def mask_3d():
    return np.load(BASE_DIR / "mask-3d-embryo.npy")


def test_can_use_single_mask(emb_img, mask_2d):
    masked_img = activity.apply_mask(emb_img, mask_2d)

    assert masked_img is not None
    assert masked_img.shape == emb_img.shape


def test_cannot_use_mask_with_wrong_shapes(emb_img):
    # reduce the mask by 5 columns, to mismatch the shapes
    mask = np.ones((emb_img.shape[1], emb_img.shape[2] - 5), dtype=np.bool_)

    with pytest.raises(ValueError):
        activity.apply_mask(emb_img, mask)


def test_can_apply_3D_mask_with_same_dims(emb_img, mask_3d):
    masked_img = activity.apply_mask(emb_img, mask_3d)

    assert masked_img is not None
    assert masked_img.shape == emb_img.shape


def test_can_use_downsampled_3D_mask(emb_img, mask_3d):
    mask_len = 5
    mask = mask_3d[:mask_len]

    masked_img = activity.apply_mask(emb_img, mask)

    assert masked_img is not None
    assert masked_img.shape == emb_img.shape


def test_can_apply_3D_mask_when_not_multiple_shape(emb_img, mask_3d):
    mask_len = emb_img.shape[0] // 5 + 2
    mask = mask_3d[:mask_len]

    masked_img = activity.apply_mask(emb_img, mask)

    assert masked_img is not None
    assert masked_img.shape == emb_img.shape


def test_can_calculate_channel_signals():
    emb_ch1 = imread(BASE_DIR / "embryo_movies" / "emb1-ch1.tif")
    emb_ch2 = imread(BASE_DIR / "embryo_movies" / "emb1-ch2.tif")

    active_ch = activity.get_activity(emb_ch1)
    struct_ch = activity.get_activity(emb_ch2)

    assert active_ch is not None
    assert struct_ch is not None
    assert active_ch.shape == struct_ch.shape
