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


def test_output_data_includes_time():
    signals_shape = (2, 50, 2)
    signals = np.arange(0, 200).reshape(signals_shape)

    output = activity.add_timepoints(signals, frame_interval=5)

    N, t, _ = signals_shape
    assert output.shape == (N, t, 3)
    # time data is added at index 0, so the rest should be equal
    # the original signals array
    assert np.array_equal(output[:, :, 1:], signals)


def test_output_data_uses_frame_interval():
    signals_shape = (2, 50, 2)
    frame_interval = 4
    signals = np.arange(0, 200).reshape(signals_shape)

    output = activity.add_timepoints(signals, frame_interval)

    assert np.array_equal(np.arange(0, 50), output[0, :, 0] / frame_interval)


def test_can_write_data_when_single_embryo(tmp_path):
    ids = [1]
    signals = np.arange(300).reshape((1, 100, 3))

    activity.export_csv(ids, signals, tmp_path)

    files = [f for f in tmp_path.iterdir() if f.suffix == ".csv"]

    assert len(files) == 1


def test_can_write_data_for_many_embryos(tmp_path):
    n_embs = 5
    ids = [i for i in range(1, n_embs + 1)]
    signals = np.arange(750).reshape((n_embs, 50, 3))

    activity.export_csv(ids, signals, tmp_path)

    files = [f for f in tmp_path.iterdir() if f.suffix == ".csv"]

    assert len(files) == n_embs


def test_export_csv_raises_when_no_embryos(tmp_path):
    # will raise when `activity.add_timepoints` is called
    with pytest.raises(ValueError):
        activity.export_csv([], [], tmp_path)
