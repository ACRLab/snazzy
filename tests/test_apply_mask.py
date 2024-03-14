import numpy as np
import pytest
from pasnascope import activity

base_dir = './tests/images/'


def test_can_use_single_mask():
    img = np.load(base_dir + 'embryo.npy')
    mask = np.load(base_dir + 'mask-2d-embryo.npy')

    try:
        activity.apply_mask(img, mask)
    except ValueError as e:
        pytest.fail(f"Test failed with: {e}")


def test_cannot_use_mask_with_different_dims():
    img = np.load(base_dir + 'embryo.npy')
    # reduce the mask by 5 columns, to mismatch the shapes
    mask = np.ones((img.shape[1], img.shape[2]-5), dtype=np.bool_)

    with pytest.raises(ValueError) as err:
        activity.apply_mask(img, mask)

    assert err.type is ValueError


def test_can_use_3D_mask_with_same_dims():
    img = np.load(base_dir + 'embryo.npy')
    mask = np.load(base_dir + 'mask-3d-embryo.npy')

    try:
        activity.apply_mask(img, mask)
    except Exception as e:
        pytest.fail(f"Test failed with: {e}")


def test_can_use_downsampled_3D_mask():
    img = np.load(base_dir + 'embryo.npy')
    mask = np.load(base_dir + 'mask-3d-downs-embryo.npy')

    try:
        activity.apply_mask(img, mask)
    except Exception as e:
        pytest.fail(f"Test failed with: {e}")
