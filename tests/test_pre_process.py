from pasnascope import pre_process
import numpy as np
import pytest


def test_calculate_rows_to_trim_no_trim():
    shape = (10, 5)
    factors = (2, 5)
    expected_size = (10, 5)
    assert pre_process.get_matching_size(shape, factors) == expected_size


def test_calculate_rows_to_trim_ones():
    shape = (10, 5)
    factors = (1, 1)
    expected_size = (10, 5)
    assert pre_process.get_matching_size(shape, factors) == expected_size


def test_calculate_rows_to_trim_with_trim():
    """Tests the function with a matrix shape requiring trimming."""

    expected_size = (10, 5)
    shape = (9, 7)
    factors = (3, 3)
    expected_size = (9, 6)
    assert pre_process.get_matching_size(shape, factors) == expected_size


def test_raises_value_error_if_not_3D():
    with pytest.raises(ValueError) as exc:
        img = np.ones((20, 20))
        pre_process.pre_process(img, (1, 2, 2))
    assert str(exc.value) == 'img must have 3 dimensions.'
