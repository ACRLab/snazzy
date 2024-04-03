from pasnascope import pre_process


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
