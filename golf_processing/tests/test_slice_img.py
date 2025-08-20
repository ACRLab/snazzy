from golf_processing import slice_img


def test_add_padding_clips_to_shape():
    """Adding padding should not result in negative or OutOfBounds indices."""
    boundaries = [2, 200, 5, 490]
    shape = [300, 500]
    expected_boundaries = [0, 210, 0, 500]

    actual = slice_img.add_padding(boundaries, shape, pad=20)
    assert actual == expected_boundaries


def test_add_padding_when_resulting_bbox_inside():
    boundaries = [50, 150, 40, 130]
    shape = [300, 500]
    expected_boundaries = [40, 160, 30, 140]

    actual = slice_img.add_padding(boundaries, shape, pad=20)
    assert actual == expected_boundaries
