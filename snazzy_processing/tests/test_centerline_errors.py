from pathlib import Path

import numpy as np
import pytest

from snazzy_processing import centerline_errors


@pytest.fixture
def mocked_data():
    """Generate a mocked pair of measured and annotated data."""
    rng = np.random.default_rng(1705)

    measured = {}
    annotated = {}

    for i in range(2):
        key = f"emb{i+1}"
        arr = rng.normal(loc=0, scale=1, size=100)

        measured[key] = arr
        if i == 0:
            annotated[key] = arr.copy()
        else:
            annotated[key] = 1.1 * arr + rng.normal(loc=0, scale=0.1, size=100)

    return measured, annotated


def test_matching_embryo_pairs_maps_strs_to_paths():
    embryos = [Path(f) for f in ["/fake/emb1-ch2.tif", "/fake/emb2-ch2.tif"]]
    annotated = [Path(f) for f in ["/fake/emb1-ch2.csv", "/fake/emb2-ch2.csv"]]
    res = centerline_errors.get_matching_embryos(embryos, annotated)

    ann_file_name, emb_path = next(iter(res.items()))
    assert isinstance(ann_file_name, str)
    assert isinstance(emb_path, str)


def test_maps_to_same_name_when_LUT_is_None():
    embryos = [Path(f) for f in ["/fake/emb1-ch2.tif", "/fake/emb2-ch2.tif"]]
    annotated = [Path(f) for f in ["/fake/emb1-ch2.csv", "/fake/emb2-ch2.csv"]]
    expected = {"emb1-ch2.tif": "emb1-ch2.csv", "emb2-ch2.tif": "emb2-ch2.csv"}
    actual = centerline_errors.get_matching_embryos(embryos, annotated)

    assert expected == actual


def test_maps_to_emb_when_given_LUT():
    LUT = {1: 4, 2: 12}
    embryos = [Path(f) for f in ["/fake/emb1-ch2.tif", "/fake/emb2-ch2.tif"]]
    annotated = [Path(f) for f in ["/fake/emb4-ch2.csv", "/fake/emb12-ch2.csv"]]
    expected = {"emb1-ch2.tif": "emb4-ch2.csv", "emb2-ch2.tif": "emb12-ch2.csv"}
    actual = centerline_errors.get_matching_embryos(embryos, annotated, LUT)

    assert expected == actual


def test_maps_to_emb_when_given_LUT_partial():
    LUT = {1: 4, 2: 12}
    embryos = [Path(f) for f in ["/fake/emb1-ch2.tif", "/fake/emb2-ch2.tif"]]
    annotated = [
        Path(f)
        for f in [
            "/fake/emb1-ch2.csv",
            "/fake/emb2-ch2.csv",
            "/fake/emb12-ch2.csv",
            "/fake/emb4-ch2.csv",
        ]
    ]
    expected = {"emb1-ch2.tif": "emb4-ch2.csv", "emb2-ch2.tif": "emb12-ch2.csv"}
    actual = centerline_errors.get_matching_embryos(embryos, annotated, LUT)

    assert expected == actual


def test_empty_annotated_dir_returns_empty_dict():
    LUT = {1: 4, 2: 12}
    embryos = [Path(f) for f in ["/fake/emb1-ch2.tif", "/fake/emb2-ch2.tif"]]
    annotated = []
    expected = {}
    actual = centerline_errors.get_matching_embryos(embryos, annotated, LUT)

    assert expected == actual


def test_empty_annotated_dir_returns_empty_dict_when_LUT_None():
    embryos = [Path(f) for f in ["/fake/emb1-ch2.tif", "/fake/emb2-ch2.tif"]]
    annotated = []
    expected = {}
    actual = centerline_errors.get_matching_embryos(embryos, annotated)

    assert expected == actual


def test_can_evaluate_centerline_metrics(mocked_data):

    measured, annotated = mocked_data

    cle_errors = centerline_errors.evaluate_CLE_global(measured, annotated)

    emb1_perc_error, emb1_max_error = cle_errors["emb1"]
    emb2_perc_error, emb2_max_error = cle_errors["emb2"]

    assert emb1_perc_error == 0
    assert emb1_max_error == 0

    assert emb2_perc_error != 0
    assert emb2_max_error != 0


def test_can_evaluate_point_wise_errors(mocked_data):
    measured, annotated = mocked_data

    same_values = "emb1"

    zeroed_pointwise_errors = centerline_errors.point_wise_err(
        measured[same_values], annotated[same_values]
    )

    assert sum(zeroed_pointwise_errors) == 0
    assert zeroed_pointwise_errors.size == measured[same_values].size

    diff_values = "emb2"

    nonzero_pointwise_errors = centerline_errors.point_wise_err(
        measured[diff_values], annotated[diff_values]
    )

    assert sum(nonzero_pointwise_errors) > 0
    assert nonzero_pointwise_errors.size == measured[diff_values].size
