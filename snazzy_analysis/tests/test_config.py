import json
from pathlib import Path
from snazzy_analysis.gui import Model

import pytest

from snazzy_analysis import Config

DATASET_1 = Path("./tests/assets/data/20250210")
DATASET_2 = Path("./tests/assets/data/20250220")
GROUP_NAME = "WT"


@pytest.fixture(scope="module")
def config():
    config = Config(DATASET_1, rel_root_path="tests")

    return config


def test_use_default_parameters_if_pd_params_does_not_exist(config):
    assert config.data == config.default_params


def test_keep_track_of_exp_path(config):
    assert "exp_path" in config.data
    assert config.data["exp_path"] == "tests/assets/data/20250210"


def test_can_persist_params(config):
    config.save_params()

    assert config.config_path.exists()


def test_can_update_params(config):
    expected_fpt = 20
    new_data = {"exp_params": {"first_peak_threshold": expected_fpt}}

    config.update_params(new_data)

    exp_params = config.get_exp_params()
    assert exp_params.get("first_peak_threshold", None) == expected_fpt


def test_raises_when_invalid_update_params(config):
    invalid_data = {"key-does-not-exist": None}

    with pytest.raises(KeyError):
        config.update_params(invalid_data)


def test_can_save_manual_data_for_new_embryo(config):
    emb_data = dict(
        emb_name="emb11",
        wlen=10,
        added_peaks=[100, 150, 200],
        removed_peaks=[50],
        manual_widths={"100": [80, 125]},
        manual_trim_idx=1500,
        manual_phase1_end=500,
    )

    assert emb_data["emb_name"] not in config.data["embryos"]

    config.save_manual_peak_data(**emb_data)

    assert emb_data["emb_name"] in config.data["embryos"]

    corrected_data = config.get_corrected_peaks(emb_data["emb_name"])

    assert corrected_data and corrected_data["manual_peaks"] == emb_data["added_peaks"]


def test_can_parse_empty_pd_params_file():
    pd_params_path = DATASET_2.joinpath("peak_detection_params.json")

    with open(pd_params_path, "w") as f:
        json.dump({}, f)

    assert Config(DATASET_2)


def test_can_parse_pd_params_file_with_extra_keys():
    pd_params_path = DATASET_2.joinpath("peak_detection_params.json")

    with open(pd_params_path, "w") as f:
        json.dump({"unused_key": {"nested_key": 12}}, f)

    assert Config(DATASET_2)


def test_can_parse_pd_params_with_missing_keys():
    pd_params_path = DATASET_2.joinpath("peak_detection_params.json")
    expected_fpt = 12

    with open(pd_params_path, "w") as f:
        json.dump({"exp_params": {"first_peak_threshold": expected_fpt}}, f)

    config = Config(DATASET_2)

    exp_params = config.get_exp_params()
    assert exp_params.get("first_peak_threshold") == expected_fpt
    assert (
        exp_params.get("has_transients")
        == config.default_params["exp_params"]["has_transients"]
    )


def test_invalid_pd_params_results_in_using_default_params():

    pd_params_path = DATASET_2.joinpath("peak_detection_params.json")

    with open(pd_params_path, "w") as f:
        json.dump({"exp_params": {"first_peak_threshold": "invalid!!"}}, f)

    config = Config(DATASET_2)

    assert config.data == config.default_params
