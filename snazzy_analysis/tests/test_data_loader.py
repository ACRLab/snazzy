from pathlib import Path

import numpy as np
import pytest

from snazzy_analysis import DataLoader

DATA_DIR = Path(__file__).parent.joinpath("assets", "data")
VALID_DIR = DATA_DIR.joinpath("20250210")
# MISSING_DATA_DIR does not have full_length.csv file.
MISSING_DATA_DIR = DATA_DIR.joinpath("20250101")
SINGLE_EMB_DIR = DATA_DIR.joinpath("20250230")


def test_creates_instance_when_dir_matches_expected_structure():
    dl = DataLoader(VALID_DIR)

    assert dl is not None
    assert dl.path == VALID_DIR


def test_raises_when_missing_files():
    with pytest.raises(ValueError):
        DataLoader(MISSING_DATA_DIR)


def test_raises_when_missing_whole_directory():
    with pytest.raises(ValueError):
        DataLoader(Path("./tests/assets/data/missingDir"))


def test_raises_when_files_dont_match(tmp_path):
    activities_dir = tmp_path / "activity"
    lengths_dir = tmp_path / "lengths"
    activities_dir.mkdir()
    lengths_dir.mkdir()

    (activities_dir / "emb1.csv").touch()
    (lengths_dir / "emb2.csv").touch()
    (tmp_path / "full-length.csv").touch()

    with pytest.raises(ValueError):
        DataLoader(tmp_path)


def test_list_files_correctly():
    dl = DataLoader(VALID_DIR)

    expected_embs = ["emb1.csv", "emb3.csv", "emb4.csv"]

    filepath_pairs = dl.get_data_path_pairs()

    for expected, (act_file, len_file) in zip(expected_embs, filepath_pairs):
        expected_act_file = VALID_DIR / "activity" / expected
        expected_len_file = VALID_DIR / "lengths" / expected
        assert expected_act_file == act_file
        assert expected_len_file == len_file


def test_can_read_csv_files_as_nparrays():
    dl = DataLoader(VALID_DIR)

    filepaths = list(dl.get_data_path_pairs())
    first_len_filepath = filepaths[0][1]

    assert type(dl.load_csv(first_len_filepath)) == np.ndarray


def test_read_full_length_file():
    dl = DataLoader(VALID_DIR)

    full_emb_lengths = dl.load_csv(VALID_DIR.joinpath("full-length.csv"))

    assert type(full_emb_lengths) == np.ndarray
    assert full_emb_lengths.ndim == 2


def test_read_single_row_csv_as_2darray():
    dl = DataLoader(SINGLE_EMB_DIR)

    full_emb_lengths = dl.load_csv(SINGLE_EMB_DIR.joinpath("full-length.csv"))

    assert type(full_emb_lengths) == np.ndarray
    assert full_emb_lengths.ndim == 2
