from pathlib import Path

import pytest

from pasna_analysis import DataLoader

VALID_DIR = Path("./tests/assets/20250210")
# MISSING_DATA_DIR does not have full_length.csv file.
MISSING_DATA_DIR = Path("./tests/assets/20250101")


def test_creates_instance_when_dir_matches_expected_structure():
    dl = DataLoader(VALID_DIR)

    assert dl is not None
    assert dl.path == VALID_DIR


def test_raises_when_missing_files():
    with pytest.raises(AssertionError):
        DataLoader(MISSING_DATA_DIR)


def test_lists_embryos_correctly():
    dl = DataLoader(VALID_DIR)

    expected_embs = ["emb1", "emb3", "emb4"]
    embs = dl.embryos()

    for expected, emb in zip(sorted(expected_embs), embs):
        assert expected == emb


def test_list_activity_files_correctly():
    dl = DataLoader(VALID_DIR)

    expected_embs = ["emb1.csv", "emb3.csv", "emb4.csv"]
    embs = dl.activities()

    for expected, emb in zip(sorted(expected_embs), embs):
        expected_path = VALID_DIR / "activity" / expected
        assert expected_path == emb


def test_list_length_files_correctly():
    dl = DataLoader(VALID_DIR)

    expected_embs = ["emb1.csv", "emb3.csv", "emb4.csv"]
    embs = dl.lengths()

    for expected, emb in zip(sorted(expected_embs), embs):
        expected_path = VALID_DIR / "lengths" / expected
        assert expected_path == emb
