from pathlib import Path

import pytest

from pasna_analysis import Experiment

VALID_DIR = Path("./tests/assets/20250210")
# MISSING_DATA_DIR does not have full_length.csv file.
MISSING_DATA_DIR = Path("./tests/assets/20250101")


@pytest.fixture
def exp():
    return Experiment(VALID_DIR)


def test_can_create_experiment(exp):
    assert exp is not None


def test_use_all_embryos(exp):
    assert exp.embryos is not None
    assert len(exp.embryos) == 3


def test_can_exclude_embryos(exp):
    to_exclude = [1]
    exp = Experiment(VALID_DIR, to_exclude=to_exclude)

    assert len(exp.embryos) == 2


def test_ignores_embryos_not_in_experiment():
    # VALID_DIR only contains emb1, emb3, and emb4
    to_exclude = [15]

    exp = Experiment(VALID_DIR, to_exclude=to_exclude)

    assert len(exp.embryos) == 3


def test_skips_embryos_before_first_peak_threshold(capsys):
    # use a high value for first_peak_threshold to force skipping embryos
    exp = Experiment(VALID_DIR, first_peak_threshold=60)

    captured = capsys.readouterr()

    # in VALID_DIR, first peaks are 2136, 8478, 2646 secs
    assert len(exp.embryos) == 1
    assert (
        captured.out.index(
            f"First peak detected before {exp.first_peak_threshold} mins."
        )
        >= 0
    )
