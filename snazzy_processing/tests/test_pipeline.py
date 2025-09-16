from pathlib import Path
from unittest.mock import patch

import pytest

from snazzy_processing import pipeline

BASE_DIR = Path(__file__).parent.joinpath("images")
EMB_SRC = BASE_DIR.joinpath("embryo_movies")


@pytest.fixture(scope="session")
def shared_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("shared_dir")


def test_skips_embryo_if_already_processed():
    with patch("snazzy_processing.pipeline.already_created", lambda x, y: True):
        processed_lens = pipeline.measure_vnc_length(EMB_SRC, EMB_SRC, downsampling=20)
        processed_acts = pipeline.calc_activities(EMB_SRC, EMB_SRC, window=10)

        assert processed_lens == 0
        assert processed_acts == 0

        # should not create output directories
        assert not EMB_SRC.joinpath("activity").exists()
        assert not EMB_SRC.joinpath("lengths").exists()


def test_cleanup_removes_all(tmp_path):
    embs_src = tmp_path / "embs"
    embs_src.mkdir()
    (embs_src / "emb1-ch1.tif").write_text("emb1")

    first_frames_path = tmp_path / "first_frames.tif"
    first_frames_path.write_text("first frames")

    tif_path = tmp_path / "temp.tif"
    tif_path.write_text("tif")

    pipeline.clean_up_files(embs_src, first_frames_path, tif_path)

    assert not embs_src.exists()
    assert not first_frames_path.exists()
    assert not tif_path.exists()


def test_cleanup_raises_when_missing_paths(tmp_path):
    # Missing paths are just ignored
    embs_src = tmp_path / "does_not_exist"
    first_frames_path = tmp_path / "missing_frame.txt"
    tif_path = tmp_path / "missing.tif"

    with pytest.raises(FileNotFoundError):
        pipeline.clean_up_files(embs_src, first_frames_path, tif_path)


def test_pipeline_write_csv_files(shared_dir):
    with patch("snazzy_processing.find_hatching.find_hatching_point", lambda x: 50):
        processed_lens = pipeline.measure_vnc_length(EMB_SRC, shared_dir, 10)
        processed_acts = pipeline.calc_activities(EMB_SRC, shared_dir, 10)
        processed_full_lens = pipeline.measure_embryo_full_length(EMB_SRC, shared_dir)

        assert shared_dir.joinpath("activity").exists()
        assert shared_dir.joinpath("lengths").exists()

        assert processed_lens == 1
        assert processed_acts == 1
        assert processed_full_lens == 1

        assert len(list(shared_dir.joinpath("activity").iterdir())) == 1
        assert len(list(shared_dir.joinpath("lengths").iterdir())) == 1


def test_pipeline_skips_if_already_exists(shared_dir):
    with patch("snazzy_processing.find_hatching.find_hatching_point", lambda x: 50):
        processed_lens = pipeline.measure_vnc_length(EMB_SRC, shared_dir, 25)

    assert shared_dir.joinpath("lengths").exists()
    assert processed_lens == 0


def test_calculate_activity_single_emb():
    act_path = EMB_SRC.joinpath("emb1-ch1.tif")
    stct_path = EMB_SRC.joinpath("emb1-ch2.tif")

    img_len = 101
    interval = 10

    id, signals = pipeline.calc_activity(act_path, stct_path, interval)

    assert id == 1
    assert signals.shape == (img_len, 2)


def test_calculate_activity_raises_if_embs_not_match():
    with pytest.raises(ValueError):
        act_path = EMB_SRC.joinpath("emb1-ch1.tif")
        stct_path = EMB_SRC.joinpath("emb5-ch1.tif")
        pipeline.calc_activity(act_path, stct_path, window=10)
