from pathlib import Path
import os

import pytest

DATA_DIR = Path(__file__).parent.joinpath("assets", "data")


@pytest.fixture(scope="module", autouse=True)
def clear_pd_params():
    yield
    for dir in DATA_DIR.iterdir():
        pd_params_path = dir.joinpath("peak_detection_params.json")
        if pd_params_path.exists():
            os.remove(pd_params_path)
