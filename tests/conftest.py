from pathlib import Path
import os

import pytest


@pytest.fixture(scope="module", autouse=True)
def clear_pd_params():
    yield
    for dir in Path("tests/assets/data").iterdir():
        pd_params_path = dir.joinpath("peak_detection_params.json")
        if pd_params_path.exists():
            os.remove(pd_params_path)
