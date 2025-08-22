from pathlib import Path

from tifffile import imread
import pytest

from golf_processing import vnc_length

MOVIE_CH2 = Path(__file__).parent.joinpath("images", "embryo_movies", "emb1-ch2.tif")


@pytest.fixture
def img():
    struct_img = imread(MOVIE_CH2)
    return struct_img[:50]


def test_can_measure_vnc(img):
    lengths = vnc_length.measure_VNC_centerline(img)

    assert lengths is not None
    assert len(lengths) == img.shape[0]
