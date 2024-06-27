from pathlib import Path
from pasnascope import centerline_errors


def test_matching_embryo_pairs_maps_strs_to_paths():
    img_dir = Path('/home/cdp58/Documents/repos/pasnascope/20240619/embs')
    annotated = [Path(f) for f in ['/fake/emb1.csv', '/fake/emb2.csv']]
    res = centerline_errors.get_matching_embryos(annotated, img_dir)

    ann_file_name, emb_path = next(iter(res.items()))
    assert isinstance(ann_file_name, str)
    assert isinstance(emb_path, Path)


def test_maps_to_same_name_when_LUT_is_None():
    img_dir = Path('/home/cdp58/Documents/repos/pasnascope/20240619/embs')
    annotated = [Path(f) for f in ['/fake/emb1.csv', '/fake/emb2.csv']]
    expected = {'emb1': img_dir.joinpath('emb1.tif'),
                'emb2': img_dir.joinpath('emb2.tif')}
    actual = centerline_errors.get_matching_embryos(annotated, img_dir)

    assert expected == actual


def test_maps_to_emb_when_given_LUT():
    LUT = {1: 4, 2: 12}
    img_dir = Path('/home/cdp58/Documents/repos/pasnascope/20240619/embs')
    annotated = [Path(f) for f in ['/fake/emb1-ch2.csv', '/fake/emb2-ch2.csv']]
    expected = {'emb1-ch2': img_dir.joinpath('emb4-ch2.tif'),
                'emb2-ch2': img_dir.joinpath('emb12-ch2.tif')}
    actual = centerline_errors.get_matching_embryos(annotated, img_dir, LUT)

    assert expected == actual


def test_maps_to_emb_when_given_LUT_partial():
    LUT = {1: 4, 2: 12}
    img_dir = Path('/home/cdp58/Documents/repos/pasnascope/20240619/embs')
    annotated = [Path(f) for f in ['/fake/emb1-ch2.csv',
                                   '/fake/emb2-ch2.csv',
                                   '/fake/emb3-ch2.csv',
                                   '/fake/emb2-ch4.csv']]
    expected = {'emb1-ch2': img_dir.joinpath('emb4-ch2.tif'),
                'emb2-ch2': img_dir.joinpath('emb12-ch2.tif')}
    actual = centerline_errors.get_matching_embryos(annotated, img_dir, LUT)

    assert expected == actual


def test_empty_annotated_dir():
    LUT = {1: 4, 2: 12}
    img_dir = Path('/home/cdp58/Documents/repos/pasnascope/20240619/embs')
    annotated = []
    expected = {}
    actual = centerline_errors.get_matching_embryos(annotated, img_dir, LUT)

    assert expected == actual
