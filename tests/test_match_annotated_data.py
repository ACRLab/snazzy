from pathlib import Path
from pasnascope import centerline_errors


def test_matching_embryo_pairs_maps_strs_to_paths():
    embryos = [Path(f) for f in ['/fake/emb1-ch2.tif', '/fake/emb2-ch2.tif']]
    annotated = [Path(f) for f in ['/fake/emb1-ch2.csv', '/fake/emb2-ch2.csv']]
    res = centerline_errors.get_matching_embryos(embryos, annotated)

    ann_file_name, emb_path = next(iter(res.items()))
    assert isinstance(ann_file_name, str)
    assert isinstance(emb_path, str)


def test_maps_to_same_name_when_LUT_is_None():
    embryos = [Path(f) for f in ['/fake/emb1-ch2.tif', '/fake/emb2-ch2.tif']]
    annotated = [Path(f) for f in ['/fake/emb1-ch2.csv', '/fake/emb2-ch2.csv']]
    expected = {'emb1-ch2.tif': 'emb1-ch2.csv',
                'emb2-ch2.tif': 'emb2-ch2.csv'}
    actual = centerline_errors.get_matching_embryos(embryos, annotated)

    assert expected == actual


def test_maps_to_emb_when_given_LUT():
    LUT = {1: 4, 2: 12}
    embryos = [Path(f) for f in ['/fake/emb1-ch2.tif', '/fake/emb2-ch2.tif']]
    annotated = [Path(f)
                 for f in ['/fake/emb4-ch2.csv', '/fake/emb12-ch2.csv']]
    expected = {'emb1-ch2.tif': 'emb4-ch2.csv',
                'emb2-ch2.tif': 'emb12-ch2.csv'}
    actual = centerline_errors.get_matching_embryos(embryos, annotated, LUT)

    assert expected == actual


def test_maps_to_emb_when_given_LUT_partial():
    LUT = {1: 4, 2: 12}
    embryos = [Path(f) for f in ['/fake/emb1-ch2.tif', '/fake/emb2-ch2.tif']]
    annotated = [Path(f) for f in ['/fake/emb1-ch2.csv',
                                   '/fake/emb2-ch2.csv',
                                   '/fake/emb12-ch2.csv',
                                   '/fake/emb4-ch2.csv']]
    expected = {'emb1-ch2.tif': 'emb4-ch2.csv',
                'emb2-ch2.tif': 'emb12-ch2.csv'}
    actual = centerline_errors.get_matching_embryos(embryos, annotated, LUT)

    assert expected == actual


def test_empty_annotated_dir_returns_empty_dict():
    LUT = {1: 4, 2: 12}
    embryos = [Path(f) for f in ['/fake/emb1-ch2.tif', '/fake/emb2-ch2.tif']]
    annotated = []
    expected = {}
    actual = centerline_errors.get_matching_embryos(embryos, annotated, LUT)

    assert expected == actual


def test_empty_annotated_dir_returns_empty_dict_when_LUT_None():
    embryos = [Path(f) for f in ['/fake/emb1-ch2.tif', '/fake/emb2-ch2.tif']]
    annotated = []
    expected = {}
    actual = centerline_errors.get_matching_embryos(embryos, annotated)

    assert expected == actual
