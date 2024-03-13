from pasnascope import roi
import numpy as np


def test_get_rois_for_each_slice():
    img = np.load('./tests/images/embryo.npy')
    rois = roi.get_roi(img, window=1)

    assert rois.shape == img.shape


def test_get_rois_with_default_window():
    '''The resulting roi will be downsampled 10x in its outer dimension.'''
    img = np.load('./tests/images/embryo.npy')
    rois = roi.get_roi(img)

    assert rois.shape == (img.shape[0]//10, *img.shape[1:])


def test_get_rois_with_given_window():
    img = np.load('./tests/images/embryo.npy')
    rois = roi.get_roi(img, window=5)

    assert rois.shape == (img.shape[0]//5, *img.shape[1:])
