from pasnascope import roi
import numpy as np


class TestRoiDimensionality:
    def test_get_rois_for_each_slice(self):
        img = np.load('./tests/images/embryo.npy')
        rois = roi.get_roi(img, window=1)

        assert rois.shape == img.shape

    def test_get_rois_with_default_window(self):
        '''The resulting roi will be downsampled 10x in its outer dimension.'''
        img = np.load('./tests/images/embryo.npy')
        rois = roi.get_roi(img)

        assert rois.shape == (img.shape[0]//10, *img.shape[1:])

    def test_get_rois_with_given_window(self):
        img = np.load('./tests/images/embryo.npy')
        rois = roi.get_roi(img, window=5)

        assert rois.shape == (img.shape[0]//5, *img.shape[1:])


def test_gets_correct_mask():
    expected_mask = np.load(
        'tests/images/mask_moving_rects.npy').astype(np.bool_)
    img = np.load('./tests/images/moving_rects.npy')
    rois = roi.get_roi(img, window=10)

    assert np.array_equal(expected_mask, rois)
