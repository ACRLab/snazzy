from golf_processing import roi
import numpy as np


class TestRoiDimensionality:
    def test_roi_shape_without_downsampling(self):
        img = np.load("./tests/images/embryo.npy")
        rois = roi.get_roi(img, window=1)

        assert rois.shape == img.shape

    def test_roi_shape_using_default_downsampling(self):
        """The resulting roi will be downsampled 10x in its outer dimension."""
        img = np.load("./tests/images/embryo.npy")
        rois = roi.get_roi(img)

        assert rois.shape == (img.shape[0] // 10, *img.shape[1:])

    def test_roi_shape_for_given_downsampling(self):
        img = np.load("./tests/images/embryo.npy")
        rois = roi.get_roi(img, window=5)

        assert rois.shape == (img.shape[0] // 5, *img.shape[1:])
