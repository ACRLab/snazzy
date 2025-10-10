import math
import numpy as np

from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label
from skimage.morphology import remove_small_objects


def get_single_roi(img: np.ndarray) -> np.ndarray:
    """Calculates the ROI of a 2D grayscale image.

    Values *outside* the ROI are marked as True, values inside are False.

    Parameters:
        img (np.ndarray):
            A 2D numpy array representing an image.
    """
    if img.ndim != 2:
        raise ValueError("img should be a 2D matrix.")

    slc = img.copy()
    thres = threshold_otsu(slc)
    binary_mask_compl = slc < thres
    # this is a bit more efficient than to call `remove_small_holes`
    remove_small_objects(binary_mask_compl, 200, out=binary_mask_compl)
    removed = np.logical_not(binary_mask_compl)

    labels, num_labels = label(removed, return_num=True, connectivity=1)

    # skip frames if no region was found
    if num_labels == 0:
        return

    # creates a boolean mask with the most frequent label marked as True
    # sckimage.measure.label will label background pixels as 0
    # bincount will count the amount of occurences of each value
    # we remove the first index returned, because it will correspond to
    # the amount of zeros (background)
    # we get the index where we have the maximum frequency
    # and then compare each element to this maximum
    largest_label = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return np.logical_not(largest_label)


def get_roi(img: np.ndarray, window=10) -> np.ndarray:
    """The ROIs for a  3D image, after downsampling the slices by `window`.

    Parameters:
        img (np.ndarray):
            A 3D np array representing an image.
        window (int):
            Interval used to calculate a new ROI.
            Defaults to 10.
    """
    if img.ndim != 3:
        raise ValueError("img should be a 3D array.")

    num_slices = img.shape[0]
    rois_length = math.ceil(num_slices / window)
    rois = np.empty((rois_length, *img.shape[1:]), dtype=np.bool_)

    # calculates a new ROI in steps of `window`:
    for idx, i in enumerate(range(0, num_slices, window)):
        avg_slc = np.average(img[i : i + window], axis=0)
        rois[idx] = get_single_roi(avg_slc)

    return rois


def get_contours(img: np.ndarray, window=10) -> list[np.ndarray]:
    """Returns the contours of each image, based on their ROI.

    Parameters:
        img (np.ndarray):
            A 3D np array representing an image.
        window (int):
            Interval used to calculate a new ROI.
            Defaults to 10.
    """
    rois = get_roi(img, window=window)

    contours = []

    for roi in rois:
        if roi is None:
            contours.append(contours[-1])
            continue
        contour = find_contours(roi)
        if len(contour) > 0:
            contours.append(contour[0])
    return contours
