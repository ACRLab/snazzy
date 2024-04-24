import numpy as np
import math

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import binary_opening, binary_erosion, remove_small_holes, disk

from pasnascope.animations.custom_animation import CentroidAnimation, ContourAnimation


def get_single_roi(img):
    '''Calculates the ROI of a 2D grayscale image.

    Values *outside* the ROI are marked as True, values inside are False.'''
    if img.ndim != 2:
        raise ValueError('img should be a 2D matrix.')

    slc = img.copy()
    thres = threshold_otsu(slc)
    binary_mask = slc > thres

    slc[...] = 0
    slc[binary_mask] = 1
    slc = slc.astype(np.bool_)
    remove_small_holes(slc, 200, out=slc)
    binary_opening(slc, footprint=disk(3), out=slc)

    labels, num_labels = label(slc, return_num=True, connectivity=1)

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
    largest_label = labels == np.argmax(
        np.bincount(labels.flat)[1:])+1

    return np.logical_not(largest_label)


def get_roi(img, window=10, mask=None):
    '''The ROI for an image, after downsampling the slices by `window`.'''
    if img.ndim != 3:
        raise ValueError('img should be a 3D array.')

    num_slices = img.shape[0]
    rois_length = math.ceil(num_slices/window)
    rois = np.empty((rois_length, *img.shape[1:]), dtype=np.bool_)

    # calculates a new ROI in steps of `window`:
    for i in range(0, num_slices, window):
        avg_slc = np.average(img[i:i+window], axis=0)
        if mask is not None:
            avg_slc[mask] = 0
        rois[i//window] = get_single_roi(avg_slc)

    return rois


def global_roi(img):
    '''Creates a single ROI for all the slices in the image.

    It's faster than calculating one ROI for each slice, and can be used if
    the embryo does not move.'''
    avg_img = np.average(img, axis=0)
    return get_single_roi(avg_img)


def cache_rois(img, file_path):
    '''Saves ROI as a numpy file.'''
    rois = get_roi(img)

    with open(file_path, 'wb+') as f:
        np.save(f, rois)

    print(f'Saved ROIs in `{file_path}`.')


def get_initial_mask(img, n):
    '''Create a mask based on the first n frames of the movies.

    The mask is eroded to give space to account for the embryo flickering.'''
    init = np.average(img[:n], axis=0)
    first_mask = get_single_roi(init)
    binary_erosion(first_mask, footprint=disk(10), out=first_mask)
    return first_mask


def get_contours(img, window=10, mask=None):
    '''Returns the contours of each image, based on their ROI.'''
    rois = get_roi(img, window=window, mask=mask)

    contours = []

    for roi in rois:
        # TODO: repeat the previous contour in no ROI is found?
        if roi is None:
            continue
        contour = find_contours(roi)[0]
        if len(contour) > 0:
            contours.append(contour)
    return contours


def get_contour(img):
    '''Returns the contour for a single image.'''
    return find_contours(img)[0]


def plot_contours(img):
    '''Plots image with contours overlayed.'''
    contours = get_contours(img)
    pa = ContourAnimation(img[200:400], contours, interval=300)
    pa.display()


def get_centroids(img):
    '''Get the centroid of each slice of a ROI within an image.'''
    centroids = []

    for slc in img:
        roi = get_single_roi(slc)
        # largest_label is a boolean mask, regionprops needs a binary image
        regions = regionprops(roi.astype(np.uint8))
        props = regions[0]
        y0, x0 = props.centroid
        centroids.append([y0, x0])

    return centroids


def plot_centroids(img):
    centroids = get_centroids(img)
    ca = CentroidAnimation(img, centroids, interval=150)
    ca.display()
