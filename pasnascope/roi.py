import numpy as np
import math

from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import binary_opening, octagon

from pasnascope.animations.custom_animation import CentroidAnimation, ContourAnimation


def get_single_roi(img):
    '''Calculates the ROI of a 2D grayscale image.

    Values *outside* the ROI are marked as True, values inside are False.'''
    slc = img.copy()
    if np.unique(slc).size <= 2:
        # use regular otsu threshold in case of a binary image
        thres = threshold_otsu(slc)
    else:
        thres = threshold_multiotsu(slc, classes=3)[0]
    binary_mask = slc > thres

    slc[...] = 0
    slc[binary_mask] = 1
    binary_opening(slc, footprint=octagon(3, 3), out=slc)

    labels, num_labels = label(slc, return_num=True, connectivity=2)

    # skip frames where no region is found
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


def get_roi(img, window=10):
    '''The ROI for an image, after downsampling the slices by `window`.'''
    num_slices = img.shape[0]
    rois_length = math.ceil(num_slices/window)
    rois = np.empty((rois_length, *img.shape[1:]), dtype=np.bool_)

    for i in range(num_slices):
        # calculates a new ROI in steps of `window`:
        if i % window == 0:
            j = i // window
            avg_slc = np.average(img[j*window:(j+1)*window], axis=0)
            rois[j] = get_single_roi(avg_slc)

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


def get_contours(img, window=10):
    '''Returns the contours of each image, base on their ROI.'''
    rois = get_roi(img, window=window)

    contours = []

    for roi in rois:
        # TODO: repeat the previous contour in no ROI is found?
        if roi is None:
            continue
        contour = find_contours(roi)[0]
        if len(contour) > 0:
            contours.append(contour)
    return contours


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
