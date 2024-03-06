import numpy as np
from tifffile import imread

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import opening, disk

from pasnascope.animations.custom_animation import CentroidAnimation, ContourAnimation


def get_single_roi(img):
    '''Calculates the ROI of a 2D grayscale image.'''
    slc = img.copy()
    thres = threshold_otsu(slc)
    binary_mask = slc > thres

    slc[...] = 0
    slc[binary_mask] = 1
    opening(slc, footprint=disk(4), out=slc)

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

    return largest_label


def get_roi(img):
    '''Returns an nparray with an individual ROI for each slice of the image.'''
    rois = np.zeros(img.shape)

    for i, slc in enumerate(img):
        rois[i] = get_single_roi(slc)

    return rois


def global_roi(img):
    '''Creates a single ROI for all the slices in the image.

    It's faster than calculating one ROI for each slice, and can be used if
    the embryo does not move.'''
    avg_img = np.average(img, axis=0)
    return get_single_roi(avg_img)


def cache_rois(img):
    '''Saves ROI as a numpy file.'''
    rois = get_roi(img)
    filename = img.split('/')[-1][:-4]

    with open(f'./results/cache/roi-{filename}.npy', 'wb') as f:
        np.save(f, rois)

    print(f'Saved ROIs in `./results/cache/roi-{filename}.npy`.')


def get_contours(img):
    '''Returns the contours of each image, base on their ROI.'''
    rois = get_roi(img)

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


if __name__ == '__main__':
    img_path = '/home/cdp58/Documents/repos/pasnascope_analysis/data/embryos/'
    img = imread(img_path + 'emb12-ch2.tif')
    plot_contours(img)
