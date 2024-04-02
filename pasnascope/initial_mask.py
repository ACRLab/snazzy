import numpy as np
from skimage.draw import ellipse
from skimage.filters import threshold_otsu
from scipy.optimize import minimize
from skimage.measure import label, regionprops


def ellipse_count_cost(opts, img, weights=[4, 1]):
    """
    Calculates the cost based on the overlapping of the ellipse and image.

    Args:
        opts: Tuple with params that will be optimized. Expects to receive the following [x_centre, y_centre, minor_axis, major_axir, angle].
        img: Grayscale image as a numpy array.
        weights: 2 element tuple, with weights for foreground and background pixel count.

    Returns:
        The weighted subtraction between foreground pixels and background pixels within the ellipse (to minimize during optimization).
    """

    rr, cc, minor_axis, major_axis, angle = opts
    mask = np.zeros_like(img)
    y, x = ellipse(rr, cc,
                   minor_axis, major_axis, shape=img.shape, rotation=angle)
    mask[y, x] = 1
    within_ellipse = img[mask == 1]
    fgd = np.count_nonzero(within_ellipse)
    bgd = within_ellipse.size - fgd
    wf, wb = weights

    return -1*wf*fgd + wb*bgd


def fit_ellipse(img):
    '''Finds the best fit for an elliptical ROI for the given img.'''
    img = np.average(img, axis=0)
    # binarize image:
    thr = threshold_otsu(img)
    binary_mask = img > thr
    img[...] = 0
    img[binary_mask] = 1
    img = img.astype(np.int64)
    # label:
    labels, num_labels = label(img, return_num=True, connectivity=1)
    if num_labels == 0:
        return
    largest_label = labels == np.argmax(
        np.bincount(labels.flat)[1:])+1
    largest_label = largest_label.astype(np.int64)

    # initial guesses:
    regions = regionprops(largest_label, img)
    rr, cc = regions[0]['centroid']
    rr = int(rr)
    cc = int(cc)
    # TODO: maybe use the guesses from regionprops
    initial_maj_axis = 60
    initial_min_axis = 15
    angle = 0

    # fit:
    x0 = [rr, cc, initial_min_axis, initial_maj_axis, angle]
    bounds = [(30, 120), (30, 120), (10, 25), (20, 80), (-0.5, 0.5)]
    res = minimize(ellipse_count_cost, x0=x0, args=(img,),
                   method="Powell", bounds=bounds)
    return res.x[:5]


def create_mask_from_ellipse(coords, shape):
    '''Converts the output from `fit_ellipse` into an image mask.'''
    mask = np.ones(shape)
    x, y = ellipse(*coords[:-1], shape=shape, rotation=coords[-1])
    mask[x, y] = 0
    return mask.astype(np.bool_)
