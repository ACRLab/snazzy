import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.measure import label
from skimage.morphology import remove_small_holes, binary_opening, disk
from skimage.draw import line
from skimage.feature import peak_local_max
from sklearn import linear_model


def binarize(image):
    '''Returns a binary image using a low threshold.'''
    thr = threshold_multiotsu(image)
    bin_img = image > thr[0]

    labels = label(bin_img, connectivity=1)
    largest_label = labels == np.argmax(
        np.bincount(labels.flat)[1:])+1

    remove_small_holes(largest_label, 200, out=bin_img)
    binary_opening(bin_img, footprint=disk(5), out=bin_img)

    return bin_img


def get_DT_maxima(image):
    '''Calculates a distance transform and returns local maxima points.'''
    distance = ndi.distance_transform_cdt(image, metric='chessboard')
    return peak_local_max(distance, footprint=np.ones((5, 5)), labels=image)


def apply_ransac(coords):
    '''Returns the centerline estimated by applying a RANSAC linear model.'''
    y = coords.T[0]
    x = coords.T[1].reshape(-1, 1)

    thres = np.median(np.abs(x - np.median(x)))
    thres = min(thres, 10)
    ransac = linear_model.RANSACRegressor(
        estimator=linear_model.LinearRegression(),
        residual_threshold=thres, max_trials=1000, min_samples=2)
    return ransac.fit(x, y)


def centerline_dist(image, verbose=False):
    '''Returns the centerline length estimation based on EDT maxima points.'''
    image = binarize(image)
    coords = get_DT_maxima(image)

    if coords.shape[0] <= 2:
        if verbose:
            print(f"Found a single point of maxima, cannot apply RANSAC.")
        return None

    estimator = apply_ransac(coords)

    # points to draw the line mask
    x1 = 0
    x2 = image.shape[1]-1
    y1, y2 = [int(y) for y in estimator.predict([[x1], [x2]])]

    rr, cc = line(y1, x1, y2, x2)
    # the RANSAC estimations might be out of the image range
    rr = np.clip(rr, 0, image.shape[0]-1)

    # create and apply mask
    inner = np.zeros_like(image)
    inner[rr, cc] = 1
    image[~inner] = 0

    # first and last points where the line intersects the image
    nonzeros = np.argwhere(image)
    nonzeros = np.sort(nonzeros, axis=0)
    vnc_start = nonzeros[0]
    vnc_end = nonzeros[-1]
    distance = np.sqrt(np.sum((vnc_end-vnc_start)**2))

    return distance


def view_centerline_dist(image):
    '''Returns the centerline length estimation based on EDT maxima points.'''
    orig_image = image.copy()
    image = binarize(image)
    coords = get_DT_maxima(image)

    y = coords.T[0]
    x = coords.T[1].reshape(-1, 1)
    if coords.shape[0] <= 2:
        return None

    estimator = apply_ransac(coords)
    inliers = estimator.inlier_mask_
    outliers = np.logical_not(inliers)

    # points to draw the line mask
    x1 = 0
    x2 = image.shape[1]-1
    y1, y2 = [int(y) for y in estimator.predict([[x1], [x2]])]

    rr, cc = line(y1, x1, y2, x2)

    # the RANSAC estimations might be out of the image range
    rr = np.clip(rr, 0, image.shape[0]-1)

    # create and apply mask
    inner = np.zeros_like(image)
    inner[rr, cc] = 1
    image[~inner] = 0

    # first and last points where the line intersects the image
    nonzeros = np.argwhere(image)
    nonzeros = np.sort(nonzeros, axis=0)

    fig, ax = plt.subplots()
    ax.scatter(x[inliers], y[inliers], color='firebrick')
    ax.scatter(x[outliers], y[outliers], color='fuchsia')
    ax.imshow(orig_image)
    ax.imshow(inner, alpha=0.3)
    fig.canvas.header_visible = False
    plt.show()
