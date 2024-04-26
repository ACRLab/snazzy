import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.measure import label
from skimage.morphology import remove_small_holes, binary_opening, disk
from skimage.draw import line
from skimage.feature import peak_local_max
from sklearn import linear_model


def centerline_dist(image):
    '''Returns the centerline length estimation based on EDT maxima points.'''
    # binarize image:
    thr = threshold_multiotsu(image)
    binary_mask = image > thr[0]
    image[...] = 0
    image[binary_mask] = 1
    image = image.astype(np.bool_)
    # label:
    labels = label(image, connectivity=1)
    largest_label = labels == np.argmax(
        np.bincount(labels.flat)[1:])+1

    remove_small_holes(largest_label, 200, out=image)
    binary_opening(image, footprint=disk(5), out=image)

    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)

    if coords.size <= 2:
        print(f"Found a single point of maxima, cannot apply RANSAC.")
        return None
    print(f"Found {coords.size} points of maxima.")

    y = coords.T[0]
    x = coords.T[1].reshape(-1, 1)

    thres = np.median(np.abs(x - np.median(x)))
    thres = min(thres, 5)
    ransac = linear_model.RANSACRegressor(
        residual_threshold=thres, max_trials=1000)
    ransac = linear_model.RANSACRegressor()
    ransac.fit(x, y)

    # predict points throught the entire image to create line
    x1 = 0
    x2 = image.shape[1]-1
    y1, y2 = [int(y) for y in ransac.predict([[x1], [x2]])]

    if y1 < 0 or y1 >= image.shape[0] or y2 < 0 or y2 >= image.shape[0]:
        print(f"Prediction is out of bounds for rows {y1} and {y2}")
        return

    rr, cc = line(y1, x1, y2, x2)

    # mask based on the line
    inner = np.zeros_like(image)
    inner[rr, cc] = 1
    image[~inner] = 0

    # first and last points where the line intersects the image
    nonzeros = np.argwhere(image)
    vnc_start = nonzeros[0]
    vnc_end = nonzeros[-1]
    distance = np.sqrt(np.sum((vnc_end-vnc_start)**2))

    return distance


def view_centerline_dist(image):
    '''Visualization for centerline method. 

    Displays inliers, outliers, centerline, and image together.'''
    orig_image = image.copy()

    # binarize image:
    thr = threshold_multiotsu(image)
    binary_mask = image > thr[0]
    image[...] = 0
    image[binary_mask] = 1
    image = image.astype(np.bool_)
    # label:
    labels = label(image, connectivity=1)
    largest_label = labels == np.argmax(
        np.bincount(labels.flat)[1:])+1

    remove_small_holes(largest_label, 200, out=image)
    binary_opening(image, footprint=disk(5), out=image)

    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    print(f"Found {coords.size} local maxima points.")

    y = coords.T[0]
    x = coords.T[1].reshape(-1, 1)

    res_thres = np.median(np.abs(x - np.median(x)))
    res_thres = min(res_thres, 5)
    ransac = linear_model.RANSACRegressor(
        residual_threshold=res_thres, max_trials=1000)
    ransac.fit(x, y)
    # visualize inliers and outliers
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # predict points throught the entire image to create line
    x1 = 0
    x2 = image.shape[1]-1
    y1, y2 = [int(y) for y in ransac.predict([[x1], [x2]])]

    if y1 < 0 or y1 >= image.shape[0] or y2 < 0 or y2 >= image.shape[0]:
        print(f"Prediction is out of bounds for rows {y1} and {y2}")
        return

    rr, cc = line(y1, x1, y2, x2)

    # mask based on the line
    inner = np.zeros_like(image)
    inner[rr, cc] = 1
    image[~inner] = 0

    # first and last points where the line intersects the image
    nonzeros = np.argwhere(image)
    vnc_start = nonzeros[0]
    vnc_end = nonzeros[-1]
    distance = np.sqrt(np.sum((vnc_end-vnc_start)**2))

    # plot results: visualize RANSAC, EDT maxima and predicted centerline
    fig, ax = plt.subplots()
    ax.scatter(x[inlier_mask], y[inlier_mask], color='yellowgreen')
    ax.scatter(x[outlier_mask], y[outlier_mask], color='fuchsia')
    ax.imshow(orig_image)
    ax.imshow(inner, alpha=0.5)
    plt.show()


def view_markers(image):
    '''Plots local maxima points overlayed to image.'''
    orig_image = image.copy()
    # binarize image:
    thr = threshold_multiotsu(image)
    binary_mask = image > thr[0]
    image[...] = 0
    image[binary_mask] = 1
    image = image.astype(np.bool_)
    # label:
    labels = label(image, connectivity=1)
    largest_label = labels == np.argmax(
        np.bincount(labels.flat)[1:])+1

    remove_small_holes(largest_label, 200, out=image)
    binary_opening(image, footprint=disk(5), out=image)

    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=disk(3), labels=image)
    print(f"Found {coords.size} points of maxima.")

    y = coords.T[0]
    x = coords.T[1].reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.scatter(x, y, color='fuchsia')
    ax.imshow(orig_image)
    plt.show()
