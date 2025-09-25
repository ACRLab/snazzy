import warnings

from matplotlib.axes import Axes
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import label
from skimage.morphology import binary_opening, disk, remove_small_holes
from sklearn import linear_model
from sklearn.linear_model import RANSACRegressor


def binarize(image: np.ndarray, threshold_method="multiotsu"):
    """Create a binary image from the largest region after thresholding.

    Parameters:
        image (np.ndarray):
            A 2 dimensional np array.
        threshold_method ('multiotsu' | 'otsu'):
            Threshold method used to binary the image.
            For a higher threshold value, use 'otsu'.

    Returns:
        binary_image (np.ndarray):
            binary image with same dimensions as `image`.
    """
    if image.ndim != 2:
        raise ValueError("Image can only have 2 dimensions.")

    if threshold_method == "multiotsu":
        thr = threshold_multiotsu(image)
        bin_img = image > thr[0]
    elif threshold_method == "otsu":
        thr = threshold_otsu(image)
        bin_img = image > thr
    else:
        raise ValueError(f"Unsupported threshold method: {threshold_method}.")

    # morphological operations to make the ROI better fit the VNC:
    remove_small_holes(bin_img, 200, out=bin_img)
    binary_opening(bin_img, footprint=disk(5), out=bin_img)

    labels = label(bin_img, connectivity=1)
    largest_label = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return largest_label


def get_DT_maxima(image: np.ndarray, thres_rel=0.6, min_dist=5) -> np.ndarray:
    """Points of local maxima from a distance transform image.

    Parameters:
        image (np.ndarray):
            A 2 dimensional np array.
        thres_rel (float):
            Minimum intensity of local maxima points relative to the maximum.
        min_dist (int):
            Minimum distance separating local maxima points.

    Returns:
        np.ndarray:
            Array of coordinate pairs.
    """
    distance_transform = ndi.distance_transform_cdt(image, metric="chessboard")
    return peak_local_max(
        distance_transform,
        footprint=np.ones((5, 5)),
        threshold_rel=thres_rel,
        min_distance=min_dist,
    )


def get_DT_image(binary_image: np.ndarray, metric="chessboard") -> np.ndarray:
    """Returns the distance transform image for visualization."""
    return ndi.distance_transform_cdt(binary_image, metric=metric)


def apply_ransac(coords: np.ndarray):
    """Returns the centerline estimated by applying a RANSAC linear model.

    Parameters:
        coords (np.ndarray):
            Array of coordinate points (y, x)
    """
    y = coords.T[0]
    x = coords.T[1].reshape(-1, 1)

    thres = 15
    # if the number of inliers for a given trial is <= 1,
    # `_regression.r2_score` will raise a warning. This is expected to happen
    # depending on the dataset, so we just ignore the warning
    with warnings.catch_warnings(action="ignore"):
        ransac = linear_model.RANSACRegressor(
            estimator=linear_model.LinearRegression(),
            residual_threshold=thres,
            max_trials=1000,
            stop_score=0.999,
            min_samples=3,
        )
        regressor = ransac.fit(x, y)
    return regressor


def centerline_mask(img_shape: tuple, predictor: RANSACRegressor.predict) -> np.ndarray:
    """Create a mask from RANSAC predicted values.

    Parameters:
        img_shape (tuple):
            Shape of a 2D image, used to create an output mask of same shape.
        predictor (RANSACRegressor.predict):
            Fitted RANSACRegressor predictor.

    Returns:
        mask (np.ndarray):
            Centerline values as a mask with same shape as `img_shape`.
    """
    if len(img_shape) != 2:
        raise ValueError("Image can only have 2 dimensions.")

    rows, cols = img_shape
    x_start = 0
    x_end = cols - 1
    y_start, y_end = [int(y) for y in predictor([[x_start], [x_end]])]

    rr, cc = line(y_start, x_start, y_end, x_end)
    # the RANSAC estimations might fall out of the image range
    # make sure that the estimate is within the image dimensions
    rr = np.clip(rr, 0, rows - 1)

    mask = np.zeros(img_shape, dtype=np.bool_)
    mask[rr, cc] = True

    return mask


def measure_length(masked_image: np.ndarray, pixel_width: float) -> float:
    """Length of the masked image.

    Calculated as the distance between ends of the image masked with the centerline.

    Parameters:
        masked_image (np.ndarray):
            2D image where the centerline masked was applied
        pixel_width (float):
            Physical size of a pixel in the image.
    """
    if masked_image.ndim != 2:
        raise ValueError("Image can only have 2 dimensions.")

    nonzeros = np.argwhere(masked_image)
    nonzeros = np.sort(nonzeros, axis=0)

    line_start, line_end = nonzeros[0], nonzeros[-1]

    distance = np.sqrt(np.sum((line_start - line_end) ** 2))

    return distance * pixel_width


def centerline_dist(
    bin_image: np.ndarray, pixel_width=1.62, thres_rel=0.6, min_dist=5
) -> float:
    """Returns the centerline length estimation based on EDT maxima points."""
    if bin_image.ndim != 2:
        raise ValueError("Centerline distance can only be calculated on a 2D image.")
    coords = get_DT_maxima(bin_image, thres_rel, min_dist)

    if coords.shape[0] <= 2:
        print(f"Found a single point of maxima, cannot apply RANSAC.")
        return 0

    try:
        estimator = apply_ransac(coords)
    except ValueError:
        # if RANSAC can't find a consensus, it raises ValueError
        # in these cases, we return 0 to signal that no distance was found
        return 0

    mask = centerline_mask(bin_image.shape, estimator.predict)
    bin_image[~mask] = 0

    return measure_length(bin_image, pixel_width)


def view_centerline_dist(binary_image: np.ndarray, ax: Axes, thres_rel=0.6, min_dist=5):
    """Plot the centerline length estimation based on EDT maxima points."""
    coords = get_DT_maxima(binary_image, thres_rel, min_dist)

    if coords.shape[0] <= 2:
        print(f"Cound not find enough DT points of maxima.")
        return None

    y = coords.T[0]
    # sklearn expects x to have shape (n_samples, n_features), ie (N, 1)
    x = coords.T[1].reshape(-1, 1)

    try:
        estimator = apply_ransac(coords)
    except ValueError:
        print(f"Cound not find centerline for the given DT points.")
        return None

    rows, cols = binary_image.shape
    x_start = 0
    x_end = cols - 1
    y_start, y_end = [int(y) for y in estimator.predict([[x_start], [x_end]])]

    rr, cc = line(y_start, x_start, y_end, x_end)

    mask = binary_image[rr, cc] == 1

    rr_m, cc_m = rr[mask], cc[mask]
    rr = np.clip(rr, 0, rows - 1)

    inliers = estimator.inlier_mask_
    outliers = np.logical_not(inliers)

    ax.scatter(x[inliers], y[inliers], color="green")
    ax.scatter(x[outliers], y[outliers], color="orange")
    ax.imshow(binary_image)
    ax.plot(cc_m, rr_m, color="red", linewidth=1)
    ax.set_axis_off()
