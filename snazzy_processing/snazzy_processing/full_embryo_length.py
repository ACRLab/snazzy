import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.draw import ellipse
from skimage.exposure import equalize_hist
from skimage.filters import gaussian, threshold_multiotsu, threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, disk, remove_small_holes
import tifffile

from snazzy_processing import csv_handler


def binarize(image):
    """Returns a binary image with a single label, using a low threshold."""
    thr = threshold_triangle(image)
    thr -= 0.1 * thr
    bin_img = image > thr

    labels = label(bin_img, connectivity=1)
    # skips the background value
    largest_label = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    remove_small_holes(largest_label, 50, out=largest_label)
    binary_opening(largest_label, footprint=disk(5), out=largest_label)

    return largest_label


def binarize_low_embryo_background(image):
    """Returns a binary image with a single label, assuming that background values are _higher_ than non-VNC pixels in the embryo."""
    blurred_image = gaussian(image, sigma=2)
    thr = threshold_multiotsu(blurred_image, classes=3)
    img = np.digitize(blurred_image, thr)

    labels = label(img, connectivity=1, background=1)
    # skips the background value
    largest_label = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    filled_label = binary_fill_holes(largest_label, output=None)

    return filled_label


def length_from_regions_props(img, pixel_width=1.62):
    """Calculates length of a binary image containing a single label."""
    regions = regionprops(img.astype(np.uint8))
    if len(regions) > 1:
        print(
            f"WARN: expected a single label to calculate length, but got {len(regions)}."
        )
    return regions[0].axis_major_length * pixel_width


def read_and_preprocess_image(img_path, start=None, end=None, interval=100):
    if start is None and end is None:
        # try to sample frames from the start of the movie
        with tifffile.TiffFile(img_path) as tif:
            shape = tif.series[0].shape
        start = 0
        end = start + interval if shape[0] > start + interval else shape[0]

    img = tifffile.imread(img_path, key=range(start, end))
    img = np.average(img, axis=0)
    return equalize_hist(img)


def label_and_get_len(img, low_non_VNC):
    if low_non_VNC:
        bin_img = binarize_low_embryo_background(img)
    else:
        bin_img = binarize(img)
    return length_from_regions_props(bin_img)


def measure(img_path, low_non_VNC=False, start=None, end=None, interval=100):
    """Calculates the embryo length, based on a movie fragment.

    It's best to use an interval of 50 to 100 frames, and to pick frames
    of the middle part of the movie.

    The length is estimated using the major axis of the ellipse that
    matches the binary image. This is a valid estimate because the
    embryo shape is fairly regular and resembles an ellipse.
    """
    img = read_and_preprocess_image(img_path, start, end, interval)

    emb_length = label_and_get_len(img, low_non_VNC)

    return emb_length


def view_full_embryo_length(img, original_img):
    """Visualization of how the length is calculated for the full embryo.

    Expects a binary image containing a single label."""
    regions = regionprops(img.astype(np.uint8))

    props = regions[0]

    y0, x0 = props.centroid
    orientation = props.orientation
    maj_len = props.axis_major_length
    min_len = props.axis_minor_length

    x1 = x0 + math.sin(orientation) * 0.5 * maj_len
    y1 = y0 + math.cos(orientation) * 0.5 * maj_len
    x2 = x0 - math.sin(orientation) * 0.5 * maj_len
    y2 = y0 - math.cos(orientation) * 0.5 * maj_len

    rr, cc = ellipse(
        y0,
        x0,
        0.5 * min_len,
        0.5 * maj_len,
        shape=img.shape,
        rotation=orientation + np.deg2rad(90),
    )
    ell = np.zeros_like(img)
    ell[rr, cc] = 1

    fig, ax = plt.subplots(1, 2)

    ax[0].plot((x0, x1), (y0, y1), "-r", linewidth=2.5)
    ax[0].plot((x0, x2), (y0, y2), "-r", linewidth=2.5)
    ax[0].plot(x0, y0, ".g", markersize=2.5)
    ax[0].imshow(img)
    ax[0].imshow(ell, alpha=0.4)

    ax[1].imshow(original_img)
    plt.show()


def get_output_data(ids, lengths):
    return np.concatenate((ids[:, None], lengths[:, None]), axis=1)


def export_csv(ids, lengths, output_path):
    lengths = np.asarray(lengths)

    ids = np.asarray(ids)

    data = get_output_data(ids, lengths)

    csv_handler.write_file(output_path, data, ["ID", "full_length"])

    return True


def get_annotated_data(csv_path):
    """Reads annotated data from a csv file."""
    return csv_handler.read(csv_path)
