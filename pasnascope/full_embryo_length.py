import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import ellipse
from skimage.exposure import equalize_hist
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes, binary_opening, disk
from tifffile import imread

from pasnascope import utils


def binarize(image):
    '''Returns a binary image with a single label, using a low threshold.'''
    thr = threshold_triangle(image)
    thr -= 0.1*thr
    bin_img = image > thr

    labels = label(bin_img, connectivity=1)
    largest_label = labels == np.argmax(
        np.bincount(labels.flat)[1:])+1

    remove_small_holes(largest_label, 50, out=largest_label)
    binary_opening(largest_label, footprint=disk(5), out=largest_label)

    return largest_label


def length_from_regions_props(img, pixel_width=1.62):
    '''Calculates length of a binary image containing a single label.'''
    regions = regionprops(img.astype(np.uint8))
    if len(regions) > 1:
        print(
            f'WARN: expected a single label to calculate length, but got {len(regions)}.')
    return regions[0].axis_major_length*pixel_width


def measure(img_path, start=0, end=100):
    '''Calculates the embryo length, based on the first 100 slices.

    The length is estimated using the major axis of the ellipse that
    matches the binary image. This is a valid estimate because the
    embryo shape is fairly regular and resembles an ellipse.
    '''
    img = imread(img_path, key=range(start, end))
    img = np.average(img, axis=0)
    img = equalize_hist(img)
    bin_img = binarize(img)
    return length_from_regions_props(bin_img)


def export_csv(lengths, embryo_names,  output):
    '''Generates a csv file with embryo full length data.

    Parameters:
        lengts: list of lists, where each nested list represents VNC lengths for a single embryo. Must have same length as the `embryos`.
        embryo_names: list of embryo names
        output: path to the output csv file.
    '''
    header = ['emb_name', 'ID', 'full_length']
    if output.exists():
        print(
            f"Warning: The file `{output.stem}` already exists. Select another file name or delete the original file.")
        return False
    with open(output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for emb_name, length in zip(embryo_names, lengths):
            id = utils.emb_number(emb_name)
            writer.writerow((emb_name, id, length))
    return True


def view_full_embryo_length(img, original_img):
    '''Visualization of how the length is calculated for the full embryo.

    Expects a binary image containing a single label.'''
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

    rr, cc = ellipse(y0, x0, 0.5*min_len, 0.5*maj_len,
                     shape=img.shape, rotation=orientation + np.deg2rad(90))
    ell = np.zeros_like(img)
    ell[rr, cc] = 1

    fig, ax = plt.subplots(1, 2)

    ax[0].plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax[0].plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax[0].plot(x0, y0, '.g', markersize=2.5)
    ax[0].imshow(img)
    ax[0].imshow(ell, alpha=0.4)

    ax[1].imshow(original_img)
    plt.show()


def get_annotated_data(csv_path):
    '''Reads annotated data from a csv file.'''
    return np.genfromtxt(csv_path, delimiter=',', skip_header=1)
