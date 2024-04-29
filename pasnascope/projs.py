import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_holes, binary_opening, disk, binary_dilation


def proj_mask(input_image):
    image = input_image.copy()

    # binarize image:
    thr = threshold_otsu(image)
    binary_mask = image > thr
    image[...] = 0
    image[binary_mask] = 1
    image = image.astype(np.bool_)
    # label:
    labels = label(image, connectivity=2)
    largest_label = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    remove_small_holes(largest_label, 300, out=largest_label)
    binary_opening(largest_label, footprint=disk(5), out=largest_label)

    # create projections on both axes
    h_proj = np.sum(largest_label, axis=1)
    v_proj = np.sum(largest_label, axis=0)

    projections = h_proj[..., np.newaxis] * v_proj[np.newaxis, ...]

    # threshold the projection values to create a mask
    thr = threshold_otsu(projections)
    binary_mask = projections > thr
    projections[...] = 0
    projections[binary_mask] = 1

    labels = label(projections, connectivity=2)
    largest_label = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    # dilate with a rect to expand on X axis only
    binary_dilation(largest_label, footprint=np.ones(
        (1, 30)), out=largest_label)

    return np.logical_not(largest_label)


def proj_mask_histogram(input_image):
    image = input_image.copy()

    thr = threshold_otsu(image)
    binary_mask = image > thr
    image[...] = 0
    image[binary_mask] = 1
    image = image.astype(np.bool_)
    labels, num_labels = label(image, return_num=True, connectivity=2)
    largest_label = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    remove_small_holes(largest_label, 300, out=largest_label)
    binary_opening(largest_label, footprint=disk(5), out=largest_label)

    h_proj = np.sum(largest_label, axis=1)
    v_proj = np.sum(largest_label, axis=0)
    projections = h_proj[..., np.newaxis] * v_proj[np.newaxis, ...]

    fig, ax = plt.subplots()
    ax.hist(projections)
    plt.show()
