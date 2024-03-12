import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.ndimage import median_filter


def measure_VNC(masks):
    '''Returns the max feret diameter for a grou of binary images.

    The VNC length is measured indirectly, based on the ROI mask length.

    Args:
        masks (nparray): a mask of a group of images, as a 3D nparray
    '''
    vnc_lengths = []

    for mask in masks:
        # Values that should be ignored in the mask are masked as True
        # So, we need to flip the values to use here:
        regions = regionprops(np.logical_not(mask).astype(np.uint8))
        # TODO: handle cases where no region is found
        # this can happen because sometimes mask is None
        if len(regions) == 0:
            continue
        props = regions[0]
        vnc_lengths.append(props['feret_diameter_max'])

    return np.array(vnc_lengths)


def plot_VNC_measures(masks):
    vnc_lengths = measure_VNC(masks)
    median_filter(vnc_lengths, size=19, output=vnc_lengths)

    x = np.arange(vnc_lengths.size)
    coef = np.polyfit(x, vnc_lengths, 1)
    poly1d = np.poly1d(coef)

    fig, ax = plt.subplots()
    ax.plot(vnc_lengths)
    ax.plot(x, poly1d(x), '--k')
    fig.suptitle('VNC length (feret diameter) over time')
    plt.show()


def get_activity(img, struct, mask, mask_path=None):
    '''Calculates the activity difference between the img and struct channels.

    `mask_path` will override the value passed in `mask`, and load it from the
    path provided'''
    if mask_path:
        mask = np.load(mask_path)
    if mask.ndim == 2:
        # TODO: handle ValueError when shape cannot be broadcasted to
        mask = np.broadcast_to(mask, img.shape).astype(np.bool_)

    masked_img = ma.masked_array(img, mask)
    masked_struct = ma.masked_array(struct, mask)

    activity = masked_img.mean(axis=(1, 2))
    activity_struct = masked_struct.mean(axis=(1, 2))

    return activity, activity_struct


def plot_activity(img, struct, mask, mask_path=None, plot_diff=False, save=False, filename=None):
    activity, activity_struct = get_activity(img, struct, mask, mask_path)

    fig, ax = plt.subplots()
    if plot_diff:
        diff = activity - activity_struct
        ax.plot(diff)
    else:
        ax.plot(activity_struct, label='structural')
        ax.plot(activity, label='active')
        ax.legend()

    if save and filename:
        fig.suptitle(filename)
        plt.savefig(f'../results/activity/{filename}.png')

    plt.show()
