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


def get_activity(img, mask, mask_path=None):
    '''Returns the average of all pixels for each slice of the image.

    Accepts a single 2D mask, that will be applied to every slice, or a 3D
    mask, where each mask is applied for one slice of the image.
    `mask_path` will override the value passed in `mask`, and load it from 
    the path provided'''
    if mask_path:
        mask = np.load(mask_path)
    if mask.ndim == 2:
        try:
            mask = np.broadcast_to(mask, img.shape).astype(np.bool_)
        except ValueError:
            print(
                f"Mask of shape {mask.shape} cannot be applied to image of shape {img.shape}. The mask dimensions should match {img.shape[1:]}.")
            raise
        except Exception:
            raise
    if mask.shape != img.shape:
        downsampling_factor = img.shape[0]//mask.shape[0]
        mask = np.repeat(mask, downsampling_factor, axis=0)

    masked_img = ma.masked_array(img, mask)

    activity = masked_img.mean(axis=(1, 2))

    return activity


def plot_activity(img, struct, mask, mask_path=None, plot_diff=False, save=False, filename=None):
    activity = get_activity(img, mask, mask_path)
    activity_struct = get_activity(struct, mask, mask_path)

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
