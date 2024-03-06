import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt


def measure_VNC(masks):
    '''Returns the max feret diameters for a set of binary images.'''
    vnc_lengths = []

    for mask in masks:
        regions = regionprops(mask.astype(np.uint8))
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

    x = np.arange(len(vnc_lengths))
    coef = np.polyfit(x, vnc_lengths, 1)
    poly1d = np.poly1d(coef)

    fig, ax = plt.subplots()
    ax.plot(vnc_lengths)
    ax.plot(x, poly1d(x), '--k')
    fig.suptitle('VNC length (feret diameter) over time')
    plt.show()


def plot_activity(img, struct, mask, mask_path=None):
    '''mask_path will override the value passed in mask, and load it from the
    path provided'''
    if mask_path:
        mask = np.load(mask_path)
    if mask.ndim == 2:
        # TODO: handle ValueError when shape cannot be broadcasted to
        mask = np.broadcast_to(mask, img.shape)

    img[mask] = 0
    struct[mask] = 0

    activity = np.average(img, axis=(1, 2))
    activity = activity/np.max(activity)
    activity_struct = np.average(struct, axis=(1, 2))
    activity_struct = activity_struct/np.max(activity_struct)

    diff = activity-activity_struct

    return diff
