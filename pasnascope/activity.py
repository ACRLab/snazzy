import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt


def measure_VNC(masks):
    vnc_lengths = []

    for mask in masks:
        regions = regionprops(mask.astype(np.uint8))
        props = regions[0]
        vnc_lengths.append(props['feret_diameter_max'])

    vnc_lengths = np.array(vnc_lengths)
    median_filter(vnc_lengths, size=19, output=vnc_lengths)

    x = np.arange(len(vnc_lengths))
    coef = np.polyfit(x, vnc_lengths, 1)
    poly1d = np.poly1d(coef)

    fig, ax = plt.subplots()
    ax.plot(vnc_lengths)
    ax.plot(x, poly1d(x), '--k')
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
