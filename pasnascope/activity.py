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


def reflect_edges(signal, window_size=160):
    half = window_size//2
    return np.concatenate((signal[:half], signal, signal[-half:]))


def compute_baseline(signal, window_size=160, n_bins=20):
    '''Reflects edges so we can fit windows of size window_size for the 
    entire signal.'''
    expanded_signal = reflect_edges(signal, window_size)

    baseline = np.zeros_like(signal)
    for i, _ in enumerate(signal):
        window = expanded_signal[i:i+window_size]
        counts, bins = np.histogram(window, bins=n_bins)
        mode_bin_idx = np.argmax(counts)
        mode_bin_mask = np.logical_and(
            window > bins[mode_bin_idx], window <= bins[mode_bin_idx+1])
        window_baseline = np.mean(window[mode_bin_mask])
        baseline[i] = window_baseline
    return baseline


def get_dff(signal, baseline):
    return (signal - baseline)/baseline


def apply_mask(img, mask):
    '''Returns a np masked array, representing the masked image.

    Accepts a few combinations of dimensions: 2D img and 2D mask, 3D img and
    2D mask, and 3D img and 3D mask.'''
    if mask.ndim == 2 and img.ndim == 3:
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

    return masked_img


def get_activity(masked_img):
    return masked_img.mean(axis=(1, 2))


def ratiometric_activity(active, structural):
    return active / structural


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
