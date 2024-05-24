import csv
import math
import random
from pathlib import Path

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from pasnascope import utils


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
        # match the mask shape to the image shape
        downsampling_factor = math.ceil(img.shape[0]/mask.shape[0])
        mask = np.repeat(mask, downsampling_factor, axis=0)
        mask = mask[:img.shape[0]]

    masked_img = ma.masked_array(img, mask)

    return masked_img


def get_activity(masked_img):
    return masked_img.mean(axis=(1, 2))


def ratiometric_activity(active, structural):
    return active / structural


def export_csv(embryos, output, frame_interval=6):
    '''Generates a csv file to be use by the `pasna_fly` package.

    Parameters:
        embryos: list of embryos, where each embryo is represented by a list
        [active, structural]. Active and structural are lists with the 
        measurements of the activity for each frame.
        output: path to the output csv file.
        frame_interval: time (seconds) between two image captures.
    '''
    header = ['Time [h:m:s]', 'ROI ID', 'Intensity(GFP)', 'Intensity(TRITC)']
    if output.exists():
        print(
            f"Warning: The file `{output.stem}` already exists. Select another file name or delete the original file.")
        return False
    with open(output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for emb_id, embryo in enumerate(embryos, 1):
            for frame, (act, strct) in enumerate(zip(*embryo)):
                row = format_csv_row(
                    frame, frame_interval, emb_id, act, strct)
                writer.writerow(row)
    return True


def format_csv_row(frame, interval, id, act, strct):
    '''Expected output for the csv file is: [Time(HH:mm:ss), id, sig1, sig2]'''
    return [utils.format_seconds(frame*interval), id, f"{act:.2f}", f"{strct:.2f}"]


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


if __name__ == '__main__':
    act = np.array([random.randint(0, 100) for _ in range(100)])
    struct = np.array([random.randint(0, 100) for _ in range(100)])
    act2 = np.array([random.randint(0, 100) for _ in range(100)])
    struct2 = np.array([random.randint(0, 100) for _ in range(100)])
    embryos = [[act, struct], [act2, struct2]]

    file_path = Path.cwd().joinpath('results', 'preview.csv')
    export_csv(embryos, file_path)
