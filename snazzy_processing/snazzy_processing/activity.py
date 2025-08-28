import math

import numpy as np

from snazzy_processing import csv_handler


def apply_mask(img, mask):
    """Returns a np masked array, representing the masked image.

    Accepts a few combinations of dimensions: 2D img and 2D mask, 3D img and
    2D mask, and 3D img and 3D mask."""
    if mask.ndim == 2 and img.ndim == 3:
        try:
            mask = np.broadcast_to(mask, img.shape).astype(np.bool_)
        except ValueError:
            print(
                f"Mask of shape {mask.shape} cannot be applied to image of shape {img.shape}. The mask dimensions should match {img.shape[1:]}."
            )
            raise
    if mask.shape != img.shape:
        # match the mask shape to the image shape
        downsampling_factor = math.ceil(img.shape[0] / mask.shape[0])
        mask = np.repeat(mask, downsampling_factor, axis=0)
        mask = mask[: img.shape[0]]

    masked_img = np.ma.masked_array(img, mask)

    return masked_img


def get_activity(masked_img):
    return masked_img.mean(axis=(1, 2))


def get_output_data(signals, frame_interval=6):
    if signals.ndim != 3:
        raise ValueError("Expected a 3D array with shape (N, t, 3)")
    N, t, _ = signals.shape
    time = np.arange(t) * frame_interval
    time = time[None, :, None]
    time = np.repeat(time, N, axis=0)

    return np.concatenate([time, signals], axis=2)


def export_csv(ids, signals, output_dir, frame_interval=6):
    signals = np.asarray(signals)

    csv_paths = [output_dir.joinpath(f"emb{id}.csv") for id in ids]

    data = get_output_data(signals, frame_interval)

    csv_handler.write_files(csv_paths, data, ["time", "gcamp", "tomato"])

    return True
