import math
from pathlib import Path

import numpy as np

from snazzy_processing import csv_handler


def apply_mask(img: np.ndarray, mask: np.ndarray):
    """Apply a mask to an image.

    Accepts a few combinations of dimensions: 2D img and 2D mask, 3D img and
    2D mask, and 3D img and 3D mask.

    Parameters:
        img (np.ndarray):
            A 2D or 3D np array representing an image.
        mask (np.ndarray):
            A 2D or 3D np array to mask the image.

    Returns:
        masked_img (np.ma.MaskedArray)
    """
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


def get_activity(masked_img: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """The median activity from a 3D MaskedArray.

    Parameters:
        masked_img (np.ma.MaskedArray):
            A 3D masked array of shape (T, Y, X).

    Returns:
        mean_activity (np.ma.MaskedArray):
            Mean activity value over time.
    """
    return masked_img.mean(axis=(1, 2))


def export_csv(ids: list[int], signals: list, output_dir: Path, frame_interval=6):
    """Write calculated activity as csv.

    Parameters:
        ids (list[int]):
            Embryo Ids used to name each csv file.
            Must match the indices of the signals list.
        signals (list):
            Activity signals.
        output_dir (Path):
            Path to write csv files.
        frame_interval (int):
            The interval of acquistion of frames in seconds.
    """
    signals = np.asarray(signals)

    csv_paths = [output_dir.joinpath(f"emb{id}.csv") for id in ids]

    data = add_timepoints(signals, frame_interval)

    csv_handler.write_files(csv_paths, data, ["time", "gcamp", "tomato"])


def add_timepoints(signals: np.ndarray, frame_interval=6) -> np.ndarray:
    """Add time information (in seconds) to signal data.

    Parameters:
        signals (np.ndarray):
            An array that represents act and struct channel signals.
        frame_interval (int):
            The interval of acquistion of frames in seconds.

    Returns:
        A np.ndarray of shape (N, t, 3).
    """
    if signals.ndim != 3:
        raise ValueError("Expected a 3D array with shape (N, t, 2)")
    N, t, _ = signals.shape
    time = np.arange(t) * frame_interval
    time = time[None, :, None]
    time = np.repeat(time, N, axis=0)

    return np.concatenate([time, signals], axis=2)
