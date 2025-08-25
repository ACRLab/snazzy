import csv
import math

import numpy as np


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


def export_csv(ids, embryos, output_dir, frame_interval=6):
    """Generates a csv file to be use by the `pasna_fly` package.

    Parameters:
        embryos: list of embryos, where each embryo is represented by a list
        [active, structural]. Active and structural are lists with the
        measurements of the activity for each frame.
        output: path to the output csv file.
        frame_interval: time (seconds) between two image captures.
    """
    header = ["time", "gcamp", "tomato"]
    for id, embryo in zip(ids, embryos):
        with open(output_dir.joinpath(f"emb{id}.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for frame, (act, strct) in enumerate(zip(*embryo)):
                row = format_csv_row(frame, frame_interval, act, strct)
                writer.writerow(row)
    return True


def format_csv_row(frame, interval, act, strct):
    """Expected output for the csv file is: [Time(HH:mm:ss), id, sig1, sig2]"""
    return [frame * interval, f"{act:.2f}", f"{strct:.2f}"]
