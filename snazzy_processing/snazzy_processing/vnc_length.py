import csv

import numpy as np

from snazzy_processing import centerline


def measure_VNC_centerline(
    image, pixel_width=1.62, thres_rel=0.6, min_dist=5, outlier_thres=0.09
):
    """Calculates the centerline distance for a 3D image."""
    vnc_lengths = np.zeros(image.shape[0])
    for i, img in enumerate(image):
        bin_img = centerline.binarize(img)
        dist = centerline.centerline_dist(
            bin_img, pixel_width=pixel_width, thres_rel=thres_rel, min_dist=min_dist
        )
        if dist:
            vnc_lengths[i] = dist
        else:
            # Leave the dist as 0 and predict it once all values are calculated
            pass

    for i, curr in enumerate(vnc_lengths[10:], 10):
        prev = vnc_lengths[i - 1]
        diff = abs((curr - prev) / prev)
        if diff > outlier_thres:
            vnc_lengths[i] = predict_next(vnc_lengths[i - 10 : i - 1])

    return vnc_lengths


def predict_next(previous):
    """Estimates the next point by linear regression."""
    y = np.array(previous)
    x = np.arange(y.shape[0])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m * len(x) + c


def get_length_from_csv(
    file_path, columns=(1,), end=None, in_pixels=False, pixel_width=1.62
):
    """Reads CSV data as a nparray.

    Expects the lengths to be in actual metric units, instead of pixels."""
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1, usecols=columns)
    lengths = data
    if in_pixels:
        lengths *= pixel_width
    if end is None:
        return lengths
    else:
        return lengths[:end]


def export_csv(ids, vnc_lengths, output_dir, downsampling, frame_interval=6):
    """Generates a csv file with VNC Length data.

    Parameters:
        embryos: list of embryo names
        vnc_lengts: list of lists, where each nested list represents VNC lengths for a single embryo. Must have same length as the `embryos`.
        output: path to the output csv file.
        downsampling: interval used to calculate VNC lengths.
        frame_interval: time (seconds) between two image captures.
    """
    header = ["time", "length"]
    for id, lengths in zip(ids, vnc_lengths):
        output_path = output_dir.joinpath(f"emb{id}.csv")
        if output_path.exists():
            print(f"File {output_path.stem} already exists. Skipping..")
            continue
        with open(output_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t, length in enumerate(lengths):
                writer.writerow(format_csv_row(t, downsampling, frame_interval, length))
    return True


def format_csv_row(t, downsampling, frame_interval, length):
    """Columns in the csv file: [time, length]."""
    return [t * downsampling * frame_interval, f"{length:.2f}"]
