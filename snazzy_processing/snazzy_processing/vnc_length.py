from pathlib import Path

import numpy as np

from snazzy_processing import centerline, csv_handler


def measure_VNC_centerline(
    image,
    pixel_width=1.62,
    thres_rel=0.6,
    min_dist=5,
    outlier_thres=0.09,
    threshold_method="multiotsu",
) -> np.ndarray:
    """Calculates the centerline distance for a 3D image.

    As RANSAC might give the wrong measurement sometimes, points with relative
    difference above `outlier_thres` are corrected by lin regression.

    Parameters:
        image (np.ndarray):
            3D matrix representing an image.
        pixel_width (float):
            Physica size of a pixel.
        thres_rel (float):
            Threshold value used to calculate centerline points.
        min_dist (float):
            Minimum distance used to find local points in the distance transform.
        outlier_thres (float):
            Maximum VNC length relative change between consecutive points.

    Returns:
        vnc_lenghts (np.ndarray):
            Array with VNC measurements.
    """
    vnc_lengths = np.zeros(image.shape[0])
    for i, img in enumerate(image):
        bin_img = centerline.binarize(img, threshold_method=threshold_method)
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


def predict_next(previous: np.ndarray) -> np.ndarray:
    """Estimates the next point by linear regression.

    Parameters:
        previous (np.ndarray):
            Array with previous points, use to predict the next point.
    """
    y = np.array(previous)
    x = np.arange(y.shape[0])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m * len(x) + c


def get_length_from_csv(file_path: Path, columns=(1,)):
    """Reads CSV data as a nparray.

    Expects the lengths to be in actual metric units, instead of pixels."""
    return csv_handler.read(file_path, usecols=columns)


def add_timepoints(
    length_data: np.ndarray, downsampling: int, frame_interval: int
) -> np.ndarray:
    """Add time information (in seconds) to VNC length data.

    Parameters:
        lendth_data (np.ndarray):
            An array of VNC length measurments.
        downsampled (int):
            Step interval used when calculating the VNC length.
        frame_interval (int):
            The interval of acquistion of frames in seconds.

    Returns:
        A np.ndarray of shape (N, t, 2).
    """
    t = length_data.size
    time = np.arange(t) * frame_interval * downsampling

    return np.column_stack((time, length_data))


def export_csv(ids, lengths, output_dir, downsampling, frame_interval=6):
    """Write vnc lengths as csv.

    Parameters:
        ids (list[int]):
            Embryo Ids used to name each csv file.
            Must match the indices of the signals list.
        lengths (list):
            VNC length measurements
        output_dir (Path):
            Path to write csv files.
        downsampling (int):
            Step interval used when calculating the VNC length.
        frame_interval (int):
            The interval of acquistion of frames in seconds.
    """
    csv_paths = [output_dir.joinpath(f"emb{id}.csv") for id in ids]

    for length_data, csv_path in zip(lengths, csv_paths):
        data = add_timepoints(length_data, downsampling, frame_interval)
        csv_handler.write_file(csv_path, data, ["time", "length"])
