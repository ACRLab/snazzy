import numpy as np

from snazzy_processing import centerline, csv_handler


def measure_VNC_centerline(
    image,
    pixel_width=1.62,
    thres_rel=0.6,
    min_dist=5,
    outlier_thres=0.09,
    threshold_method="multiotsu",
):
    """Calculates the centerline distance for a 3D image."""
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


def predict_next(previous):
    """Estimates the next point by linear regression."""
    y = np.array(previous)
    x = np.arange(y.shape[0])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m * len(x) + c


def get_length_from_csv(file_path, columns=(1,)):
    """Reads CSV data as a nparray.

    Expects the lengths to be in actual metric units, instead of pixels."""
    return csv_handler.read(file_path, usecols=columns)


def get_output_data(length_data, downsampling, frame_interval):
    t = length_data.size
    time = np.arange(t) * frame_interval * downsampling

    return np.column_stack((time, length_data))


def export_csv(ids, lengths, output_dir, downsampling, frame_interval=6):
    csv_paths = [output_dir.joinpath(f"emb{id}.csv") for id in ids]

    for length_data, csv_path in zip(lengths, csv_paths):
        data = get_output_data(length_data, downsampling, frame_interval)
        csv_handler.write_file(csv_path, data, ["time", "length"])

    return True
