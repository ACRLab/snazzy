import numpy as np

from snazzy_processing import centerline, csv_handler


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


def get_length_from_csv(file_path, columns=(1,)):
    """Reads CSV data as a nparray.

    Expects the lengths to be in actual metric units, instead of pixels."""
    return csv_handler.read(file_path, usecols=columns)


def get_output_data(lengths, downsampling, frame_interval):
    N, t = lengths.shape
    time = np.arange(t) * frame_interval * downsampling
    time = time[None, :, None]
    time = np.repeat(time, N, axis=0)

    lengths = lengths[:, :, None]

    return np.concatenate((time, lengths), axis=2)


def export_csv(ids, lengths, output_dir, downsampling, frame_interval=6):
    max_len = max(len(l) for l in lengths)
    padded = [np.pad(l, (0, max_len - len(l)), constant_values=0) for l in lengths]
    lengths = np.asarray(padded)

    csv_paths = [output_dir.joinpath(f"emb{id}.csv") for id in ids]

    data = get_output_data(lengths, downsampling, frame_interval)

    csv_handler.write_files(csv_paths, data, ["time", "length"])

    return True
