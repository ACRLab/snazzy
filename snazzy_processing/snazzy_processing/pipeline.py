from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import shutil

import numpy as np
from tifffile import imread

from snazzy_processing import (
    activity,
    find_hatching,
    full_embryo_length,
    roi,
    utils,
    vnc_length,
)


def measure_vnc_length(embs_src, res_dir, interval):
    """Calculates VNC length for all embryos in a directory."""
    embs = sorted(embs_src.glob("*ch2.tif"), key=utils.emb_number)
    output = res_dir.joinpath("lengths")
    output.mkdir(parents=True, exist_ok=True)
    embs = [emb for emb in embs if not already_created(emb, output)]
    lengths = []
    ids = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calculate_length, emb, interval) for emb in embs]
        for future in as_completed(futures):
            id, vnc_len = future.result()
            ids.append(id)
            lengths.append(vnc_len)

    vnc_length.export_csv(ids, lengths, output, interval)
    return len(ids)


def already_created(emb, output):
    id = utils.emb_number(emb)
    output_path = output.joinpath(f"emb{id}.csv")
    return output_path.exists()


def calculate_length(emb, interval):
    id = utils.emb_number(emb.stem)
    hp = find_hatching.find_hatching_point(emb)
    hp -= hp % interval

    img = imread(emb, key=range(0, hp, interval))
    vnc_len = vnc_length.measure_VNC_centerline(img)
    return id, vnc_len


def measure_embryo_full_length(embs_src, res_dir, low_non_VNC=False):
    embs = sorted(embs_src.glob("*ch2.tif"), key=utils.emb_number)
    output = res_dir.joinpath("full-length.csv")
    full_lengths = []
    embryo_names = []
    if output.exists():
        print(f"The file {output.stem} already exists, and won't be overwritten.")
        return 0

    for emb in embs:
        embryo_names.append(emb.stem)
        full_lengths.append(full_embryo_length.measure(emb, low_non_VNC))

    # NOTE: temporary fix -> warn when measuments deviate too much from others
    # This happens sparsely due to the VNC position inside the embryo
    z_scores = np.abs(full_lengths - np.mean(full_lengths)) / np.std(full_lengths)
    threshold = 2
    outliers = np.where(z_scores > threshold)[0]
    for i in outliers:
        print(
            f"Embryo {embryo_names[i]} full length measurement should be manually checked."
        )

    full_embryo_length.export_csv(full_lengths, embryo_names, output)
    return len(full_lengths)


def calc_activities(embs_src, res_dir, window):
    """Calculate activity for active and structural channels"""
    active = sorted(embs_src.glob("*ch1.tif"), key=utils.emb_number)
    struct = sorted(embs_src.glob("*ch2.tif"), key=utils.emb_number)

    output = res_dir.joinpath("activity")
    output.mkdir(parents=True, exist_ok=True)

    active = [emb for emb in active if not already_created(emb, output)]
    struct = [emb for emb in struct if not already_created(emb, output)]

    embryos = []
    ids = []
    # NOTE: number of workers is limited here because it was crashing jupyter
    # on a machine with low RAM. More workers will result in faster processing
    # but also more RAM usage
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(calc_activity, act, stct, window)
            for act, stct in zip(active, struct)
        ]
        for future in as_completed(futures):
            id, signal = future.result()
            ids.append(id)
            embryos.append(signal)

    if embryos:
        activity.export_csv(ids, embryos, output)
    return len(ids)


def calc_activity(act, stct, window):
    id = utils.emb_number(act)
    if id != utils.emb_number(stct):
        raise ValueError(
            "Active and structural channels must come from the same embryo."
        )
    active_img = imread(act)
    struct_img = imread(stct)
    mask = roi.get_roi(struct_img, window=window)

    masked_active = activity.apply_mask(active_img, mask)
    masked_struct = activity.apply_mask(struct_img, mask)

    signal_active = activity.get_activity(masked_active)
    signal_struct = activity.get_activity(masked_struct)

    emb = [signal_active, signal_struct]
    return id, emb


def clean_up_files(embs_src, first_frames_path, tif_path):
    if embs_src:
        shutil.rmtree(embs_src)
    if first_frames_path.exists():
        first_frames_path.unlink(missing_ok=True)
    if tif_path and tif_path.exists():
        tif_path.unlink(missing_ok=True)


def log_params(path, **kwargs):
    with open(path, "+a") as f:
        f.write("Starting a new analysis...\n")
        f.write(f"{datetime.now()}\n")
        for name, value in kwargs.items():
            f.write(f"{name}: {value}\n")
        f.write("=" * 79 + "\n")
