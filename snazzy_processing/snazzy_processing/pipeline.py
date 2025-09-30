from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
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


def measure_vnc_length(
    embs_src: Path, res_dir: Path, downsampling: int, threshold_method="multiotsu"
):
    """Calculates VNC length for all embryos in a directory and saves as CSV.

    Parameters:
        embs_src (Path):
            Directory with embryo files.
        res_dir (Path):
            Path to the results directory, where the csv files will be saved.
        downsampling (int):
            Step size to calculate ROI lengths.
        threshold_method ('mulitotsu' | 'otsu'):
            Threshold method used to calculate the ROI.
            Refer to `centerline.binarize` for more details.
            Defaults to 'multiotsu'.
    """
    embs = sorted(embs_src.glob("*ch2.tif"), key=utils.emb_number)
    output_dir = res_dir.joinpath("lengths")
    embs = [emb for emb in embs if not already_created(emb, output_dir)]

    if not embs:
        return 0

    lengths = []
    ids = []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(calculate_length, emb, downsampling, threshold_method)
            for emb in embs
        ]
        for future in as_completed(futures):
            id, vnc_len = future.result()
            ids.append(id)
            lengths.append(vnc_len)

    output_dir.mkdir(parents=True, exist_ok=True)
    vnc_length.export_csv(ids, lengths, output_dir, downsampling)
    return len(ids)


def already_created(emb: str, output: Path) -> bool:
    """Check if the embryo was already processed based on tif file name.

    Parameters:
        emb (str):
            File name, for example emb2-ch2.tif.
        output (Path):
            Path where to search if csv file was created.
    """
    if not output.exists():
        return False
    id = utils.emb_number(emb)
    output_path = output.joinpath(f"emb{id}.csv")
    return output_path.exists()


def calculate_length(emb: Path, downsampling: int, threshold_method="multiotsu"):
    """Calculates VNC length for an embryo.

    Parameters:
        emb (Path):
            Path to emb tif file.
        downsampling (int):
            Step size to calculate ROI lengths.
        threshold_method ('mulitotsu' | 'otsu'):
            Threshold method used to calculate the ROI.
            Refer to `centerline.binarize` for more details.
            Defaults to 'multiotsu'.
    """
    id = utils.emb_number(emb.stem)
    hp = find_hatching.find_hatching_point(emb)
    hp -= hp % downsampling

    key = list(range(0, hp, downsampling))
    img = imread(emb, key=key)
    vnc_len = vnc_length.measure_VNC_centerline(img, threshold_method=threshold_method)
    return id, vnc_len


def measure_embryo_full_length(embs_src: Path, res_dir: Path, low_non_VNC=False) -> int:
    """Calculates full embryo length for all embryos in a directory.

    Parameters:
        embs_src (Path):
            Directory with embryo files.
        res_dir (Path):
            Path to the results directory, where the csv files will be saved.
        low_non_VNC (bool):
            Pass `False` if the signal in the vnc is lower than in the rest of the embryo.
            Defaults to `True`.

    Returns:
        count (int):
            Number of embryos that were processed
    """
    embs = sorted(embs_src.glob("*ch2.tif"), key=utils.emb_number)
    output = res_dir.joinpath("full-length.csv")
    full_lengths = []
    ids = []

    if output.exists():
        print(f"The file {output.stem} already exists, and won't be overwritten.")
        return 0

    for emb in embs:
        ids.append(utils.emb_number(emb.stem))
        full_lengths.append(full_embryo_length.measure(emb, low_non_VNC))

    # Warn when measuments deviate too much from others
    # Can happen due to odd VNC positions or wrong bounding boxes
    if len(full_lengths) > 1:
        z_scores = np.abs(full_lengths - np.mean(full_lengths)) / np.std(full_lengths)
        threshold = 2
        outliers = np.where(z_scores > threshold)[0]
        for i in outliers:
            print(
                f"Embryo {ids[i]} full length measurement should be manually checked."
            )

    full_embryo_length.export_csv(ids, full_lengths, output)
    return len(full_lengths)


def calc_activities(embs_src: Path, res_dir: Path, window: int):
    """Calculates activity signal for all embryos in a directory.

    Parameters:
        embs_src (Path):
            Directory with embryo files.
        res_dir (Path):
            Path to the results directory, where the csv files will be saved.
        window (int):
            Step interval to take each measurement.

    Returns:
        count (int):
            Number of embryos that were processed
    """
    active = sorted(embs_src.glob("*ch1.tif"), key=utils.emb_number)
    struct = sorted(embs_src.glob("*ch2.tif"), key=utils.emb_number)

    output_dir = res_dir.joinpath("activity")

    active = [emb for emb in active if not already_created(emb, output_dir)]
    struct = [emb for emb in struct if not already_created(emb, output_dir)]

    if not active or not struct:
        return 0

    signals = []
    ids = []
    # NOTE: number of workers is limited here because it was crashing jupyter
    # on a machine with low RAM. More workers will result in faster processing
    # but also more RAM usage
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(calc_activity, act, stct, window)
            for act, stct in zip(active, struct)
        ]
        for future in as_completed(futures):
            id, signal = future.result()
            ids.append(id)
            signals.append(signal)

    output_dir.mkdir(parents=True, exist_ok=True)
    if signals:
        activity.export_csv(ids, signals, output_dir)
    return len(ids)


def calc_activity(act: Path, stct: Path, window: int) -> tuple[int, np.ma.MaskedArray]:
    """Calculates activity signal for an embryo.

    Parameters:
        act (Path):
            Directory with embryo files.
        stct (Path):
            Path to the results directory, where the csv files will be saved.
        window (int):
            Step interval to take each measurement.

    Returns:
        count (int):
            Number of embryos that were processed
    """
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

    signals = np.ma.column_stack((signal_active, signal_struct))

    return id, signals


def clean_up_files(
    embs_src: Path | None, first_frames_path: Path | None, tif_path: Path | None
):
    """Remove files generated by the pipeline.

    Passing None to any of the parameters if you want to keep them.

    Parameters:
        embs_src (Path):
            Path where the individual movies are saved.
        first_frames_path (Path):
            Path where the image with first frames is saved.
        tif_path (Path):
            Path to the converted tif file.
    """
    if embs_src:
        shutil.rmtree(embs_src)
    if first_frames_path.exists():
        first_frames_path.unlink(missing_ok=True)
    if tif_path and tif_path.exists():
        tif_path.unlink(missing_ok=True)


def log_params(output_path: Path, **kwargs):
    """Write analysis parameters.

    Parameters:
        output_path (Path):
            Path to write.
        **kwargs (dict):
            All parameters to be saved.
    """
    with open(output_path, "+a") as f:
        f.write("Starting a new analysis...\n")
        f.write(f"{datetime.now()}\n")
        for name, value in kwargs.items():
            f.write(f"{name}: {value}\n")
        f.write("=" * 79 + "\n")
