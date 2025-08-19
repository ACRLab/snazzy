from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread

from gowf_processing import find_hatching, utils, vnc_length


def get_random_files(path, n=5):
    files = list(path.iterdir())
    shuffle(files)
    return files[:n]


def percentual_err(measured, annotated):
    err = np.abs((measured - annotated)) / annotated
    return np.average(err), np.max(err)


def plot_err(measured, annotated, emb_name=None, interval=20):
    fig, ax = plt.subplots()
    x = np.arange(0, measured.shape[0] * interval, interval)
    ax.plot(x, measured, label="estimated")
    ax.plot(x, annotated, label="annotated")
    ax.legend()
    if emb_name is not None:
        fig.suptitle(emb_name)
    plt.show()


def compare(measured, annotated):
    # make sure both nparrays have the same size:
    min_len = min(measured.shape[0], annotated.shape[0])
    annotated = annotated[:min_len]
    measured = measured[:min_len]
    return percentual_err(measured, annotated)


def point_wise_err(measured, annotated):
    min_len = min(measured.shape[0], annotated.shape[0])
    annotated = annotated[:min_len]
    measured = measured[:min_len]
    return np.abs(measured - annotated) / annotated


def read_annotated(annotated_path, cols):
    return vnc_length.get_length_from_csv(annotated_path, columns=cols)


def measure_embryos(emb_files, interval, thres_rel=0.6, min_dist=5):
    measured = {k.stem: [] for k in emb_files}
    for emb in emb_files:
        print(f"Processing {emb.stem}..")
        hp = find_hatching.find_hatching_point(emb)
        hp -= hp % interval
        img = imread(emb, key=range(0, hp, interval))
        measured[emb.stem] = vnc_length.measure_VNC_centerline(
            img, thres_rel=thres_rel, min_dist=min_dist
        )
    return measured


def get_comparison_metrics(
    img_dir,
    annotated_files,
    LUT=None,
    cols=(1,),
    interval=20,
    thres_rel=0.6,
    min_dist=5,
):
    annotated_to_emb = get_matching_embryos(annotated_files, img_dir, LUT)
    embryos = annotated_to_emb.values()
    measured = measure_embryos(embryos, interval, thres_rel, min_dist)

    annotated = {e.stem: [] for e in embryos}
    for ann in annotated_files:
        calc = annotated_to_emb[ann.stem]
        annotated[calc.stem] = read_annotated(ann, cols)

    return measured, annotated


def get_matching_embryos(embryos, annotated, LUT=None):
    """Maps embryo files to corresponding annotated files, based on the LUT.

    The look-up table is only composed of numbers, so this function ports those numbers to the filename convention used here. Also makes sure that the embryos in the LUT actually exist as files.
    """
    pairs = {}

    if LUT is None:
        # if no LUT, embryos and annotated files match 1:1
        annotated_names = [ann.stem for ann in annotated]
        for emb in embryos:
            if emb.stem in annotated_names:
                ann_file = f"{emb.stem}.csv"
                pairs[emb.name] = ann_file
        return pairs

    annotated_filenames = [ann.name for ann in annotated]
    embryos_filenames = [emb.name for emb in embryos]

    for emb, ann in LUT.items():
        emb_file = utils.emb_name(emb, ch=2, ext="tif")
        ann_file = utils.emb_name(ann, ch=2, ext="csv")
        if emb_file in embryos_filenames and ann_file in annotated_filenames:
            pairs[emb_file] = ann_file
            # pairs[ann_emb] = img_dir.joinpath(f'{calc_emb}.tif')

    return pairs


def evaluate_CLE_global(measured, annotated):
    """Compares measured values against manually annotated data."""
    return {e: compare(measured[e], annotated[e]) for e in measured.keys()}


def load_files(emb_dir, annotated_dir):
    """Selects the matching files from both the emb dir and the annotated data dir."""
    annotated = sorted(list(annotated_dir.glob("*.csv")), key=utils.emb_number)
    selected = [e.stem for e in annotated]
    embs = [emb for emb in emb_dir.glob("*.tif") if emb.stem in selected]
    embs = sorted(embs, key=utils.emb_number)
    return embs, annotated


def match_names(annotated, name_LUT):
    """Gets the corresponding movie name for a list of annotated data, based
    on the mapping passed in `name_LUT`."""
    embs = []
    for a in annotated:
        a_idx = int(a.stem.split("-")[0][3:])
        e_idx = name_LUT.get(a_idx, None)
        if not e_idx:
            continue
        embs.append(f"emb{e_idx}-ch2.tif")
    return embs
