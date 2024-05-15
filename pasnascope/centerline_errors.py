from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread

from pasnascope import find_hatching, vnc_length


def get_random_files(path, n=5):
    files = list(path.iterdir())
    shuffle(files)
    return files[:n]


def percentual_err(measured, annotated):
    err = np.abs((measured - annotated)) / annotated
    return (np.average(err), np.max(err), np.argmax(err))


def plot_err(measured, annotated, emb_name=None, interval=20):
    fig, ax = plt.subplots()
    x = np.arange(0, measured.shape[0]*interval, interval)
    ax.plot(x, measured, label='estimated')
    ax.plot(x, annotated, label='annotated')
    ax.legend()
    if emb_name is not None:
        fig.suptitle(emb_name)
    plt.show()


def count_valleys(measured, thres=0.85):
    diffs = measured[1:] / measured[:-1]
    return np.count_nonzero(np.where(diffs <= thres))


def compare_against_annotated(measured, annotated):
    # make sure both nparrays have the same size:
    min_len = min(measured.shape[0], annotated.shape[0])
    annotated = annotated[:min_len]
    measured = measured[:min_len]

    num_valleys = count_valleys(measured)
    errors = percentual_err(measured, annotated)
    return [*errors,  num_valleys]


def read_annotated(annotated_path):
    return vnc_length.get_length_from_csv(annotated_path)


def evaluate_centerline_estimation(emb_files, annotated_dir, interval=20, thres_rel=0.6, min_dist=5):
    measured = {k.stem: [] for k in emb_files}
    errors = {k.stem: [] for k in emb_files}

    for emb in emb_files:
        print(f"Processing {emb.stem}..")
        hp = find_hatching.find_hatching_point(emb)
        hp -= hp % interval
        img = imread(emb, key=range(0, hp, interval))
        measured[emb.stem] = vnc_length.measure_VNC_centerline(
            img, thres_rel=thres_rel, min_dist=min_dist)

    for k, v in measured.items():
        annotated = read_annotated(annotated_dir.joinpath(f"{k}.csv"))
        errors[k] = compare_against_annotated(v, annotated)

    for k, v in errors.items():
        v[2] = v[2]*interval
        print(f"{k}: {v}")

    return errors
