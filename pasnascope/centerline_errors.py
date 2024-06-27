from pathlib import Path
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread

from pasnascope import find_hatching, utils, vnc_length


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


def compare(measured, annotated):
    # make sure both nparrays have the same size:
    min_len = min(measured.shape[0], annotated.shape[0])
    annotated = annotated[:min_len]
    measured = measured[:min_len]

    num_valleys = count_valleys(measured)
    errors = percentual_err(measured, annotated)
    return [*errors,  num_valleys]


def point_wise_err(measured, annotated):
    min_len = min(measured.shape[0], annotated.shape[0])
    annotated = annotated[:min_len]
    measured = measured[:min_len]
    return (measured - annotated) / annotated


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
            img, thres_rel=thres_rel, min_dist=min_dist)
    return measured


def get_comparison_metrics(img_dir, annotated_files, LUT=None, cols=(1,), interval=20, thres_rel=0.6, min_dist=5):
    annotated_to_emb = get_matching_embryos(annotated_files, img_dir, LUT)
    embryos = annotated_to_emb.values()
    measured = measure_embryos(embryos, interval, thres_rel, min_dist)

    annotated = {k.stem: [] for k in annotated_files}
    for ann in annotated_files:
        calc = annotated_to_emb[ann.stem]
        annotated[calc.stem] = read_annotated(ann, cols)

    # for ann, k in zip(ann_files, measured.keys()):
    #     annotated[k] = read_annotated(ann, cols)
    return measured, annotated


def get_matching_embryos(annotated, img_dir, LUT=None):
    '''Maps annotated files to corresponding embryo images, based on the LUT.'''
    pairs = {}
    if LUT is None:
        for ann in annotated:
            pairs[ann.stem] = img_dir.joinpath(f'{ann.stem}.tif')
        return pairs

    annotated_file_names = [ann.stem for ann in annotated]

    for ann, calc in LUT.items():
        calc_emb = utils.emb_name(calc, ch=2)
        ann_emb = utils.emb_name(ann, ch=2)
        if ann_emb in annotated_file_names:
            pairs[ann_emb] = img_dir.joinpath(f'{calc_emb}.tif')

    return pairs


def evaluate_CLE_global(img_dir, annotated, LUT=None, cols=(1,), interval=20, thres_rel=0.6, min_dist=5):
    annotated_to_emb = get_matching_embryos(annotated, img_dir, LUT)
    measured = measure_embryos(
        annotated_to_emb.values(), interval, thres_rel, min_dist)

    errors = {k.stem: [] for k in annotated_to_emb.values()}

    for ann in annotated:
        annotated = read_annotated(ann, cols)
        calc = annotated_to_emb[ann.stem]
        errors[calc.stem] = compare(measured[calc.stem], annotated)

    for v in errors.values():
        v[2] = v[2]*interval

    return errors


def load_files(emb_dir, annotated_dir):
    '''Selects the matching files from both the emb dir and the annotated data dir.'''
    annotated = sorted(list(annotated_dir.glob('*.csv')), key=utils.emb_number)
    selected = [e.stem for e in annotated]
    print(selected)
    embs = [emb for emb in emb_dir.glob('*.tif') if emb.stem in selected]
    embs = sorted(embs, key=utils.emb_number)
    return embs, annotated


def match_names(annotated, name_LUT):
    '''Gets the corresponding movie name for a list of annotated data, based
    on the mapping passed in `name_LUT`.'''
    embs = []
    for a in annotated:
        a_idx = int(a.stem.split('-')[0][3:])
        e_idx = name_LUT.get(a_idx, None)
        if not e_idx:
            continue
        embs.append(f"emb{e_idx}-ch2.tif")
    return embs
