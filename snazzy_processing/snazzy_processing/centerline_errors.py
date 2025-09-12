import numpy as np

from snazzy_processing import utils


def percentual_err(measured, annotated):
    err = np.abs((measured - annotated)) / annotated
    return np.average(err), np.max(err)


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

    return pairs


def evaluate_CLE_global(measured, annotated):
    """Compares measured values against manually annotated data."""
    return {e: compare(measured[e], annotated[e]) for e in measured.keys()}
