from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from golf_analysis import Experiment, Trace
from peak_annot_parser import (
    PeakAnnotationParser,
    GroundTruthPointData,
    GroundTruthWindowData,
)

ANNOT_DIR = "./tests/assets/annotated_data/"


def load_data(annotated_path, exp_dir, annot_type):
    """Generates ground truth and calculated data in pairs.

    Parameters:
        annotated_path (Path):
            Path to annotated csv file.
        exp_dir (Path):
            Path to experiment dir.
        annot_type ('point' | 'window'):
            Type of GT data.

    Yields:
        comparison_data (tuple):
            (annotations, peak_idxes, exp_name, trace)
    """
    annotated_data = PeakAnnotationParser(annotated_path, annot_type)
    exp_names = annotated_data.get_exp_names()
    for exp_name in exp_names:
        annotations = annotated_data.get_annotation_by_exp_name(exp_name)
        emb_names = [a.emb_name for a in annotations]
        exp = Experiment(
            Path(exp_dir + "/" + exp_name),
            to_exclude=[x for x in range(99) if f"emb{x}" not in emb_names],
            first_peak_threshold=0,
            dff_strategy="local_minima",
        )

        for annot in annotations:
            emb_name = annot.emb_name
            trace = exp.get_embryo(emb_name).trace
            calc_idxes = get_peak_idxes(exp, emb_name)
            yield (annot, calc_idxes, exp_name, trace)


def abs_bound_distance(bound1, bound2):
    s1, e1 = bound1
    s2, e2 = bound2
    return abs(s1 - s2) + abs(e1 - e2)


def evaluate_bounds_detection(annot_bounds, calc_bounds, tolerance):
    matched = []
    unmatched_annotated = set(annot_bounds)
    unmatched_calculated = set(calc_bounds)
    visited = set()

    for ann in annot_bounds:
        close_candidates = [
            calc
            for calc in calc_bounds
            if abs_bound_distance(calc, ann) <= tolerance and calc not in visited
        ]
        if close_candidates:
            best_match = min(close_candidates, key=lambda c: abs_bound_distance(c, ann))
            matched.append((ann, best_match, abs_bound_distance(best_match, ann)))
            visited.add(best_match)
            unmatched_annotated.discard(ann)
            unmatched_calculated.discard(best_match)

    return {
        "matches": matched,
        "mean_error": sum(e[-1] for e in matched) / len(matched) if matched else None,
        "misses": list(unmatched_annotated),
        "false_positives": list(unmatched_calculated),
    }


def evaluate_peak_detection(annotated_peaks, calculated_peaks, tolerance):
    matched = []
    unmatched_annotated = set(annotated_peaks)
    unmatched_calculated = set(calculated_peaks)
    visited = set()

    for ann in annotated_peaks:
        close_candidates = [
            calc
            for calc in calculated_peaks
            if abs(calc - ann) <= tolerance and calc not in visited
        ]
        if close_candidates:
            best_match = min(close_candidates, key=lambda c: abs(c - ann))
            matched.append((ann, best_match, abs(best_match - ann)))
            visited.add(best_match)
            unmatched_annotated.discard(ann)
            unmatched_calculated.discard(best_match)

    return {
        "matches": matched,
        "mean_error": sum(e[-1] for e in matched) / len(matched) if matched else None,
        "misses": list(unmatched_annotated),
        "false_positives": list(unmatched_calculated),
    }


def get_comparison_results(annot_to_exp, annot_type):
    results = []
    for annot_file, exp_file in annot_to_exp.items():
        annot_path = ANNOT_DIR + annot_file
        exp_path = "./" + exp_file
        if annot_type == "point":
            for annot, calc_idxes, exp_name, _ in load_data(
                annot_path, exp_path, annot_type
            ):
                annot_idxes = annot.episode_idxes
                res = evaluate_peak_detection(annot_idxes, calc_idxes, tolerance=30)
                res["exp_name"] = exp_name
                results.append(res)
        elif annot_type == "window":
            for annot, _, exp_name, trace in load_data(
                annot_path, exp_path, annot_type
            ):
                annot_bounds = annot.episode_bounds
                trace.compute_peak_bounds(rel_height=0.99)
                calc_bounds = [tuple(b.tolist()) for b in trace.peak_bounds_indices]
                res = evaluate_bounds_detection(annot_bounds, calc_bounds, tolerance=90)
                res["exp_name"] = exp_name
                results.append(res)

    return results


def get_peak_idxes(exp: Experiment, emb_name):
    idxs = exp.get_embryo(emb_name).trace.peak_idxes
    return [int(idx) for idx in idxs]


def plot_single_trace(trace: Trace, annot: GroundTruthPointData):
    annot_peaks = annot.episode_idxes
    dff = trace.dff[: trace.trim_idx]
    annot_amps = [dff[p] for p in annot_peaks]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dff)
    ax.plot(annot_peaks, annot_amps, "b.", ms=12, label="annot")
    ax.plot(trace.peak_idxes, trace.peak_amplitudes, "r.", label="calc")
    ax.set_title(annot.emb_name)
    ax.legend()
    plt.show()


def plot_single_trace_bounds(trace: Trace, annot: GroundTruthWindowData):
    annot_bounds = annot.episode_bounds
    dff = trace.dff[: trace.trim_idx]
    trace.compute_peak_bounds(rel_height=0.99)
    calc_bounds = trace.peak_bounds_indices

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dff)
    for s, e in annot_bounds:
        ax.axvline(s, ls=":", color="b", alpha=0.5)
        ax.axvline(e, ls=":", color="b", alpha=0.5)
    for s, e in calc_bounds:
        ax.axvspan(s, e, facecolor="r", alpha=0.3)
    ax.set_title(annot.emb_name)
    plt.show()


def plot_all_traces(annot_to_exp, annot_type):
    for annot_file, exp_file in annot_to_exp.items():
        annot_path = ANNOT_DIR + annot_file
        exp_path = "./" + exp_file
        if annot_type == "point":
            for annot, _, _, trace in load_data(annot_path, exp_path, annot_type):
                plot_single_trace(trace, annot)
        elif annot_type == "window":
            for annot, _, _, trace in load_data(annot_path, exp_path, annot_type):
                plot_single_trace_bounds(trace, annot)


def plot_mean_error(results):
    exps_to_errs = {}

    for res in results:
        exp_name = res.get("exp_name", None)
        if exp_name is None:
            print(f"WARN: Invalid data, skipping..")
        if exp_name not in exps_to_errs:
            exps_to_errs[exp_name] = []

        exps_to_errs[exp_name].append(res.get("mean_error"))

    mean_errors = [sum(res) / len(res) for res in exps_to_errs.values()]
    exp_names = list(exps_to_errs.keys())
    x = list(range(len(mean_errors)))

    fig, ax = plt.subplots()
    ax.plot(x, mean_errors, "bo")
    ax.grid(True)
    ax.set_xticks(x, exp_names, rotation="vertical")
    ax.set_ylabel("# of frames")
    ax.set_title("Mean absolute error")

    plt.tight_layout()
    plt.show()


def plot_eval_results(results):
    rows = []
    for res in results:
        for k, v in res.items():
            if k == "exp_name" or k == "mean_error":
                continue
            rows.append(
                {
                    "exp_name": group_name(res.get("exp_name")),
                    "metric": k,
                    "count": len(v),
                }
            )

    df = pd.DataFrame(rows)

    sns.set_theme(style="darkgrid")
    ax = sns.catplot(
        data=df,
        x="exp_name",
        y="count",
        hue="metric",
        kind="bar",
        width=0.4,
        edgecolor="k",
    )
    ax.figure.suptitle("Calc vs GT peak positions")
    plt.show()


def plot_eval_results_bounds(results):
    rows = []
    for res in results:
        for ann, calc, d in res["matches"]:
            rows.append(
                {
                    "group_name": group_name(res["exp_name"]),
                    "exp_name": res["exp_name"],
                    "left_match": ann[0] - calc[0],
                    "right_match": ann[1] - calc[1],
                    "distance": d,
                }
            )

    df = pd.DataFrame(rows)
    df_long = df.melt(
        id_vars=["group_name", "exp_name"],
        value_vars=["left_match", "right_match", "distance"],
        var_name="metric",
        value_name="value",
    )

    sns.set_theme(style="darkgrid")
    ax = sns.stripplot(
        data=df_long,
        x="metric",
        y="value",
        hue="group_name",
        dodge=True,
        alpha=0.7,
        zorder=1,
    )

    sns.pointplot(
        data=df_long,
        x="metric",
        y="value",
        hue="group_name",
        dodge=0.532,
        errorbar=None,
        linewidth=0,
        color="k",
        markers=".",
        markersize=8,
        ax=ax,
        legend=False,
    )
    ax.figure.suptitle("Calc vs GT width positions")
    plt.show()


def group_name(exp_name):
    if exp_name.endswith("vglutdf"):
        return "vglutdf"
    elif exp_name.endswith("vgatdf"):
        return "vgatdf"
    elif exp_name.endswith("wt") or exp_name.endswith("25C"):
        return "wt"


def write_results(res_path, results):
    with open(res_path, "w+") as f:
        for res in results:
            for k, v in res.items():
                f.write(f"{k}: {v}" + "\n")
    print(f"Wrote data at {res_path}")


if __name__ == "__main__":

    annot_to_exp = {"VGAT-": "./data/vgat", "VGluT-": "./data/vglut", "WT": "./data"}

    # view results for peak boundaries:
    # results = get_comparison_results(annot_to_exp, "window")
    # plot_eval_results_bounds(results)
    # plot_all_traces(annot_to_exp, "window")

    # view results for peak data:
    # results = get_comparison_results(annot_to_exp, "point")
    # plot_eval_results(results)
    # plot_all_traces(annot_to_exp, "point")
