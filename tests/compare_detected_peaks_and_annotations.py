from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pasna_analysis import Experiment
from peak_annot_parser import PeakAnnotationParser

ANNOT_DIR = "./tests/assets/annotated_data/"


def load_data(annotated_path, exp_dir):
    annotated_data = PeakAnnotationParser(annotated_path)
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
            trace = exp.embryos[emb_name].trace
            calc_idxes = get_peak_idxes(exp, emb_name)
            yield (annot, calc_idxes, exp_name, trace)


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


def get_comparison_results(annot_to_exp):
    results = []
    for annot_path, exp_path in annot_to_exp.items():
        for annot, calc_idxes, exp_name, _ in load_data(
            ANNOT_DIR + annot_path, "./" + exp_path
        ):
            annot_idxes = annot.episode_idxes
            res = evaluate_peak_detection(annot_idxes, calc_idxes, tolerance=30)
            res["exp_name"] = exp_name
            results.append(res)

    return results


def get_peak_idxes(exp: Experiment, emb_name):
    idxs = exp.embryos[emb_name].trace.peak_idxes
    return [int(idx) for idx in idxs]


def plot_single_trace(trace, annot):
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


def plot_all_traces(annot_to_exp):
    for annot_file, exp_file in annot_to_exp.items():
        annot_path = ANNOT_DIR + annot_file
        exp_path = "./" + exp_file
        for annot, _, _, trace in load_data(annot_path, exp_path):
            plot_single_trace(trace, annot)


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
    annot_to_exp = {"WT": "data", "VGAT-": "data/vgat", "VGluT-": "data/vglut"}

    results = get_comparison_results(annot_to_exp)
    plot_eval_results(results)

    # plot_all_traces(annot_to_exp)

    # results = get_comparison_results(annot_to_exp)
    # plot_mean_error(results)
