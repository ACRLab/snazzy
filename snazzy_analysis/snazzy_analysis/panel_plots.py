import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator



def plot_trace(
    ax,
    time,
    dff,
    bursts=None,
    minibursts=None,
    *,
    color=None,
):
    # x axis
    right_lim = ax.get_xlim()[0]
    left_lim = ax.get_xlim()[1]
    xinter = 60 if left_lim-right_lim > 60 else left_lim-right_lim
    min_tick = 0 if right_lim <=0 else right_lim
    ax.set_xticks(np.arange(min_tick, left_lim + xinter, xinter))
    ax.set_xlabel("Time (min)")

    # y axis
    top_lim = ax.get_ylim()[1]
    ax.set_ylabel("ΔF/F")
    ax.yaxis.set_label_coords(1.05, 0.35)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines['left'].set_visible(False)

    ax.plot(time, dff, color=color)

    if bursts:
        plt.plot(bursts, np.full(len(bursts), top_lim), "|", mew=1, markersize=8, color="black")
    if minibursts:
        plt.plot(minibursts, np.full(len(minibursts), top_lim), "|", mew=1, markersize=8, color="red")

    return ax

def plot_burst_points(
        ax,
        burst,
        offset,
        mean,
        yerr=None,
        max_burst=None,
        color=None
    ):

    # x axis
    xticks = np.arange(0, max_burst)
    burst_labels = xticks+1
    burst_labels = [i if i % 2 == 0 or i==1 else "" for i in burst_labels]
    ax.set_xticks(ticks=xticks, labels=burst_labels)

    ax.plot(burst+offset, mean, marker="o", markersize=1, color=color, linestyle="none")
    ax.spines['right'].set_visible(False)
    if yerr:
        ax.errorbar(burst+offset, mean, yerr=yerr, fmt="None", linewidth=1, capsize=2, capthick=1, color=color)
    return ax

def plot_bin_bars(
        ax,
        bins,
        offset,
        mean,
        width,
        yerr=None,
        color=None
    ):

    ax.bar(bins+offset, mean, width=width, color=color, alpha=0.5, edgecolor="black")
    ax.spines['right'].set_visible(False)
    if yerr:
        ax.errorbar(bins+offset, mean, yerr=yerr, fmt="None", linewidth=1, capsize=2, capthick=1, color=color)
    return ax

import pandas as pd

def save_with_summary(data: pd.DataFrame, summary, path: str, index: bool = False) -> None:
    """Save data to CSV, followed by mean/SEM/count of Count per Group and Bin."""
    with open(path, "w", newline="") as f:
        data.to_csv(f, index=index)
        f.write("\n")                      # blank separator row
        f.write("Summary statistics\n")
        summary.to_csv(f, index=False)     # reset_index already made Group/Bin columns