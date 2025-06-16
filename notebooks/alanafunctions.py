import numpy as np
import scipy.signal as spsig

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

from pathlib import Path
from pasna_analysis import Experiment, ExperimentConfig, Group

def load_sample_data():
    """
    load_sample_data gets wildtype data to test functions

    """
    wt_folder = '25C'
    wt_config  = {
        '20240919_25C': ExperimentConfig(first_peak_threshold=30, to_exclude=[5,6,9,10,11,16]),
    }

    wt_experiments = {}
    for exp, config in wt_config.items():
        exp_path = Path.cwd().joinpath('data', wt_folder, exp)
        print('path: ', exp_path)
        wt_experiments[exp] = Experiment(exp_path, config.first_peak_threshold, config.to_exclude, dff_strategy='local_minima', hatches=True)

    wt = Group('WT', wt_experiments)

    groups = [wt]

    group = groups[0]
    exp = group.experiments.get('20240919_25C') 
    embryos = list(exp.embryos.values())
    emb = embryos[14]

    all_embryos = []
    for exp in group.experiments.values():
        all_embryos.extend(exp.embryos.values())

    return group, exp, all_embryos, emb

def preprocess_dff(emb, duration=3600, onset_pad=300):
    """
    preprocess_embryo standardizes the trace to a given duration.
    If embryo hatches, only data from onset to hatching are included.
    If activity after onset does not continue for the entire duration,
    the dff is padded. The resulting time and dff must be the same length. 
    
    Args:
        emb (Embryo): The embryo to process.
        duration (int): Target num of frames for time and dff. 
        onset_pad (int): Num of frames to include before onset.
    
    Returns:
        time_processed (arr): Trimmed/extended time. 
        dff_processed (arr): Trimmed/extended dff.
    """

    time = emb.activity[:, 0] / 60 # covert to mins
    trace = emb.trace
    dff = trace.dff

    # trim dff from onset to hatching
    onset = trace.peak_bounds_indices[0][0]
    start_index = onset - onset_pad
    dff = dff[start_index:trace.trim_idx]

    # trim/extend dff to duration
    if duration > len(dff):
        pad_size = duration - len(dff)
        dff_processed = np.array(list(dff) + [None] * pad_size, dtype=object)
    else:
        dff_processed = dff[0:duration]

    increment = time[1] - time[0]
    time_processed = np.arange(0,duration/10, increment)

    return emb.name, time_processed, dff_processed

def preprocess_multi_dffs(embs, duration=3600, onset_pad=300):
    all_names = []
    all_time_processed = []
    all_dff_processed = []
    for emb in embs:
        name, time, dff = preprocess_dff(emb, duration=3600, onset_pad=300)
        all_names.append(name)
        all_time_processed.append(time)
        all_dff_processed.append(dff)
    return all_names, all_time_processed, all_dff_processed

def calculate_STFT(dff, fs=1/6, fft_size=600, noverlap=450):
    """
    calculate_STFT calculates the Short Time Fourier Transform
    for a single dff. It replaces None values with 1e-11. 
    
    Args:
        dff (arr): Preprocessed dff.
        fs (float): Sampling rate.
        fft_size (int): Num frames in each segment.
        noverlap (int): Num frames to overlap between segments. 
    
    Returns:
        f (arr): STFT frequency bins.
        t (arr): STFT time columns.
        magnitude (array): STFT magnitude (excludes phase). 
    """
    dff = np.where(dff == None, 1e-11, dff).astype(float)
    f, t, Zxx = spsig.stft(
        dff, fs, nperseg=fft_size, noverlap=noverlap
    )
    magnitude = abs(Zxx)
    return f, t, magnitude


def calculate_multi_STFTs(dffs, fs=1/6, fft_size=600, noverlap=450):
    """
    calculate_multi_STFTs calculates the Short Time Fourier Transforms
    for a group of preprocessed dffs. 
    
    Args:
        dff (arr): Preprocessed dff.
        fs (float): Sampling rate.
        fft_size (int): Num frames in each segment.
        noverlap (int): Num frames to overlap between segments. 
    
    Returns:
        f (arr): STFT frequency bins.
        t (arr): STFT time columns.
        magnitudes (arr): STFT average magnitude (excludes phase). 
    """
    freqs = []
    times = []
    Zxxs = []
    for dff in dffs:
        if dff is None or len(dff) == 0:
            print('dff invalid')
            continue
        f, t, Zxx = calculate_STFT(dff, fs=fs, fft_size=fft_size, noverlap=noverlap)
        freqs.append(f)
        times.append(t)
        Zxxs.append(Zxx)
    
    f = freqs[0]
    t =  times[0]

    Zxxs = np.array(Zxxs)
    magnitudes = np.abs(Zxxs)
    return f, t, magnitudes

def calculate_avg_magnitude(magnitudes):
    return np.mean(magnitudes, axis=0)

subplots_rc = {
    'figure.frameon': False,

    'lines.linewidth': 2,

    'font.family': 'Arial',
    'font.size': 10,

    'xtick.major.size': 10, # length
    'xtick.major.width': 2,
    'ytick.major.size': 10, # length
    'ytick.major.width': 2,

    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelpad': 10,
    'axes.linewidth': 2.5
}

trace_rc = {
    'figure.figsize': (35, 8),
    'figure.frameon': False,

    'lines.linewidth': 4,

    'font.family': 'Arial',
    'font.size': 40,

    'xtick.major.size': 10, # length
    'xtick.major.width': 2,
    'ytick.major.size': 10, # length
    'ytick.major.width': 2,

    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelpad': 10,
    'axes.linewidth': 2.5
}

spec_rc = {
    'figure.figsize': (35, 8),
    'figure.frameon': False,

    'font.family': 'Arial',
    'font.size': 40,

    'axes.spines.top': False,
    'axes.spines.right': False,

    'xtick.major.size': 10, # length
    'xtick.major.width': 2,
    'ytick.major.size': 10, # length
    'ytick.major.width': 2,
    'axes.labelpad': 10,
    'axes.linewidth': 2.5
}

def plot_raw_signals(embryos, rc):
    with plt.rc_context(rc):
        fig_width = rc['figure.figsize'][0]
        fig, axes = plt.subplots(len(embryos), 3, figsize=(fig_width, 2.5 * (len(embryos))), gridspec_kw={'width_ratios': [2, 10, 10]})
        fig.subplots_adjust(top=0.99, bottom=0.05, hspace=0.9, wspace=0.15)

        for axes, emb in zip(axes, embryos):
            label, left, right = axes

            time = emb.activity[:, 0]/60
            active = emb.trace.active
            struct = emb.trace.struct
            left.plot(time, active, color='green')
            right.plot(time, struct, color='firebrick')

            # x axis
            left.set_xlabel("Time (mins)")
            time_min = time[0]
            time_max = time[len(time)-1]
            minute_ticks = np.arange(time_min, time_max + 60, 60, int)
            left.set_xticks(minute_ticks, minute_ticks)
            right.set_xlabel("Time (mins)")
            right.set_xticks(minute_ticks, minute_ticks)

            # y axis
            left.set_ylabel("Intensity")
            right.set_ylabel("Intensity")

            # label
            label.axis('off')
            label.text(0, 0.5, f"{emb.name}", 
                       va='center', ha='left', 
                       fontsize=rc['font.size'] * 1.5, 
                       rotation=90,
                       transform=label.transAxes)
    
    plt.show()


def plot_trace(time, dff, rc, color='black', xmin=0, xmax=360, xinterval=60, ymin=-0.1, ymax=1):
    with plt.rc_context(rc):
        fig = plt.figure()
        plt.plot(time, dff, color=color)
        
        # x axis
        plt.xlabel("Time (mins)")
        plt.xlim(xmin, xmax)
        minute_ticks = np.arange(xmin, xmax + xinterval, xinterval, int)
        plt.xticks(minute_ticks, minute_ticks)
        fig.tight_layout()

        # y axis
        plt.ylabel("ΔF/F")
        plt.ylim(ymin, ymax)
        increment = 1
        dff_ticks = np.arange(0, ymax+increment, increment)
        plt.yticks(dff_ticks, dff_ticks)
        fig.tight_layout()
        plt.show()

def plot_spec(f, t, mag, mymap, rc, display_colorbar=True, xmin=0, xmax=360, ymin=0, ymax=0.03):
    with plt.rc_context(rc):
        fig = plt.figure()
        spec = plt.pcolormesh(
            t,
            f,
            mag,
            cmap=mymap,
            shading="nearest",
            snap=True,
            norm=colors.LogNorm(vmin=0.001, vmax=0.1)
        )

        # x axis
        plt.xlabel("Time (mins)", y=-0.25, labelpad=20)
        # set limits (seconds)
        plt.xlim(xmin * 60, xmax * 60) # seconds
        left_lim = plt.xlim()[0] / 60 
        right_lim = plt.xlim()[1] / 60
        # convert seconds to minutes for labels
        increment = 60
        minute_ticks = np.arange(left_lim, right_lim + increment, increment)
        second_ticks = [x * 60 for x in minute_ticks]
        plt.xticks(second_ticks, minute_ticks)

        # y axis
        plt.ylabel("Frequency (mHz)", labelpad=20)
        plt.ylim(ymin, ymax)
        locs, _ = plt.yticks()
        plt.yticks(locs, [int(x * 1000) for x in locs]) # Hz to mHz

        # colorbar
        if display_colorbar:
            colorbar = plt.colorbar(spec, extend='max', aspect=10)
            colorbar.ax.set_yticks([0.1, 0.01, 0.001])
            colorbar.ax.set_title('Intensity\n($Log_{10}$)', y=-0.40)

    fig.tight_layout()
    plt.show()

def plot_traces(names, time, dffs, rc, color='black', xmin=0, xmax=360, ymin=-0.1, ymax=1):
    with plt.rc_context(rc):
        fig_width = rc['figure.figsize'][0]
        fig, axes = plt.subplots(len(dffs), 2, figsize=(fig_width, 2 * (len(dffs))), gridspec_kw={'width_ratios': [1, 20]})
        fig.subplots_adjust(top=0.99, bottom=0.05, hspace=0.9, wspace=0.15)

        for axes, dff, name in zip(axes, dffs, names):
            label, ax = axes
            ax.plot(time, dff, color=color)

            # x axis
            ax.set_xlabel("Time (mins)")
            ax.set_xlim(xmin, xmax)
            minute_ticks = np.arange(xmin, xmax + 60, 60, int)
            ax.set_xticks(minute_ticks, minute_ticks)
            fig.tight_layout()

            # y axis
            ax.set_ylabel("ΔF/F")
            ax.set_ylim(ymin, ymax)
            increment = 0.5
            dff_ticks = np.arange(0, ymax+increment, increment)
            ax.set_yticks(dff_ticks, dff_ticks)

            # label
            label.axis('off')
            label.text(0, 0.5, f"{name}", 
                       va='center', ha='left', 
                       fontsize=rc['font.size'] * 1.5, 
                       rotation=90,
                       transform=label.transAxes)
    
    plt.show()

def plot_specs(names, f, t, mags, mymap, rc, display_colorbar=True, xmin=0, xmax=360, ymin=0, ymax=0.03):
    with plt.rc_context(rc):
        fig_width = rc['figure.figsize'][0]
        fig, axes = plt.subplots(len(mags), 2, figsize=(fig_width, 3 * (len(mags))), gridspec_kw={'width_ratios': [1, 20]})
        fig.subplots_adjust(top=0.99, bottom=0.05, hspace=0.9, wspace=0.15)

        for axes, mag, name in zip(axes, mags, names):
            label, ax = axes
            
            spec = ax.pcolormesh(
                t,
                f,
                mag,
                cmap=mymap,
                shading="nearest",
                snap=True,
                norm=colors.LogNorm(vmin=0.001, vmax=0.1)
            )

            # x axis
            ax.set_xlabel("Time (mins)", y=-0.25, labelpad=20)
            # set limits (seconds)
            ax.set_xlim(xmin * 60, xmax * 60) # seconds
            left_lim = ax.get_xlim()[0] / 60 
            right_lim = ax.get_xlim()[1] / 60
            # convert seconds to minutes for labels
            increment = 60
            minute_ticks = np.arange(left_lim, right_lim + increment, increment)
            second_ticks = [x * 60 for x in minute_ticks]
            ax.set_xticks(second_ticks, minute_ticks)

            # y axis
            ax.set_ylabel("Frequency (mHz)", labelpad=20)
            ax.set_ylim(ymin, ymax)
            locs = ax.get_yticks()
            ax.set_yticks(locs, [int(x * 1000) for x in locs]) # Hz to mHz

            # colorbar
            if display_colorbar:
                colorbar = ax.colorbar(spec, extend='max', aspect=10)
                colorbar.ax.set_yticks([0.1, 0.01, 0.001])
                colorbar.ax.set_title('Intensity\n($Log_{10}$)', y=-0.40)

            # label
            label.axis('off')
            label.text(0, 0.5, f"{name}", 
                       va='center', ha='left', 
                       fontsize=rc['font.size'] * 1,
                       rotation=90,
                       transform=label.transAxes)
    
    plt.show()

def plot_pointplot(dataframe, x, y, rc, category, linestyle=None, xmin=0, xmax=15, xinterval=2, xlabels=None, ymin=None, ymax=None, lines=True):
    with plt.rc_context(rc):
        fig, ax = plt.subplots()
        sns.set(style="whitegrid", palette="colorblind", rc=rc)
        sns.pointplot(data=dataframe, x=x, y=y, hue=category, linestyle=linestyle, ax=ax, join=lines)
        legend = ax.get_legend()
        if legend is not None: 
            sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.1), 
                            ncol=3, title=None, frameon=False)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if xlabels is not None:
            ax.set_xticks(ticks=list(range(len(xlabels))), labels=xlabels)
        ax.set_ylim(ymin, ymax)
        plt.show()

def plot_cdf(dataframe, x, category, rc):
    with plt.rc_context(rc):
        fig, ax = plt.subplots()
        sns.set(style="whitegrid", palette="colorblind", rc=rc)
        sns.ecdfplot(data=dataframe, x=x, hue=category, ax=ax)
        legend = ax.get_legend()
        if legend is not None: 
            sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.1), 
                            ncol=3, title=None, frameon=False)
        ax.set_xlabel(x)
        ax.set_ylabel("Proportion")
        plt.show()

def main():
    group, exp, all_embryos, emb = load_sample_data()
    mymap = mpl.colormaps['turbo']
    mymap.set_over('black')

    _, time, dff = preprocess_dff(emb)
    names, times, dffs = preprocess_multi_dffs(all_embryos)

    # all raw signals
    plot_raw_signals(all_embryos, subplots_rc)

    # single trace
    _, time, dff = preprocess_dff(emb)
    plot_trace(time, dff, trace_rc)

    # all traces
    plot_traces(names, times[0], dffs, subplots_rc)

    # single spec
    f, t, mag = calculate_STFT(dff)
    plot_spec(f, t, mag, mymap, spec_rc, display_colorbar=False)

    # average spectrogram
    f, t, mags = calculate_multi_STFTs(dffs)
    avg_mag = calculate_avg_magnitude(mags)
    plot_spec(f, t, avg_mag, mymap, spec_rc, display_colorbar=False)

    # all spectrograms
    plot_specs(names, f, t, mags, mymap, subplots_rc, display_colorbar=False)


if __name__ == "__main__":
    main()