import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator

from snazzy_analysis import FrequencyAnalysis


def set_darkbg(dark):
    plt.rcdefaults()
    plt.style.use("default" if not dark else "dark_background")


def plot_raw_signals(embryos, rc, title=None):
    with plt.rc_context(rc):
        fig_width = rc["figure.figsize"][0]
        fig, axes = plt.subplots(
            len(embryos),
            3,
            squeeze=False,
            figsize=(fig_width, 2.5 * (len(embryos))),
            gridspec_kw={"width_ratios": [2, 10, 10]},
        )
        fig.subplots_adjust(top=0.99, bottom=0.05, hspace=0.9, wspace=0.15)

        if title is not None:
            fig.suptitle(title, fontsize=rc["font.size"] * 2)
            fig.tight_layout(
                rect=[0, 0, 1, 0.95]
            )  # reserve 5% of vertical space at the top

        for axes, emb in zip(axes, embryos):
            label, left, right = axes

            time = emb.activity[:, 0] / 60
            active = emb.trace.active
            struct = emb.trace.struct
            left.plot(time, active, color="green")
            right.plot(time, struct, color="firebrick")

            # x axis
            left.set_xlabel("Time (mins)")
            time_min = time[0]
            time_max = time[len(time) - 1]
            minute_ticks = np.arange(time_min, time_max + 60, 60, int)
            left.set_xticks(minute_ticks, minute_ticks)
            right.set_xlabel("Time (mins)")
            right.set_xticks(minute_ticks, minute_ticks)

            # y axis
            left.set_ylabel("Intensity")
            right.set_ylabel("Intensity")

            # label
            label.axis("off")

            label.text(
                0.35,
                0.5,
                f"{emb.name}",
                va="center",
                ha="center",
                fontsize=rc["font.size"] * 1.5,
                rotation=90,
                transform=label.transAxes,
            )

    plt.show()


def plot_trace_with_overlay(
    time,
    dff,
    overlay,
    rc,
    color=None,
    xmin=0,
    xmax=360,
    xinterval=60,
    ymin=-0.1,
    ymax=1,
    yinterval=1,
):
    with plt.rc_context(rc):
        if color is None:
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][9]

        fig = plt.figure()
        plt.plot(time, dff, color=color)
        plt.plot(time, overlay, color="orange")

        plt.xlabel("Time (mins)")
        plt.xlim(xmin, xmax)
        minute_ticks = np.arange(xmin, xmax + xinterval, xinterval, int)
        plt.xticks(minute_ticks, minute_ticks)

        plt.ylabel("ΔF/F")
        plt.ylim(ymin, ymax)
        if abs(ymin - round(ymin)) < 0.2:
            ymin = round(ymin)
        dff_ticks = np.arange(ymin, ymax + yinterval, yinterval)
        plt.yticks(dff_ticks, dff_ticks)
        fig.tight_layout()
        plt.show()


def plot_trace(
    emb,
    time,
    dff,
    rc,
    color=None,
    xmin=0,
    xmax=360,
    xinterval=None,
    ymin=-0.1,
    ymax=1,
    yinterval=1,
    bursts=False, 
    minibursts=False, 
    save=False,
    title="None"
):
    with plt.rc_context(rc):
        if color is None:
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][9]

        fig = plt.figure()
        plt.plot(time, dff, color=color)


        # x axis
        plt.xlabel("Time (mins)")
        if xinterval is None:
            plt.xlim(xmin, xmax)
            plt.xticks([], [])
        else:
            plt.xlim(xmin, xmax+30)
            aligned_minute_ticks = np.arange(xmin + 30, xmax + 30+ xinterval, xinterval, int)
            minute_ticks = np.arange(xmin, xmax + xinterval, xinterval, int)
            plt.xticks(aligned_minute_ticks, minute_ticks)

        # y axis
        plt.ylabel("ΔF/F")
        plt.ylim(ymin, ymax)
        # if the ymin is close to a whole number, just round
        if abs(ymin - round(ymin)) < 0.2:
            ymin = round(ymin)
        dff_ticks = np.arange(ymin, ymax + yinterval, yinterval)
        plt.yticks(dff_ticks, dff_ticks)
        plt.yticks([0, 1], [0, 1])

        trace = emb.trace
        xmin, xmax = plt.xlim()
        mask = (time >= xmin) & (time <= xmax)

        ymax = np.nanmax(dff[mask].astype(float))
        time = trace.time[:trace.trim_idx - trace.aligned_offset]
        if bursts:
            p = (trace.peak_times - time[trace.aligned_offset])/60
            plt.plot(p, np.full(len(p), ymax+0.15), "|", mew=5, markersize=20, color="black")
        if minibursts:
            lp = (trace.localpeak_times - time[trace.aligned_offset])/60
            plt.plot(lp, np.full(len(lp), ymax+0.15), "|", mew=5, markersize=20, color="red")

        fig.tight_layout()

        if save:
            plt.savefig(f"{title}")
        plt.show()


def plot_spec(
    f,
    t,
    Zxx,
    mymap,
    rc,
    display_colorbar=True,
    xmin=0,
    xmax=360,
    ymin=0,
    ymax=0.03,
    vmax=None,
    vmin=None,
):
    with plt.rc_context(rc):
        fig = plt.figure()
        mag = abs(Zxx)
        max_mag = np.max(mag)
        if vmax is None:
            vmax = max_mag
        if vmin is None:
            vmin = 0.01 * vmax
        spec = plt.pcolormesh(
            t,
            f,
            mag,
            cmap=mymap,
            shading="nearest",
            snap=True,
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        )

        # x axis
        plt.xlabel("Time (mins)", y=-0.25, labelpad=20)
        # set limits (seconds)
        plt.xlim(xmin * 60, xmax * 60)  # seconds
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
        plt.yticks(locs, [int(x * 1000) for x in locs])  # Hz to mHz

        # colorbar
        if display_colorbar:
            colorbar = plt.colorbar(spec, extend="max", aspect=10)
            colorbar.ax.set_yticks([0.1, 0.01, 0.001])
            colorbar.ax.set_title("Intensity\n($Log_{10}$)", y=-0.40)

    fig.tight_layout()
    plt.show()


def plot_scalogram(
    f,
    t,
    cwtmatr,
    mymap,
    rc,
    display_colorbar=True,
    xmin=0,
    xmax=360,
    xinterval=60,
    vmax=None,
    vmin=None,
    save=False,
    title="None"
):
    with plt.rc_context(rc):
        fig = plt.figure()
        mag = abs(cwtmatr)
        max_mag = np.max(mag)
        if vmax is None:
            vmax = max_mag
        if vmin is None:
            vmin = 0.01 * vmax
        
        scalogram = plt.pcolormesh(t, f, cwtmatr, cmap=mymap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), rasterized=True)

        # x axis
        plt.xlabel("Time (mins)")
        plt.xlim(xmin, xmax)
        aligned_minute_ticks = np.arange(xmin + 30, xmax + 30+ xinterval, xinterval, int)
        minute_ticks = np.arange(xmin, xmax + xinterval, xinterval, int)
        plt.xticks(aligned_minute_ticks, minute_ticks)

        # y axis
        ax_freq = plt.gca()
        ax_period = ax_freq.twinx()
        ax_freq.set_yscale("log")
        ax_freq.set_ylabel("Frequency (mHz)", labelpad=25)
        ax_freq.set_ylim(0.001, 0.1)
        y_ticks = [0.001, 0.01, 0.1]
        # y_ticks = [0.001, 0.005, 0.01, 0.05, 0.1]
        ax_freq.set_yticks(y_ticks)
        ax_freq.set_yticklabels([f"{hz_to_mhz(t)} " for t in y_ticks])
        for label in ax_freq.get_yticklabels():
            label.set_verticalalignment('top')
        ax_freq.yaxis.set_major_locator(LogLocator(base=10))

        ax_period.set_ylabel("Period (min:s)", rotation=270, labelpad=35)
        ax_period.set_ylim(0.00065, 0.1)
        ax_period.set_yscale("log")
        ax_period.set_yticks(y_ticks)
        ax_period.set_yticklabels([f"  {fmt_period(t)}" for t in y_ticks])
        for label in ax_period.get_yticklabels():
            label.set_verticalalignment('top')
        ax_period.yaxis.set_major_locator(LogLocator(base=10))

        # colorbar
        if display_colorbar:
            colorbar = plt.colorbar(scalogram, extend="max", aspect=10)
            # colorbar.ax.set_yticks([0.1, 0.01, 0.001])
            colorbar.ax.set_title("Intensity\n($Log_{10}$)", y=-0.40)
    if save:
        plt.savefig(f"{title}", bbox_inches='tight')
    plt.show()

def fmt_period(f):
    p = 1 / f
    mins = int(p // 60)
    secs = int(p % 60)
    return f"{mins}:{secs:02d}"

def hz_to_mhz(f):
    return int(f*1000)



def plot_traces(
    embryos, rc, title=None, color=None, xmin=0, xmax=360, ymin=-0.1, ymax=1, bursts=False, minibursts=False
):
    with plt.rc_context(rc):
        if color is None:
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][9]

        fig_width = rc["figure.figsize"][0]
        fig_height = rc["figure.figsize"][1]
        fig, axes = plt.subplots(
            len(embryos),
            2,
            squeeze=False,
            figsize=(fig_width, fig_height * (len(embryos))),
            gridspec_kw={"width_ratios": [1, 20]},
        )
        fig.subplots_adjust(top=0.99, bottom=0.05, hspace=0.9, wspace=0.15)

        if title is not None:
            fig.suptitle(title, fontsize=rc["font.size"] * 2)
            fig.tight_layout(
                rect=[0, 0, 1, 0.95]
            )  # reserve 5% of vertical space at the top

        for axes, emb in zip(axes, embryos):
            label, ax = axes
            trace = emb.trace
            time = trace.aligned_time
            dff = trace.aligned_dff
            ax.plot(time, dff, color=color)


            # x axis
            ax.set_xlabel("Time (mins)")
            ax.set_xlim(xmin, xmax+30)
            aligned_minute_ticks = np.arange(xmin + 30, xmax + 30+ 60, 60, int)
            minute_ticks = np.arange(xmin, xmax + 60, 60, int)
            ax.set_xticks(aligned_minute_ticks, minute_ticks)
            fig.tight_layout()

            # y axis
            ax.set_ylabel("ΔF/F")
            ax.set_ylim(ymin, ymax)
            increment = 0.5
            dff_ticks = np.arange(0, ymax + increment, increment)
            ax.set_yticks(dff_ticks, dff_ticks)
#
            # label
            label.axis("off")
            label.text(
                0,
                0.5,
                f"{emb.name}",
                va="center",
                ha="left",
                fontsize=rc["font.size"] * 1.5,
                rotation=90,
                transform=label.transAxes,
            )
            if bursts:
                for p in trace.peak_times:
                    p_time = trace.time_to_aligned_time(p)
                    ax.axvline(p_time, color="green", alpha=0.5)
                for b in trace.get_peak_bounds_times():
                    ax.axvspan((b[0]- trace.time[trace.aligned_offset])/60, (b[1] - trace.time[trace.aligned_offset])/60, color="green", alpha=0.3)
            if minibursts:
                for lp in trace.localpeak_times:
                    ax.axvline((lp - trace.time[trace.aligned_offset])/60, color="red", alpha=0.3)

    plt.show()


def plot_specs(
    embryos,
    mymap,
    rc,
    title=None,
    display_colorbar=True,
    xmin=0,
    xmax=360,
    ymin=0,
    ymax=0.03,
    vmax=None,
    vmin=None,
):
    with plt.rc_context(rc):
        fig_width = rc["figure.figsize"][0]
        fig_height = rc["figure.figsize"][1]
        fig, axes = plt.subplots(
            len(embryos),
            2,
            squeeze=False,
            figsize=(fig_width, fig_height * (len(embryos))),
            gridspec_kw={"width_ratios": [1, 20]},
        )
        fig.subplots_adjust(top=0.99, bottom=0.05, hspace=0.9, wspace=0.15)

        if title is not None:
            fig.suptitle(title, fontsize=rc["font.size"] * 2)
            fig.tight_layout(
                rect=[0, 0, 1, 0.95]
            )  # reserve 5% of vertical space at the top

        for axes, emb in zip(axes, embryos):
            label, ax = axes

            trace = emb.trace
            dff = trace.aligned_dff
            f, t, Zxx = FrequencyAnalysis.calculate_STFT(dff)

            mag = abs(Zxx)
            max_mag = np.max(mag)
            if vmax is None:
                vmax = max_mag
            if vmin is None:
                vmin = 0.01 * vmax
            spec = ax.pcolormesh(
                t,
                f,
                mag,
                cmap=mymap,
                shading="nearest",
                snap=True,
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            )

            # x axis
            ax.set_xlabel("Time (mins)", y=-0.25, labelpad=20)
            # set limits (seconds)
            ax.set_xlim(xmin * 60, xmax * 60)  # seconds
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
            ax.set_yticks(locs, [int(x * 1000) for x in locs])  # Hz to mHz

            # colorbar
            if display_colorbar:
                colorbar = ax.colorbar(spec, extend="max", aspect=10)
                colorbar.ax.set_yticks([0.1, 0.01, 0.001])
                colorbar.ax.set_title("Intensity\n($Log_{10}$)", y=-0.40)

            # label
            label.axis("off")
            label.text(
                0,
                0.5,
                f"{emb.name}",
                va="center",
                ha="left",
                fontsize=rc["font.size"] * 1,
                rotation=90,
                transform=label.transAxes,
            )

    plt.show()


def plot_pointplot(
    dataframe,
    x,
    y,
    rc,
    category,
    linestyle=None,
    xlabels=None,
    errorbar="sd",
    ymin=0,
    ymax=1,
    yinterval=0.1,
    palette=None,
    save=False,
    title="None"
):
    with plt.rc_context(rc):
        fig, ax = plt.subplots()
        sns.set_theme(style="whitegrid", palette="colorblind", rc=rc)
        sns.pointplot(
            data=dataframe, x=x, y=y, hue=category, linestyle=linestyle, ax=ax, errorbar=errorbar, palette=palette, legend=True, err_kws={"color": "black", "linewidth": 2}
        )
        # sns.stripplot(
        #     data=dataframe, x=x, y=y, hue=category, ax=ax
        # )
        # legend = ax.get_legend()
        # if legend is not None:
        #     sns.move_legend(
        #         ax,
        #         "lower center",
        #         bbox_to_anchor=(0.5, 1.1),
        #         ncol=3,
        #         title=None,
        #         frameon=False,
        #     )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if xlabels is not None:
            ax.set_xticks(ticks=list(range(len(xlabels))), labels=xlabels)
        plt.xticks(rotation=45)
        plt.ylim(ymin, ymax)
        dff_ticks = np.arange(ymin, ymax + yinterval, yinterval)
        dff_ticks = [tick for tick in dff_ticks]
        plt.yticks(dff_ticks, dff_ticks)

        fig.tight_layout(pad=0)
        if save:
            plt.savefig(f"{title}")
        plt.show()


def plot_cdf(dataframe, x, category, rc):
    with plt.rc_context(rc):
        fig_height = rc["figure.figsize"][1]
        fig, ax = plt.subplots(figsize=(fig_height, fig_height))
        sns.set_theme(style="whitegrid", palette="colorblind", rc=rc)
        sns.ecdfplot(data=dataframe, x=x, hue=category, ax=ax)
        legend = ax.get_legend()
        if legend is not None:
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(0.5, 1.1),
                ncol=3,
                title=None,
                frameon=False,
            )
        ax.set_xlabel(x)
        ax.set_ylabel("Proportion")
        plt.show()
