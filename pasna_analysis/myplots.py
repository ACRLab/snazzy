import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pasna_analysis import FrequencyAnalysis


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


def plot_trace(
    time,
    dff,
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

        # x axis
        plt.xlabel("Time (mins)")
        plt.xlim(xmin, xmax)
        minute_ticks = np.arange(xmin, xmax + xinterval, xinterval, int)
        plt.xticks(minute_ticks, minute_ticks)
        fig.tight_layout()

        # y axis
        plt.ylabel("ΔF/F")
        plt.ylim(ymin, ymax)
        # if the ymin is close to a whole number, just round
        if abs(ymin - round(ymin)) < 0.2:
            ymin = round(ymin)
        dff_ticks = np.arange(ymin, ymax + yinterval, yinterval)
        plt.yticks(dff_ticks, dff_ticks)
        fig.tight_layout()
        plt.show()


def plot_spec(
    f, t, Zxx, mymap, rc, display_colorbar=True, xmin=0, xmax=360, ymin=0, ymax=0.03
):
    with plt.rc_context(rc):
        fig = plt.figure()
        mag = abs(Zxx)
        spec = plt.pcolormesh(
            t,
            f,
            mag,
            cmap=mymap,
            shading="nearest",
            snap=True,
            norm=colors.LogNorm(vmin=0.001, vmax=0.1),
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


def plot_traces(
    embryos, rc, title=None, color=None, xmin=0, xmax=360, ymin=-0.1, ymax=1
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
            ax.set_xlim(xmin, xmax)
            minute_ticks = np.arange(xmin, xmax + 60, 60, int)
            ax.set_xticks(minute_ticks, minute_ticks)
            fig.tight_layout()

            # y axis
            ax.set_ylabel("ΔF/F")
            ax.set_ylim(ymin, ymax)
            increment = 0.5
            dff_ticks = np.arange(0, ymax + increment, increment)
            ax.set_yticks(dff_ticks, dff_ticks)

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
            spec = ax.pcolormesh(
                t,
                f,
                mag,
                cmap=mymap,
                shading="nearest",
                snap=True,
                norm=colors.LogNorm(vmin=0.001, vmax=0.1),
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
    ymin=None,
    ymax=None,
):
    with plt.rc_context(rc):
        fig, ax = plt.subplots()
        sns.set_theme(style="whitegrid", palette="colorblind", rc=rc)
        sns.pointplot(
            data=dataframe, x=x, y=y, hue=category, linestyle=linestyle, ax=ax
        )
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
        ax.set_ylabel(y)
        if xlabels is not None:
            ax.set_xticks(ticks=list(range(len(xlabels))), labels=xlabels)
        ax.set_ylim(ymin, ymax)
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
