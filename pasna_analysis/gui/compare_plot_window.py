from datetime import datetime

import matplotlib
import matplotlib.axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget

from pasna_analysis.gui import GroupModel

matplotlib.use("QtAgg")


class ComparePlotWindow(QWidget):
    def __init__(self, groups: list[GroupModel]):
        super().__init__()

        self.groups = groups
        self.setWindowTitle("Plots")

        layout = QVBoxLayout()
        self.save_all_btn = QPushButton("Save all plots")
        self.save_all_btn.clicked.connect(self.save_all_plots)
        layout.addWidget(self.save_all_btn)

        plot_layout = QHBoxLayout()
        layout.addLayout(plot_layout)

        self.sidebar_widget = QWidget(self)
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        plot_layout.addWidget(self.sidebar_widget)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(16, 10)))
        sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.5)
        self.ax = self.canvas.figure.subplots()
        self.axes = [self.ax]

        plot_layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.btns = {
            "Dev times at first peak": self.dt_first_peak,
            "Dev times when hatching": self.dt_hatching,
            "SNA duration": self.sna_duration,
            "Number of episodes": self.num_episodes,
            "CDF of developmental peak times": self.cdf_dt,
            "Peak amplitudes by episode": self.peak_amplitudes_by_ep,
            "Dev time by episode": self.dt_by_ep,
            "Episode intervals": self.ep_intervals,
            "Baseline / burst ratio per episode": self.baseline_ratio,
            "Episode durations": self.ep_durations,
            "Decay times": self.decay_times,
            "Rise times": self.rise_times,
            "Average spectrograms": self.average_spectrogram,
        }
        self.create_buttons()

    def save_all_plots(self):
        for group in self.groups:
            for exp in group.experiments.values():
                exp_dir = exp.directory.parent
                timestamp = datetime.now().strftime("%m%d%Y_%H:%M:%S")
                save_path = exp_dir / "plots" / timestamp
                save_path.mkdir(parents=True, exist_ok=True)

                for plot_fn in self.btns.values():
                    plot_fn(save=True, save_dir=save_path)

        notification = QMessageBox(self)
        notification.setWindowTitle("Save plots")
        notification.setText("All plots were saved.")
        notification.exec()

    def clear_axes(self, rows=1, cols=1):
        """Clears the canvas and creates a new axes object for a new plot."""
        self.canvas.figure.clear()
        self.ax = self.canvas.figure.subplots(rows, cols)

    def create_buttons(self):
        for label, fn in self.btns.items():
            button = QPushButton(label)
            button.clicked.connect(fn)
            button.setMaximumWidth(250)
            self.sidebar.addWidget(button)

    def dt_first_peak(self, save=False, save_dir=None):
        """Developmental time at first peak."""
        self.clear_axes()
        data = {"dev_fp": [], "group": []}

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                if emb.trace.peak_times.size == 0:
                    continue
                time_first_peak = emb.trace.peak_times[0]
                dev_time_first_peak = emb.get_DT_from_time(time_first_peak)
                data["dev_fp"].append(dev_time_first_peak)
                data["group"].append(group.name)

        sns.swarmplot(
            data=data,
            x="group",
            y="dev_fp",
            hue="group",
            size=7,
            legend="brief",
            ax=self.ax,
        )

        self.ax.set_title("Developmental times at first peak")
        self.ax.set_ylabel("Dev time")
        self.ax.set_xlabel("Group")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "dev_time_first_peak.png")

    def dt_hatching(self, save=False, save_dir=None):
        """Developmental times when hatching."""
        self.clear_axes()
        data = {"dev_hatching": [], "group": []}

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                trace = emb.trace
                time_hatching = trace.time[trace.trim_idx]
                dev_time_first_peak = emb.get_DT_from_time(time_hatching)

                data["dev_hatching"].append(dev_time_first_peak)
                data["group"].append(group.name)

        ax = sns.swarmplot(
            data=data,
            x="group",
            y="dev_hatching",
            hue="group",
            size=7,
            legend="brief",
            ax=self.ax,
        )
        ax.set_ylabel("Dev time")
        ax.set_xlabel("Group")
        ax.set_title("Dev time when hatching")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "dt_hatching.png")

    def sna_duration(self, save=False, save_dir=None):
        """Number of episodes"""
        self.clear_axes()
        data = {"group": [], "duration": []}

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                trace = emb.trace
                if trace.peak_times.size == 0:
                    continue
                data["group"].append(group.name)
                duration = (trace.time[trace.trim_idx] - trace.peak_times[0]) / 60
                data["duration"].append(duration)

        ax = sns.swarmplot(data=data, x="group", y="duration", hue="group", ax=self.ax)
        ax.set_title("SNA duration")
        ax.set_ylabel("time (mins)")
        ax.set_xlabel("Group")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "sna_duration.png")

    def num_episodes(self, save=False, save_dir=None):
        """Number of episodes"""
        self.clear_axes()
        data = {"group": [], "num_eps": []}

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                trace = emb.trace
                data["group"].append(group.name)
                data["num_eps"].append(len(trace.peak_idxes))

        ax = sns.swarmplot(data=data, x="group", y="num_eps", hue="group", ax=self.ax)
        ax.set_title("Number of episodes")
        ax.set_ylabel("# eps")
        ax.set_xlabel("Group")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "num_episodes.png")

    def cdf_dt(self, save=False, save_dir=None):
        """Cummulative distribution function of peak developmental times."""
        self.clear_axes()
        data = {"dev_time": [], "group": []}

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                dev_times = [emb.get_DT_from_time(t) for t in emb.trace.peak_times]
                data["dev_time"].extend(dev_times)
                data["group"].extend([group.name] * len(dev_times))

        sns.ecdfplot(data=data, x="dev_time", hue="group", ax=self.ax)

        self.ax.set_title("Cummulative distribution of peak dev times.")
        self.ax.set_xlim([1.9, 2.9])
        self.ax.set_ylabel("Proportion")
        self.ax.set_xlabel("Developmental Time")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "cdf_dev_time.png")

    def peak_amplitudes_by_ep(self, save=False, save_dir=None):
        """Peak amplitudes for each episode."""
        self.clear_axes()
        data = {"peak_amp": [], "group": [], "peak_idx": []}
        num_of_peaks = 15

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                for i, amp in zip(range(num_of_peaks), emb.trace.peak_amplitudes):
                    data["peak_amp"].append(amp)
                    data["group"].append(group.name)
                    data["peak_idx"].append(i)

        dodge = len(set(data["group"])) > 1
        sns.pointplot(
            data=data,
            x="peak_idx",
            y="peak_amp",
            hue="group",
            errorbar="ci",
            dodge=dodge,
            ax=self.ax,
        )

        self.ax.set_xticks(list(range(0, num_of_peaks, 2)))
        self.ax.set_title(f"Burst amplitudes")
        self.ax.set_xlabel("Burst #")
        self.ax.set_ylabel("\u0394F/F")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "peak_amplitudes_by_ep.png")

    def dt_by_ep(self, save=False, save_dir=None):
        """Developmental time for each episode."""
        self.clear_axes()
        data = {"group": [], "dev_time": [], "idx": []}
        num_of_peaks = 15

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                for i, t in zip(range(num_of_peaks), emb.trace.peak_times):
                    data["group"].append(group.name)
                    data["dev_time"].append(emb.get_DT_from_time(t))
                    data["idx"].append(i)

        dodge = len(set(data["group"])) > 1
        sns.pointplot(
            data=data,
            x="idx",
            y="dev_time",
            hue="group",
            errorbar="ci",
            dodge=dodge,
            ax=self.ax,
        )

        self.ax.set_xticks(list(range(0, num_of_peaks, 2)))
        self.ax.set_title("Dev time per burst")
        self.ax.set_xlabel("Burst #")
        self.ax.set_ylabel("Dev time")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "dev_time_by_ep.png")

    def ep_intervals(self, save=False, save_dir=None):
        """Intervals between each episode."""
        self.clear_axes()
        data = {"group": [], "interval": [], "idx": []}
        num_of_peaks = 15

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                for i, interval in zip(range(num_of_peaks), emb.trace.peak_intervals):
                    data["group"].append(group.name)
                    data["interval"].append(interval / 60)
                    data["idx"].append(i)

        dodge = len(set(data["group"])) > 1
        sns.pointplot(
            data=data,
            x="idx",
            y="interval",
            hue="group",
            errorbar="ci",
            dodge=dodge,
            ax=self.ax,
        )

        self.ax.set_xticks(list(range(0, num_of_peaks, 2)))
        self.ax.set_title("Intervals by burst")
        self.ax.set_xlabel("Interval #")
        self.ax.set_ylabel("Interval (min)")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "episode_intervals.png")

    def baseline_ratio(self, save=False, save_dir=None):
        """Quiescent / active ratio per episode"""
        self.clear_axes()
        num_of_peaks = 15
        data = {"group": [], "qa_ratio": [], "idx": []}

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                trace = emb.trace
                for i, (ps, pe) in zip(range(num_of_peaks), trace.peak_bounds_times):
                    try:
                        next_ps = trace.peak_bounds_times[i + 1][0]
                    except IndexError:
                        break
                qa = (next_ps - pe) / (next_ps - ps)
                data["group"].append(group.name)
                data["qa_ratio"].append(qa)
                data["idx"].append(i)

        sns.pointplot(
            data=data, x="idx", y="qa_ratio", hue="group", linestyle="None", ax=self.ax
        )
        self.ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        self.ax.set_title("Baseline ratio per episode")
        self.ax.set_xlabel("Peak #")
        self.ax.set_ylabel("t_base / episode")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "baseline_ratio.png")

    def decay_times(self, save=False, save_dir=None):
        """Decay times.

        Time between the peak time and end of the peak (right peak width boundary)."""
        self.clear_axes()
        data = {"group": [], "decay_times": [], "idx": []}

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                for i, decay in zip(range(15), emb.trace.peak_decay_times):
                    data["group"].append(group.name)
                    data["decay_times"].append(decay / 60)
                    data["idx"].append(i)

        dodge = len(set(data["group"])) > 1
        sns.pointplot(
            data=data,
            x="idx",
            y="decay_times",
            hue="group",
            errorbar="ci",
            dodge=dodge,
            ax=self.ax,
        )
        self.ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        self.ax.set_title("Peak decay times")
        self.ax.set_xlabel("Peak #")
        self.ax.set_ylabel("Duration (min)")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "decay_times.png")

    def average_spectrogram(self, save=False, save_dir=None):
        self.canvas.figure.clear()
        ax = self.canvas.figure.subplots(len(self.groups), 1)

        if isinstance(ax, matplotlib.axes.Axes) == 1:
            self.ax = [ax]
        else:
            self.ax = ax

        for i, group in enumerate(self.groups):
            f_zero = None
            t_zero = None
            for exp in group.experiments.values():
                Zxxs = []
                for emb in exp.embryos:
                    stft = emb.trace.calculate_STFT(emb.trace.aligned_dff)
                    if stft is None:
                        continue
                    f, t, Zxx = stft
                    if f_zero is None and t_zero is None:
                        f_zero = f
                        t_zero = t
                    Zxxs.append(Zxx)
                Zxxs = np.array(Zxxs)
                magnitudes = np.abs(Zxxs)
                average_magnitude = np.mean(magnitudes, axis=0)

            max_mag = np.max(average_magnitude)
            self.ax[i].pcolormesh(
                t_zero // 60,
                f_zero,
                average_magnitude,
                cmap="turbo",
                shading="nearest",
                snap=True,
                norm=matplotlib.colors.LogNorm(vmin=0.01 * max_mag),
            )
            self.ax[i].set_ylim(0, 0.016)
            self.ax[i].set_title(group.name)
            self.ax[i].set_ylabel("Hz")
            self.ax[i].set_xlabel("time (mins)")

        self.canvas.figure.tight_layout()

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "average_spectrogram.png")

    def ep_durations(self, save=False, save_dir=None):
        """Duration of each episode."""
        self.clear_axes()
        data = {"group": [], "duration": [], "idx": []}

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                for i, duration in zip(range(15), emb.trace.peak_durations):
                    data["group"].append(group.name)
                    data["duration"].append(duration / 60)
                    data["idx"].append(i)

        dodge = len(set(data["group"])) > 1
        ax = sns.pointplot(
            data=data, x="idx", y="duration", hue="group", dodge=dodge, ax=self.ax
        )

        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_title("Durations by peak")
        ax.set_ylabel("Duration (min)")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "peak_durations.png")

    def rise_times(self, save=False, save_dir=None):
        """Peak rise times.

        Time between the start of the peak (left width boundary) and the peak time."""
        self.clear_axes()
        data = {"group": [], "duration": [], "idx": []}

        for group in self.groups:
            for exp_name, emb in group.iter_all_embryos():
                for i, duration in zip(range(15), emb.trace.peak_rise_times):
                    data["group"].append(group.name)
                    data["duration"].append(duration)
                    data["idx"].append(i)

        dodge = len(set(data["group"])) > 1
        ax = sns.pointplot(
            data=data, x="idx", y="duration", hue="group", dodge=dodge, ax=self.ax
        )

        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_title("Rise times")
        ax.set_xlabel("Peak #")
        ax.set_ylabel("Duration (s)")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "peak_durations.png")
