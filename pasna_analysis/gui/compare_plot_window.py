from datetime import datetime

import matplotlib
import matplotlib.axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget

from pasna_analysis import Experiment

matplotlib.use("QtAgg")


class ComparePlotWindow(QWidget):
    def __init__(self, groups: dict[str, dict[str, Experiment]]):
        super().__init__()

        self.groups = groups
        self.setWindowTitle("Plots")

        layout = QVBoxLayout()
        self.save_all_btn = QPushButton("Save all plots")
        self.save_all_btn.clicked.connect(self.save_all_plots)
        layout.addWidget(self.save_all_btn)

        plot_layout = QHBoxLayout()
        layout.addLayout(plot_layout)

        self.sidebar = QVBoxLayout()
        self.sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)
        plot_layout.addLayout(self.sidebar)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(16, 10)))
        sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.5)
        self.ax = self.canvas.figure.subplots()
        self.axes = [self.ax]

        plot_layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.btns = {
            "Dev times at first peak": self.dt_first_peak,
            "CDF of developmental peak times": self.cdf_dt,
            "Peak amplitudes by episode": self.peak_amplitudes_by_ep,
            "Dev time by episode": self.dt_by_ep,
            "Episode intervals": self.ep_intervals,
            "Decay times": self.decay_times,
            "Average spectrograms": self.average_spectrogram,
        }
        self.create_buttons()

    def save_all_plots(self):
        for group in self.groups.values():
            for exp in group.values():
                exp_dir = exp.pd_params_path.parent
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

        for group_name, group in self.groups.items():
            for exp in group.values():
                for emb in exp.embryos.values():
                    time_first_peak = emb.trace.peak_times[0]
                    dev_time_first_peak = emb.get_DT_from_time(time_first_peak)
                    data["dev_fp"].append(dev_time_first_peak)
                    data["group"].append(group_name)

        sns.swarmplot(data=data, x="group", y="dev_fp", hue="group", size=7, ax=self.ax)

        self.ax.set_title("Developmental times at first peak")
        self.ax.set_ylabel("Dev time")
        self.ax.set_xlabel("Group")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "dev_time_first_peak.png")

    def cdf_dt(self, save=False, save_dir=None):
        """Cummulative distribution function of peak developmental times."""
        self.clear_axes()
        data = {"dev_time": [], "group": []}

        for group_name, group in self.groups.items():
            for exp in group.values():
                for emb in exp.embryos.values():
                    dev_times = [emb.get_DT_from_time(t) for t in emb.trace.peak_times]
                    data["dev_time"].extend(dev_times)
                    data["group"].extend([group_name] * len(dev_times))

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

        for group_name, group in self.groups.items():
            for exp in group.values():
                for emb in exp.embryos.values():
                    for i, amp in zip(range(num_of_peaks), emb.trace.peak_amplitudes):
                        data["peak_amp"].append(amp)
                        data["group"].append(group_name)
                        data["peak_idx"].append(i)
        sns.pointplot(
            data=data,
            x="peak_idx",
            y="peak_amp",
            hue="group",
            errorbar="ci",
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

        for group_name, group in self.groups.items():
            for exp in group.values():
                for emb in exp.embryos.values():
                    for i, t in zip(range(num_of_peaks), emb.trace.peak_times):
                        data["group"].append(group_name)
                        data["dev_time"].append(emb.get_DT_from_time(t))
                        data["idx"].append(i)

        sns.pointplot(
            data=data, x="idx", y="dev_time", hue="group", errorbar="ci", ax=self.ax
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

        for group_name, group in self.groups.items():
            for exp in group.values():
                for emb in exp.embryos.values():
                    for i, interval in zip(
                        range(num_of_peaks), emb.trace.peak_intervals
                    ):
                        data["group"].append(group_name)
                        data["interval"].append(interval / 60)
                        data["idx"].append(i)

        sns.pointplot(
            data=data, x="idx", y="interval", hue="group", errorbar="ci", ax=self.ax
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

    def decay_times(self, save=False, save_dir=None):
        """Decay times"""
        self.clear_axes()
        data = {"group": [], "decay_times": [], "idx": []}

        for group_name, group in self.groups.items():
            for exp in group.values():
                for emb in exp.embryos.values():
                    for i, decay in zip(range(15), emb.trace.peak_decay_times):
                        data["group"].append(group_name)
                        data["decay_times"].append(decay)
                        data["idx"].append(i)

        sns.pointplot(
            data=data,
            x="idx",
            y="decay_times",
            hue="group",
            errorbar="ci",
            linestyle="None",
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

        for i, (group_name, group) in enumerate(self.groups.items()):
            f_zero = None
            t_zero = None
            for exp in group.values():
                Zxxs = []
                for emb in exp.embryos.values():
                    stft = emb.trace.stft(duration=3600)
                    if stft is None:
                        continue
                    f, t, zxx = stft
                    if f_zero is None and t_zero is None:
                        f_zero = f
                        t_zero = t
                    Zxxs.append(zxx)
                Zxxs = np.array(Zxxs)
                abs_Zxx = np.abs(Zxxs)
                avg_Zxx = np.mean(abs_Zxx, axis=0)

            self.ax[i].pcolormesh(
                t_zero,
                f_zero,
                abs(avg_Zxx),
                vmin=0,
                vmax=0.03,
                cmap="plasma",
                shading="nearest",
                snap=True,
            )
            self.ax[i].set_title(group_name)
            self.ax[i].set_ylabel("Hz")
            self.ax[i].set_xlabel("time (mins)")

        self.canvas.figure.tight_layout()

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "average_spectrogram.png")
