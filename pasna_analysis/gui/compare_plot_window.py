import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import seaborn as sns

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget


matplotlib.use("QtAgg")


class ComparePlotWindow(QWidget):
    def __init__(self, groups):
        super().__init__()

        self.groups = groups
        self.setWindowTitle("Plots")

        layout = QHBoxLayout()
        self.sidebar = QVBoxLayout()
        self.sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(self.sidebar)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(8, 6)))
        sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.5)
        self.ax = self.canvas.figure.subplots()

        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.create_buttons()

    def create_buttons(self):
        btns = {
            "Dev times at first peak": self.dt_first_peak,
            "CDF of developmental peak times": self.cdf_dt,
            "Peak amplitudes by episode": self.peak_amplitudes_by_ep,
            "Dev time by episode": self.dt_by_ep,
            "Episode intervals": self.ep_intervals,
            "Decay times": self.decay_times,
        }

        for label, fn in btns.items():
            button = QPushButton(label)
            button.clicked.connect(fn)
            button.setMaximumWidth(250)
            self.sidebar.addWidget(button)

    def dt_first_peak(self):
        """Developmental time at first peak."""
        self.ax.clear()
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
        self.canvas.draw()

    def cdf_dt(self):
        """Cummulative distribution function of peak developmental times."""
        self.ax.clear()
        data = {"dev_time": [], "group": []}

        for group_name, group in self.groups.items():
            for exp in group.values():
                for emb in exp.embryos.values():
                    dev_times = [emb.get_DT_from_time(t) for t in emb.trace.peak_times]
                    data["dev_time"].extend(dev_times)
                    data["group"].extend([group_name] * len(dev_times))

        sns.ecdfplot(data=data, x="dev_time", hue="group", ax=self.ax)

        self.ax.set_xlim([1.9, 2.9])
        self.ax.set_ylabel("Proportion")
        self.ax.set_xlabel("Developmental Time")
        self.canvas.draw()

    def peak_amplitudes_by_ep(self):
        """Peak amplitudes for each episode."""
        self.ax.clear()
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
        self.canvas.draw()

    def dt_by_ep(self):
        """Developmental time for each episode."""
        self.ax.clear()
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
        self.canvas.draw()

    def ep_intervals(self):
        """Intervals between each episode."""
        self.ax.clear()
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
        self.canvas.draw()

    def decay_times(self):
        """Decay times"""
        self.ax.clear()
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
        self.canvas.draw()
