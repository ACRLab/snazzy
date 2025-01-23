import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from pasna_analysis import Experiment, utils


matplotlib.use("QtAgg")


class PlotWindow(QWidget):
    def __init__(self, group: dict[str, Experiment], group_name: str):
        super().__init__()

        self.embryos = [emb for exp in group.values() for emb in exp.embryos.values()]

        self.setWindowTitle(f"Plots - {group_name}")

        layout = QHBoxLayout()
        self.sidebar = QVBoxLayout()
        self.sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(self.sidebar)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(5, 4)))
        self.ax = self.canvas.figure.subplots()

        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.create_buttons()

    def create_buttons(self):
        """Populates buttons in the sidebar.

        `btns` should receive a dict[str, Callable], where the str is the btn
        text and Callable is a function that adds a plot to `self.ax`.
        """
        btns = {
            "Peaks": self.plot_peaks,
            "Area Under the Curve": self.plot_AUC,
        }

        for label, fn in btns.items():
            button = QPushButton(label)
            button.clicked.connect(fn)
            button.setMaximumWidth(150)
            self.sidebar.addWidget(button)

    def plot_peaks(self):
        """Peak times for all embryos.

        Embryos are represented as horizontal lines."""
        self.ax.clear()
        times = []
        for emb in self.embryos:
            trace = emb.trace
            times.append(
                [t / 60 for t in trace.peak_times if t < trace.time[trace.trim_idx]]
            )
        style = {"marker": ".", "linestyle": "dashed", "linewidth": 0.5}
        for i, time in enumerate(times):
            self.ax.plot(time, [i] * len(time), **style)
        self.ax.set_title(f"Peak times")
        self.ax.set_xlabel("time (mins)")
        self.ax.set_ylabel("emb")
        self.ax.set_yticks([])
        self.canvas.draw()

    def plot_AUC(self):
        """Binned area under the curve."""
        self.ax.clear()
        data = {"auc": [], "bin": [], "emb": []}
        n_bins = 5
        first_bin = 2
        bin_width = 0.2
        for i, emb in enumerate(self.embryos):
            trace = emb.trace

            dev_time_at_peaks = emb.get_DT_from_time(trace.peak_times)

            bins = [first_bin + j * bin_width for j in range(n_bins)]
            bin_idxs = utils.split_in_bins(dev_time_at_peaks, bins)
            if not isinstance(bin_idxs, np.ndarray):
                continue
            data["auc"].extend(trace.peak_aucs)
            data["bin"].extend(bin_idxs)
            data["emb"].extend([str(i)] * len(trace.peak_aucs))

        bins.append(first_bin + bin_width * n_bins)
        x_labels = [f"{s}~{e}" for (s, e) in zip(bins[:-1], bins[1:])]
        sns.pointplot(data=data, x="bin", y="auc", linestyle="None", ax=self.ax)
        self.ax.set_xticks(ticks=list(range(n_bins)), labels=x_labels)
        self.ax.set_title(f"Binned AUC")
        self.ax.set_ylabel("AUC [activity*t]")
        self.canvas.draw()
