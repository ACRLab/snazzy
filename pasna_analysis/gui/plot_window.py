from datetime import datetime

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget

from pasna_analysis import Experiment, utils


matplotlib.use("QtAgg")


class PlotWindow(QWidget):
    def __init__(self, group: dict[str, Experiment], group_name: str):
        super().__init__()

        self.group = group
        self.embryos = [emb for exp in group.values() for emb in exp.embryos.values()]

        self.setWindowTitle(f"Plots - {group_name}")

        layout = QVBoxLayout()
        self.save_all_btn = QPushButton("Save all plots")
        self.save_all_btn.clicked.connect(self.save_all_plots)
        layout.addWidget(self.save_all_btn)

        plots_container = QHBoxLayout()
        layout.addLayout(plots_container)

        self.sidebar = QVBoxLayout()
        self.sidebar.setAlignment(Qt.AlignmentFlag.AlignTop)
        plots_container.addLayout(self.sidebar)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(10, 6), layout="constrained"))
        self.ax = self.canvas.figure.subplots()

        plots_container.addWidget(self.canvas)
        self.setLayout(layout)

        self.btns = {
            "Peaks": self.plot_peaks,
            "Area Under the Curve": self.plot_AUC,
        }

        self.create_buttons()

    def save_all_plots(self):
        for exp in self.group.values():
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

    def create_buttons(self):
        """Populates buttons in the sidebar.

        `btns` should receive a dict[str, Callable], where the str is the btn
        text and Callable is a function that adds a plot to `self.ax`.
        """
        for label, fn in self.btns.items():
            button = QPushButton(label)
            button.clicked.connect(fn)
            button.setMaximumWidth(150)
            self.sidebar.addWidget(button)

    def plot_peaks(self, save=False, save_dir=None):
        """Peak times for all embryos.

        Embryos are represented as horizontal lines."""
        self.ax.clear()
        times = []
        for emb in self.embryos:
            trace = emb.trace
            times.append(
                [t / 60 for t in trace.peak_times if t < trace.time[trace.trim_idx]]
            )
        emb_names = [emb.name for emb in self.embryos]
        style = {"marker": ".", "linestyle": "dashed", "linewidth": 0.5}
        for emb_name, (i, time) in zip(emb_names, enumerate(times)):
            self.ax.plot(time, [i] * len(time), label=emb_name, **style)
        self.ax.legend()
        self.ax.set_title(f"Peak times")
        self.ax.set_xlabel("time (mins)")
        self.ax.set_ylabel("emb")
        self.ax.set_yticks([])

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "peak_times.png")

    def plot_AUC(self, save=False, save_dir=None):
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

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "area_under_curve.png")
