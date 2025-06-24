from datetime import datetime

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import numpy as np
import seaborn as sns

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget

from pasna_analysis import Embryo, Trace, utils


matplotlib.use("QtAgg")


class PlotWindow(QWidget):
    def __init__(self, embryos: list[Embryo], group_name: str, curr_trace: Trace):
        super().__init__()

        self.curr_trace = curr_trace
        self.embryos = embryos

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

        plots_container.addWidget(self.canvas)
        self.setLayout(layout)

        self.btns = {
            "Peaks": self.plot_peaks,
            "Area Under the Curve": self.plot_AUC,
            "Burst correlogram": self.plot_burst_correlogram,
            "Burst high freq content": self.plot_high_frequency_distances,
        }

        self.create_buttons()

    def clear_plot(self):
        self.canvas.figure.clear()
        self.ax = self.canvas.figure.add_subplot(111)

    def save_all_plots(self):
        for exp in self.group.values():
            # exp_dir = exp.pd_params_path.parent
            exp_dir = exp.directory
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
        self.clear_plot()

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

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "peak_times.png")

    def plot_AUC(self, save=False, save_dir=None):
        """Binned area under the curve."""
        self.clear_plot()

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

    def get_normalized_interval(self, signal, peak_index):
        left_interval = 10
        right_interval = 40
        s = peak_index - left_interval
        e = peak_index + right_interval

        window = signal[s:e]
        return (window - np.mean(window)) / (np.std(window) + 1e-6)

    def plot_burst_correlogram(self, save=False, save_dir=None):
        self.clear_plot()

        dff = self.curr_trace.dff.copy()
        peak_indices = self.curr_trace.peak_idxes

        n = len(peak_indices)
        corr_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                norm_signal_i = self.get_normalized_interval(dff, peak_indices[i])
                norm_signal_j = self.get_normalized_interval(dff, peak_indices[j])

                corr_matrix[i][j] = np.corrcoef(norm_signal_i, norm_signal_j)[0, 1]

        sns.heatmap(
            corr_matrix,
            cmap="viridis",
            square=True,
            xticklabels=False,
            yticklabels=False,
            ax=self.ax,
        )
        self.ax.set_title("Correlogram of Burst Similarities")
        self.ax.set_xlabel("Burst Index")
        self.ax.set_ylabel("Burst Index")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "area_under_curve.png")

    def plot_high_frequency_distances(self, save=False, save_dir=None):
        self.clear_plot()

        hi_pass = self.curr_trace.get_filtered_signal(0.02, low_pass=False)

        features = []
        for s, e in self.curr_trace.peak_bounds_indices:
            feature = []
            rms = np.sqrt(np.mean(np.power(hi_pass[s:e], 2)))
            feature.append(rms)
            features.append(feature)

        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        dist_matrix = squareform(pdist(features_scaled, metric="euclidean"))

        sns.heatmap(
            dist_matrix,
            cmap="viridis",
            square=True,
            xticklabels=False,
            yticklabels=False,
            ax=self.ax,
        )
        self.ax.set_title("Frequency content by burst")
        self.ax.set_xlabel("Burst Index")
        self.ax.set_ylabel("Burst Index")

        if not save:
            self.canvas.draw()
        else:
            if save_dir is None:
                raise ValueError("Cannot save the image: path to save not provided.")
            self.canvas.print_figure(save_dir / "area_under_curve.png")
