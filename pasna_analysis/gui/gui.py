import json
from pathlib import Path
import sys

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from pasna_analysis import Experiment
from pasna_analysis.interactive_find_peaks import (
    get_initial_values,
    save_detection_params,
    local_peak_at,
    save_remove_peak,
    save_add_peak,
)
from pasna_analysis.gui.image_window import ImageSequenceViewer, ImageWindow
from pasna_analysis.gui.interactive_plot import InteractivePlotWidget
from pasna_analysis.gui.sliders import LabeledSlider
from pasna_analysis.gui.plot_window import PlotWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pasna Analysis")
        self.setGeometry(100, 100, 1200, 600)

        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("Open Directory", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_directory)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        plot_menu = menu_bar.addMenu("&Plots")
        display_FOV_action = QAction("View all embryos", self)
        display_FOV_action.triggered.connect(self.display_field_of_view)
        plot_menu.addAction(display_FOV_action)

        display_embryo_action = QAction("View embryo raw data", self)
        display_embryo_action.triggered.connect(self.display_embryo_movie)
        plot_menu.addAction(display_embryo_action)

        generate_plots_action = QAction("View plots", self)
        generate_plots_action.triggered.connect(self.display_plots)
        plot_menu.addAction(generate_plots_action)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.placeholder = QLabel(
            "To get started, open a directory with pasnascope output."
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.placeholder)

    def display_plots(self):
        self.pw = PlotWindow(self.exp.embryos, self.exp.name)
        self.pw.show()

    def display_embryo_movie(self):
        dff_traces = {
            emb_name: e.trace.dff for (emb_name, e) in self.exp.embryos.items()
        }
        try:
            self.viewer = ImageSequenceViewer(self.directory, dff_traces)
        except FileNotFoundError as e:
            self.show_error_message(str(e))
            return
        self.viewer.show()

    def display_field_of_view(self):
        img_path = self.directory / "emb_numbers.png"
        try:
            self.image_window = ImageWindow(str(img_path))
        except FileNotFoundError as e:
            self.show_error_message(str(e))
            return
        self.image_window.show()

    def show_error_message(self, msg):
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(msg)
        error_dialog.exec()

    def clear_layout(self, layout=None):
        if not layout:
            layout = self.layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())

    def paint_main_view(self):
        # Top layout start (sliders)
        self.top_layout = QHBoxLayout()
        self.layout.addLayout(self.top_layout)

        self.mpd_slider = LabeledSlider("Minimum peak distance", 10, 300, 70)
        self.order_zero_slider = LabeledSlider("Order 0 min", 0, 0.5, 0.06, 0.005)
        self.order_one_slider = LabeledSlider("Order 1 min", 0, 0.1, 0.005, 0.0005)
        self.prominence_slider = LabeledSlider("Prominence", 0, 1, 0.06, 0.005)

        self.top_layout.addWidget(self.mpd_slider)
        self.top_layout.addWidget(self.order_zero_slider)
        self.top_layout.addWidget(self.order_one_slider)
        self.top_layout.addWidget(self.prominence_slider)

        self.button = QPushButton("Apply Changes")
        self.button.clicked.connect(self.detect_peaks_all)
        self.top_layout.addWidget(self.button)

        self.toggle_graph_btn = QPushButton("View all traces")
        self.toggle_graph_btn.setCheckable(True)
        self.toggle_graph_btn.clicked.connect(self.toggle_graph_view)
        self.top_layout.addWidget(self.toggle_graph_btn)
        # Top layout end (sliders)

        # Bottom layout start: sidebar and graph container
        self.bottom_layout = QHBoxLayout()
        self.layout.addLayout(self.bottom_layout)
        # Bottom layout end

        self.single_graph_frame = QFrame()
        self.bottom_layout.addWidget(self.single_graph_frame)
        self.single_graph_layout = QHBoxLayout()
        self.single_graph_frame.setLayout(self.single_graph_layout)

        # Sidebar start
        self.sidebar = QWidget()
        self.sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(self.sidebar_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.sidebar)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(150)
        self.single_graph_layout.addWidget(self.scroll_area)
        # Sidebar end

        # Graph start
        self.plot_widget = InteractivePlotWidget()
        self.single_graph_layout.addWidget(self.plot_widget)
        self.plot_widget.hide()

        self.scatter_plot_item = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkColor("m"),
        )
        self.plot_widget.addItem(self.scatter_plot_item)
        self.plot_widget.add_peak_fired.connect(self.add_peak)
        self.plot_widget.remove_peak_fired.connect(self.remove_peak)
        # Graph end

        # Multi graphs
        self.graph_scroll = QScrollArea()
        self.graph_scroll.setWidgetResizable(True)
        self.bottom_layout.addWidget(self.graph_scroll)
        self.scatter_items = []

        self.graph_container = QWidget()
        self.graph_layout = QVBoxLayout()
        self.graph_container.setLayout(self.graph_layout)
        self.graph_scroll.setWidget(self.graph_container)

        self.graph_scroll.hide()
        # Multi graphs end

    def toggle_graph_view(self, checked):
        if checked:
            self.toggle_graph_btn.setText("View single trace")
            self.single_graph_frame.hide()
            self.graph_scroll.show()
        else:
            self.toggle_graph_btn.setText("View all traces")
            self.graph_scroll.hide()
            self.single_graph_frame.show()

    def remove_peak(self, x, y):
        pd_params_path = self.directory / "peak_detection_params.json"

        with open(pd_params_path, "r") as f:
            config = json.load(f)
        if "embryos" not in config.keys():
            config["embryos"] = {}

        wlen = 10
        x = int(x / 6)

        trace = self.exp.embryos[self.curr_emb_name].trace
        target = (trace.peak_idxes >= x - wlen) & (trace.peak_idxes <= x + wlen)
        # FIXME: this will remove more than one peak if they fall within wlen
        removed = trace.peak_idxes[target].tolist()
        new_arr = trace.peak_idxes[~target]
        trace.peak_idxes = new_arr

        save_remove_peak(self.curr_emb_name, config, removed, x, wlen)

        with open(pd_params_path, "w") as f:
            json.dump(config, f, indent=4)

        self.render_trace(self.curr_emb_name)

    def add_peak(self, x, y):
        pd_params_path = self.directory / "peak_detection_params.json"

        with open(pd_params_path, "r") as f:
            config = json.load(f)
        if "embryos" not in config:
            config["embryos"] = {}

        wlen = 10

        x = int(x / 6)

        trace = self.exp.embryos[self.curr_emb_name].trace
        window = slice(x - wlen, x + wlen)
        peak = local_peak_at(x, trace.dff[window], wlen)
        new_arr = np.append(trace.peak_idxes, peak)
        new_arr.sort()
        trace.peak_idxes = new_arr

        save_add_peak(self.curr_emb_name, config, peak, wlen)

        with open(pd_params_path, "w") as f:
            json.dump(config, f, indent=4)

        self.render_trace(self.curr_emb_name)

    def repaint_curr_emb(self):
        """Repaints peaks for the trace currently being displayed.

        This funciton runs after any of the peak detection params sliders change."""
        order_zero_min = self.order_zero_slider.value()
        order_one_min = self.order_one_slider.value()
        mpd = self.mpd_slider.value()
        prominence = self.prominence_slider.value()

        curr_trace = self.exp.embryos[self.curr_emb_name].trace
        curr_trace.detect_peaks(
            mpd,
            order_zero_min,
            order_one_min,
            prominence,
        )
        self.render_trace(self.curr_emb_name)

    def detect_peaks_all(self):
        """Recalculates peak indices for all embryos.

        Persists peak detection params in `peak_detection_params.json`."""
        order_zero_min = self.order_zero_slider.value()
        order_one_min = self.order_one_slider.value()
        mpd = self.mpd_slider.value()
        prominence = self.prominence_slider.value()

        for emb in self.exp.embryos.values():
            emb.trace.detect_peaks(
                mpd,
                order_zero_min,
                order_one_min,
                prominence,
            )

        self.render_trace(self.curr_emb_name)
        self.repaint_peaks()

        pd_params_path = self.directory / "peak_detection_params.json"
        save_detection_params(
            pd_params_path=pd_params_path,
            mpd=mpd,
            order0_min=order_zero_min,
            order1_min=order_one_min,
            prominence=prominence,
        )

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory:
            return
        self.directory = Path(directory)
        self.clear_layout()

        try:
            self.exp = Experiment(
                self.directory,
                first_peak_threshold=0,
                to_exclude=[],
                dff_strategy="local_minima",
            )
        except (FileNotFoundError, AssertionError):
            self.show_error_message(f"Could not read data from {self.directory}")
            return

        self.curr_emb_name = next(iter(self.exp.embryos))

        self.paint_main_view()
        self.render_trace(self.curr_emb_name)
        self.plot_graphs()
        self.add_sidebar_buttons([emb_name for emb_name in self.exp.embryos])
        self.calibrate_sliders()

    def plot_graphs(self):
        for emb in self.exp.embryos.values():
            plot_widget = pg.PlotWidget()
            plot_widget.setMinimumHeight(200)

            trace = emb.trace
            time = trace.time[: trace.trim_idx]
            dff = trace.dff[: trace.trim_idx]

            peak_amps = trace.peak_amplitudes
            peak_times = trace.peak_times

            scatter_plot_item = pg.ScatterPlotItem(
                size=8,
                brush=pg.mkColor("m"),
            )
            scatter_plot_item.setData(peak_times, peak_amps)

            plot_widget.addItem(scatter_plot_item)

            self.scatter_items.append(scatter_plot_item)

            plot_widget.plot(time, dff)
            plot_widget.setTitle(emb.name)
            self.graph_layout.addWidget(plot_widget)

    def repaint_peaks(self):
        for scatter, emb in zip(self.scatter_items, self.exp.embryos.values()):
            scatter.setData(emb.trace.peak_times, emb.trace.peak_amplitudes)

    def calibrate_sliders(self):
        pd_params = get_initial_values(self.directory / "peak_detection_params.json")
        print(f"Looking for pd.json at {pd_params}")

        self.mpd_slider.setValue(pd_params["mpd"])
        self.mpd_slider.set_custom_slot(self.repaint_curr_emb)
        self.order_zero_slider.setValue(pd_params["order0_min"])
        self.order_zero_slider.set_custom_slot(self.repaint_curr_emb)
        self.order_one_slider.setValue(pd_params["order1_min"])
        self.order_one_slider.set_custom_slot(self.repaint_curr_emb)
        self.prominence_slider.setValue(pd_params["prominence"])
        self.prominence_slider.set_custom_slot(self.repaint_curr_emb)

    def add_sidebar_buttons(self, emb_names):
        for emb_name in emb_names:
            btn = QPushButton(emb_name)
            btn.clicked.connect(lambda checked, name=emb_name: self.render_trace(name))
            self.sidebar_layout.addWidget(btn)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.sidebar_layout.addWidget(spacer)

    def render_trace(self, emb_name):
        self.curr_emb_name = emb_name
        self.plot_widget.clear()
        self.plot_widget.show()

        trace = self.exp.embryos[emb_name].trace
        time = trace.time[: trace.trim_idx]
        dff = trace.dff[: trace.trim_idx]

        peak_amps = trace.peak_amplitudes
        peak_times = trace.peak_times

        self.scatter_plot_item.setData(peak_times, peak_amps)
        self.plot_widget.addItem(self.scatter_plot_item)
        self.plot_widget.plot(time, dff)
        self.plot_widget.setTitle(emb_name)


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
