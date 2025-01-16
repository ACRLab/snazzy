from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from pasna_analysis import Experiment, Trace
from pasna_analysis.interactive_find_peaks import (
    get_initial_values,
    save_detection_params,
    local_peak_at,
    save_remove_peak,
    save_add_peak,
)

from pasna_analysis.gui.compare_plot_window import ComparePlotWindow
from pasna_analysis.gui.image_window import ImageSequenceViewer, ImageWindow
from pasna_analysis.gui.interactive_plot import InteractivePlotWidget
from pasna_analysis.gui.plot_window import PlotWindow
from pasna_analysis.gui.sidebar import RemovableSidebar, FixedSidebar
from pasna_analysis.gui.sliders import LabeledSlider


class Model:
    def __init__(self):
        self.initial_state()

    def initial_state(self):
        self.groups = {"group1": {}}
        self.curr_group = "group1"
        self.to_remove = {}
        self.curr_exp = None
        self.curr_emb_name = None

    def add_experiment(self, experiment: Experiment, group=None):
        if group is None:
            group = self.curr_group

        if group not in self.groups:
            raise ValueError("Group not found.")

        if experiment in self.groups[group]:
            raise ValueError("Experiment already added to this group.")

        self.groups[group][experiment.name] = experiment
        self.to_remove[experiment.name] = set()

        if self.curr_exp is None:
            self.curr_exp = experiment.name

        if self.curr_emb_name is None:
            emb_name = next(iter(experiment.embryos))
            self.curr_emb_name = emb_name

    def get_filtered_groups(self):
        groups = deepcopy(self.groups)
        for group_name, group in groups.items():
            for exp_name, exp in group.items():
                exp.embryos = self.get_filtered_embs(exp_name, group_name)
        return groups

    def get_filtered_embs(self, exp_name, group_name=None):
        exp = self.get_experiment(exp_name, group_name)
        if exp_name not in self.to_remove:
            return exp.embryos
        return {
            emb_name: emb
            for emb_name, emb in exp.embryos.items()
            if emb_name not in self.to_remove[exp_name]
        }

    def get_filtered_group(self):
        group = deepcopy(self.groups[self.curr_group])
        for exp_name, exp in group.items():
            exp.embryos = self.get_filtered_embs(exp_name)
        return group

    def set_curr_group(self, group=str):
        if group not in self.groups:
            raise ValueError("Group not found.")
        self.curr_group = group

        self.curr_exp = next(iter(self.groups[group]))
        curr_exp = self.get_curr_experiment()

        self.curr_emb_name = next(iter(curr_exp.embryos))

    def get_curr_experiment(self) -> Experiment:
        return self.groups[self.curr_group][self.curr_exp]

    def get_experiment(self, exp_name, group_name=None) -> Experiment:
        if group_name is None:
            curr_group = self.get_curr_group()
        else:
            curr_group = self.groups[group_name]
        return curr_group[exp_name]

    def get_curr_group(self) -> dict[str, Experiment]:
        return self.groups[self.curr_group]

    def get_curr_trace(self) -> Trace:
        exp = self.get_curr_experiment()
        return exp.embryos[self.curr_emb_name].trace

    def add_group(self, group):
        self.groups[group] = {}

    def has_combined_experiments(self):
        return len(self.get_curr_group()) > 1


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = Model()
        self.moveable_width_bars = False
        self.show_peak_widths = True
        self.is_dragging_slider = False

        self.setWindowTitle("Pasna Analysis")
        self.setGeometry(100, 100, 1200, 600)

        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("Open Directory", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_directory)
        file_menu.addAction(open_action)

        self.add_experiment_action = QAction("Add Experiment", self)
        self.add_experiment_action.triggered.connect(self.add_experiment)
        self.add_experiment_action.setEnabled(False)
        file_menu.addAction(self.add_experiment_action)

        self.compare_experiment_action = QAction("Compare with experiment", self)
        self.compare_experiment_action.triggered.connect(self.compare_experiments)
        self.compare_experiment_action.setEnabled(False)
        file_menu.addAction(self.compare_experiment_action)

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

        generate_comp_plots_action = QAction("View comparison plots", self)
        generate_comp_plots_action.triggered.connect(self.display_compare_plots)
        plot_menu.addAction(generate_comp_plots_action)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.placeholder = QLabel(
            "To get started, open a directory with pasnascope output."
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.placeholder)

    def change_group(self, i):
        self.model.set_curr_group(self.group_combo_box.itemText(i))

        self.clear_layout(self.bottom_layout)
        self.paint_graphs()
        self.render_trace()
        self.plot_graphs()

    def compare_experiments(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory:
            return
        directory = Path(directory)

        group_name, ok = QInputDialog.getText(self, "Enter Group Name", "Group Name:")
        if not ok:
            return
        if not group_name:
            group_name = f"group{len(self.models) + 1}"

        try:
            exp = Experiment(
                directory,
                first_peak_threshold=0,
                to_exclude=[],
                dff_strategy="local_minima",
            )
        except (FileNotFoundError, AssertionError):
            self.show_error_message(f"Could not read data from {directory}")
            return

        self.model.add_group(group_name)
        self.model.add_experiment(exp, group_name)

        self.clear_layout(self.top_app_bar)
        self.paint_top_app_bar()
        self.clear_layout(self.top_layout)
        self.paint_controls()

    def add_experiment(self):
        self._open_directory()

    def display_plots(self):
        group = self.model.get_filtered_group()
        embryos = [
            (emb, exp.name) for exp in group.values() for emb in exp.embryos.values()
        ]
        embryos, exp_names = list(zip(*embryos))

        self.pw = PlotWindow(embryos, exp_names, self.model.curr_group)
        self.pw.show()

    def display_compare_plots(self):
        groups = self.model.get_filtered_groups()
        self.cpw = ComparePlotWindow(groups)
        self.cpw.show()

    def display_embryo_movie(self):
        exp = self.model.get_curr_experiment()
        dff_traces = {name: e.trace.dff for (name, e) in exp.embryos.items()}
        try:
            self.viewer = ImageSequenceViewer(exp.directory, dff_traces)
        except FileNotFoundError as e:
            self.show_error_message(str(e))
            return
        self.viewer.show()

    def display_field_of_view(self):
        exp = self.model.get_curr_experiment()
        img_path = exp.directory / "emb_numbers.png"
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
        if layout is None:
            layout = self.layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())

    def get_top_bar_content(self):
        if len(self.model.groups) > 1:
            select_group = QComboBox()
            # TODO: pick a proper width for the combo box
            select_group.setMaximumWidth(400)
            select_group.activated.connect(self.change_group)

            for i, group in enumerate(self.model.groups):
                select_group.insertItem(i, group)

            return select_group
        curr_group = self.model.get_curr_group()
        if len(curr_group) > 1:
            label_text = f"Group: {' '.join(curr_group.keys())}"
            return QLabel(label_text)
        curr_exp = self.model.get_curr_experiment()
        label_text = f"Experiment: {curr_exp.name}"
        return QLabel(label_text)

    def paint_top_app_bar(self):
        top_app_bar_content = self.get_top_bar_content()

        if isinstance(top_app_bar_content, QComboBox):
            self.group_combo_box = top_app_bar_content

        self.top_app_bar.addWidget(top_app_bar_content)
        self.top_app_bar.addStretch()

    def toggle_moveable_widths(self):
        self.moveable_width_bars = not self.moveable_width_bars
        ils = (
            item
            for item in self.plot_widget.get_items()
            if isinstance(item, pg.InfiniteLine)
        )
        for i, il in enumerate(ils):
            if self.moveable_width_bars:
                il.setMovable(True)
                if i % 2 == 0:
                    il.addMarker("<|")
                else:
                    il.addMarker("|>")
            else:
                il.setMovable(False)
                il.clearMarkers()

    def toggle_width_view(self):
        self.show_peak_widths = not self.show_peak_widths
        self.render_trace()

    def paint_controls(self):
        # Sliders are only avaialable if a single experiment is open
        if len(self.model.groups) == 1 and not self.model.has_combined_experiments():
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

            self.moveable_width_btn = QPushButton("Adjust widths")
            self.moveable_width_btn.clicked.connect(self.toggle_moveable_widths)
            self.top_layout.addWidget(self.moveable_width_btn)

            self.toggle_width_view_btn = QPushButton("View widths")
            self.toggle_width_view_btn.clicked.connect(self.toggle_width_view)
            self.top_layout.addWidget(self.toggle_width_view_btn)

            self.calibrate_sliders()

        self.toggle_graph_btn = QPushButton("View all traces")
        self.toggle_graph_btn.setCheckable(True)
        self.toggle_graph_btn.clicked.connect(self.toggle_graph_view)
        self.top_layout.addWidget(self.toggle_graph_btn)

    def toggle_emb_visibility(self, emb_name):
        exp = self.model.get_curr_experiment()
        if emb_name in self.model.to_remove[exp.name]:
            self.model.to_remove[exp.name].remove(emb_name)
        else:
            self.model.to_remove[exp.name].add(emb_name)

    def paint_graphs(self):
        # Bottom layout end
        self.single_graph_frame = QFrame()
        self.bottom_layout.addWidget(self.single_graph_frame)
        self.single_graph_layout = QHBoxLayout()
        self.single_graph_frame.setLayout(self.single_graph_layout)

        # Sidebar start
        if not self.model.has_combined_experiments():
            exp = self.model.get_curr_experiment()
            accepted_embs = set(self.model.get_filtered_embs(exp.name).keys())
            removed_embs = set(self.model.to_remove[exp.name])
            self.sidebar = RemovableSidebar(
                self.render_trace,
                accepted_embs,
                removed_embs,
                exp.pd_params_path,
            )
            self.sidebar.emb_visibility_toggled.connect(self.toggle_emb_visibility)
        else:
            group = self.model.get_curr_group()
            exp_to_embs = {}
            for exp_name, exp in group.items():
                exp_to_embs[exp_name] = exp.embryos.keys()
            self.sidebar = FixedSidebar(exp_to_embs, self.render_trace)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.sidebar)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(200)
        self.single_graph_layout.addWidget(self.scroll_area)
        # Sidebar end

        # Graph start
        self.plot_widget = InteractivePlotWidget()
        self.single_graph_layout.addWidget(self.plot_widget)
        self.plot_widget.hide()

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

    def paint_main_view(self):
        self.top_app_bar = QHBoxLayout()
        self.layout.addLayout(self.top_app_bar)
        self.paint_top_app_bar()

        self.top_layout = QHBoxLayout()
        self.layout.addLayout(self.top_layout)
        self.paint_controls()

        self.bottom_layout = QHBoxLayout()
        self.layout.addLayout(self.bottom_layout)
        self.paint_graphs()

    def toggle_graph_view(self, checked):
        if checked:
            self.toggle_graph_btn.setText("View single trace")
            self.single_graph_frame.hide()
            # repaint graphs to make sure they are in sync with accepted embs
            self.clear_layout(self.graph_layout)
            self.plot_graphs()
            self.graph_scroll.show()
        else:
            self.toggle_graph_btn.setText("View all traces")
            self.graph_scroll.hide()
            self.single_graph_frame.show()

    def remove_peak(self, x, y):
        exp = self.model.get_curr_experiment()

        with open(exp.pd_params_path, "r") as f:
            config = json.load(f)
        if "embryos" not in config.keys():
            config["embryos"] = {}

        wlen = 10
        x = int(x / 6)

        trace = self.model.get_curr_trace()
        target = (trace.peak_idxes >= x - wlen) & (trace.peak_idxes <= x + wlen)
        # FIXME: this will remove more than one peak if they fall within wlen
        removed = trace.peak_idxes[target].tolist()
        new_arr = trace.peak_idxes[~target]
        trace.peak_idxes = new_arr

        save_remove_peak(self.model.curr_emb_name, config, removed, x, wlen)

        with open(exp.pd_params_path, "w") as f:
            json.dump(config, f, indent=4)

        self.render_trace()

    def save_peak_pos(self, peak_widths, peak_index):
        exp = self.model.get_curr_experiment()

        with open(exp.pd_params_path, "r") as f:
            config = json.load(f)
        if "embryos" not in config:
            config["embryos"] = {}

        emb_name = self.model.curr_emb_name
        if not emb_name in config["embryos"]:
            config["embryos"][emb_name] = {
                "wlen": 10,
                "manual_peaks": [],
                "manual_remove": [],
                "manual_widths": {},
            }

        config["embryos"][emb_name]["manual_widths"][peak_index] = peak_widths

        with open(exp.pd_params_path, "w") as f:
            json.dump(config, f, indent=4)

    def add_peak(self, x, y):
        exp = self.model.get_curr_experiment()

        with open(exp.pd_params_path, "r") as f:
            config = json.load(f)
        if "embryos" not in config:
            config["embryos"] = {}

        wlen = 10

        x = int(x / 6)

        trace = self.model.get_curr_trace()
        window = slice(x - wlen, x + wlen)
        peak = local_peak_at(x, trace.dff[window], wlen)
        new_arr = np.append(trace.peak_idxes, peak)
        new_arr.sort()
        trace.peak_idxes = new_arr

        save_add_peak(self.model.curr_emb_name, config, peak, wlen)

        with open(exp.pd_params_path, "w") as f:
            json.dump(config, f, indent=4)

        self.render_trace()

    def repaint_curr_emb(self):
        """Repaints peaks for the trace currently being displayed.

        This funciton runs after any of the peak detection params sliders change."""
        order_zero_min = self.order_zero_slider.value()
        order_one_min = self.order_one_slider.value()
        mpd = self.mpd_slider.value()
        prominence = self.prominence_slider.value()

        trace = self.model.get_curr_trace()
        trace.detect_peaks(
            mpd,
            order_zero_min,
            order_one_min,
            prominence,
        )
        self.render_trace()

    def detect_peaks_all(self):
        """Recalculates peak indices for all embryos.

        Persists peak detection params in `peak_detection_params.json`."""
        order_zero_min = self.order_zero_slider.value()
        order_one_min = self.order_one_slider.value()
        mpd = self.mpd_slider.value()
        prominence = self.prominence_slider.value()

        exp = self.model.get_curr_experiment()

        for emb in exp.embryos.values():
            emb.trace.detect_peaks(
                mpd,
                order_zero_min,
                order_one_min,
                prominence,
            )

        self.render_trace()
        self.repaint_peaks()

        save_detection_params(
            pd_params_path=exp.pd_params_path,
            mpd=mpd,
            order0_min=order_zero_min,
            order1_min=order_one_min,
            prominence=prominence,
        )

    def open_directory(self):
        self.add_experiment_action.setEnabled(True)
        self.compare_experiment_action.setEnabled(True)
        self.model.initial_state()
        self._open_directory()

    def _open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory:
            return
        directory = Path(directory)
        self.clear_layout()

        try:
            exp = Experiment(
                directory,
                first_peak_threshold=0,
                to_exclude=[],
                dff_strategy="local_minima",
            )
        except (FileNotFoundError, AssertionError):
            self.show_error_message(f"Could not read data from {directory}")
            return

        self.model.add_experiment(exp)

        self.paint_main_view()
        self.render_trace()
        self.plot_graphs()

    def request_repaint_graphs(self):
        self.clear_layout(self.graph_layout)
        self.plot_graphs()

    def plot_graphs(self):
        group = self.model.get_filtered_group()
        for exp_name, exp in group.items():
            for emb in exp.embryos.values():
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
                if self.model.has_combined_experiments():
                    plot_widget.setTitle(f"{exp_name} - {emb.name}")
                else:
                    plot_widget.setTitle(emb.name)
                self.graph_layout.addWidget(plot_widget)

    def repaint_peaks(self):
        exp = self.model.get_curr_experiment()
        for scatter, emb in zip(self.scatter_items, exp.embryos.values()):
            scatter.setData(emb.trace.peak_times, emb.trace.peak_amplitudes)

    def calibrate_sliders(self):
        """Adjusts the sliders based on pd_params.json.

        The sliders should not be available when more than one experiment is loaded."""
        if self.model.has_combined_experiments():
            return
        exp = self.model.get_curr_experiment()
        pd_params = get_initial_values(exp.pd_params_path)

        self.mpd_slider.setValue(pd_params["mpd"])
        self.mpd_slider.set_custom_slot(self.repaint_curr_emb)
        self.mpd_slider.slider.sliderPressed.connect(self.started_dragging)
        self.mpd_slider.slider.sliderReleased.connect(self.stopped_dragging)
        self.order_zero_slider.setValue(pd_params["order0_min"])
        self.order_zero_slider.set_custom_slot(self.repaint_curr_emb)
        self.order_one_slider.slider.sliderPressed.connect(self.started_dragging)
        self.order_one_slider.slider.sliderReleased.connect(self.stopped_dragging)
        self.order_one_slider.setValue(pd_params["order1_min"])
        self.order_one_slider.set_custom_slot(self.repaint_curr_emb)
        self.order_one_slider.slider.sliderPressed.connect(self.started_dragging)
        self.order_one_slider.slider.sliderReleased.connect(self.stopped_dragging)
        self.prominence_slider.setValue(pd_params["prominence"])
        self.prominence_slider.set_custom_slot(self.repaint_curr_emb)
        self.prominence_slider.slider.sliderPressed.connect(self.started_dragging)
        self.prominence_slider.slider.sliderReleased.connect(self.stopped_dragging)

    def started_dragging(self):
        self.is_dragging_slider = True

    def stopped_dragging(self):
        self.is_dragging_slider = False
        self.render_trace()

    def render_trace(self, emb_name=None, exp_name=None):
        if emb_name and exp_name:
            exp = self.model.get_experiment(exp_name)
        if not exp_name:
            exp = self.model.get_curr_experiment()
            exp_name = exp.name
        if not emb_name:
            emb_name = self.model.curr_emb_name

        self.model.curr_emb_name = emb_name
        self.plot_widget.clear()
        self.plot_widget.show()

        trace = exp.embryos[emb_name].trace
        time = trace.time[: trace.trim_idx]
        dff = trace.dff[: trace.trim_idx]

        peak_amps = trace.peak_amplitudes
        peak_times = trace.peak_times

        scatter_plot_item = pg.ScatterPlotItem(size=8, brush=pg.mkColor("m"))
        scatter_plot_item.setData(peak_times, peak_amps)
        self.plot_widget.addItem(scatter_plot_item)
        self.plot_widget.plot(time, dff)

        if self.model.has_combined_experiments():
            self.plot_widget.setTitle(f"{exp_name} - {emb_name}")
        else:
            self.plot_widget.setTitle(emb_name)

        if not self.show_peak_widths:
            return

        if not self.is_dragging_slider:
            trace.compute_peak_bounds()
            peak_bounds = trace.peak_bounds_indices.flatten()
            peak_bound_times = time[peak_bounds]

            for i, idx in enumerate(peak_bound_times):
                il = pg.InfiniteLine(idx, movable=self.moveable_width_bars)
                il.peak_index = i
                if self.moveable_width_bars:
                    if i % 2 == 0:
                        il.addMarker("<|")
                    else:
                        il.addMarker("|>")
                il.sigPositionChangeFinished.connect(self.change_peak_pos)
                self.plot_widget.addItem(il)

    def change_peak_pos(self, il_obj):
        trace = self.model.get_curr_trace()

        row, col = divmod(il_obj.peak_index, 2)
        trace.peak_bounds_indices[row, col] = il_obj.getXPos() // 6
        peak_bounds = trace.peak_bounds_indices[row].tolist()
        # cast values to int / str because that will be json dumped
        peak_bounds = [int(pb) for pb in peak_bounds]
        # indirectly the row represents the peak_index that this il is associated to
        peak_index = str(trace.peak_idxes[row])
        self.save_peak_pos(peak_bounds, peak_index)


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
