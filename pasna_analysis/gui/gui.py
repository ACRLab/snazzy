import json
from pathlib import Path
import sys

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QBrush, QColor, QKeySequence, QPen
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

from pasna_analysis import Experiment
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
from pasna_analysis.gui.model import Model
from pasna_analysis.gui.plot_window import PlotWindow
from pasna_analysis.gui.sidebar import RemovableSidebar, FixedSidebar
from pasna_analysis.gui.sliders import LabeledSlider


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = Model()
        self.moveable_width_bars = False
        self.show_peak_widths = True
        self.is_dragging_slider = False
        self.color_mode = False
        self.brushes = [QBrush(pg.mkColor("m"))]

        self.setWindowTitle("Pasna Analysis")
        self.setGeometry(100, 100, 1200, 600)

        self.paint_menu()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)

        placeholder = QLabel("To get started, open a directory with pasnascope output.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(placeholder)

    def change_group(self, i):
        self.model.set_curr_group(self.group_combo_box.itemText(i))

        self.clear_layout(self.bottom_layout)
        self.paint_graphs()
        self.render_trace()
        self.plot_all_traces()

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
        self.pw = PlotWindow(self.model.get_curr_group())
        self.pw.show()

    def display_compare_plots(self):
        groups = self.model.get_filtered_groups()
        self.cpw = ComparePlotWindow(groups)
        self.cpw.show()

    def display_embryo_movie(self):
        exp = self.model.get_curr_experiment()
        try:
            self.viewer = ImageSequenceViewer(exp.directory, exp)
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

    def show_notification(self, note_title, msg):
        notification = QMessageBox(self)
        notification.setWindowTitle(note_title)
        notification.setText(msg)
        notification.exec()

    def get_group_selector(self):
        if len(self.model.groups) > 1:
            select_group = QComboBox()
            select_group.setMaximumWidth(400)
            select_group.activated.connect(self.change_group)

            for i, group in enumerate(self.model.groups):
                select_group.insertItem(i, group)

            return select_group

        curr_group = self.model.get_curr_group()
        if len(curr_group) > 1:
            label_text = f"Group: {'-'.join(curr_group.keys())}"
            return QLabel(label_text)
        curr_exp = self.model.get_curr_experiment()
        label_text = f"Experiment: {curr_exp.name}"
        return QLabel(label_text)

    def paint_top_app_bar(self):
        top_app_bar_content = self.get_group_selector()

        if isinstance(top_app_bar_content, QComboBox):
            self.group_combo_box = top_app_bar_content

        self.top_app_bar.addWidget(top_app_bar_content)
        self.top_app_bar.addStretch()

        self.color_mode_btn = QPushButton("Colored Peaks")
        self.color_mode_btn.clicked.connect(self.toggle_color_mode)
        self.top_app_bar.addWidget(self.color_mode_btn)

        self.moveable_width_btn = QPushButton("Adjust widths")
        self.moveable_width_btn.clicked.connect(self.toggle_moveable_widths)
        self.top_app_bar.addWidget(self.moveable_width_btn)

        self.toggle_width_view_btn = QPushButton("View widths")
        self.toggle_width_view_btn.clicked.connect(self.toggle_width_view)
        self.top_app_bar.addWidget(self.toggle_width_view_btn)

        self.toggle_graph_btn = QPushButton("View all traces")
        self.toggle_graph_btn.setCheckable(True)
        self.toggle_graph_btn.clicked.connect(self.toggle_graph_view)
        self.top_app_bar.addWidget(self.toggle_graph_btn)

        self.clear_manual_data_btn = QPushButton("Clear manual data")
        self.clear_manual_data_btn.clicked.connect(self.clear_manual_data)
        self.top_app_bar.addWidget(self.clear_manual_data_btn)

    def toggle_color_mode(self):
        self.color_mode = not self.color_mode
        if self.color_mode:
            self.color_mode_btn.setText("Single color peaks")
            self.brushes = self.create_brushes_table()
        else:
            self.color_mode_btn.setText("Colored peaks")
            self.brushes = [QBrush(pg.mkColor("m"))]
        self.render_trace()

    def toggle_moveable_widths(self):
        self.moveable_width_bars = not self.moveable_width_bars
        ils = sorted(
            (
                item
                for item in self.plot_widget.get_items()
                if isinstance(item, pg.InfiniteLine)
            ),
            key=lambda il: il.peak_index,
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
            self.order_zero_slider = LabeledSlider("Order 0 min", 0, 0.8, 0.06, 0.005)
            self.order_one_slider = LabeledSlider("Order 1 min", 0, 0.1, 0.005, 0.0005)
            self.prominence_slider = LabeledSlider("Prominence", 0, 1, 0.06, 0.005)
            self.width_slider = LabeledSlider("Rel height", 0.7, 1, 0.92, 0.005)

            self.top_layout.addWidget(self.mpd_slider)
            self.top_layout.addWidget(self.order_zero_slider)
            self.top_layout.addWidget(self.order_one_slider)
            self.top_layout.addWidget(self.prominence_slider)
            self.top_layout.addWidget(self.width_slider)

            self.button = QPushButton("Apply Changes")
            self.button.clicked.connect(self.update_all_embs)
            self.top_layout.addWidget(self.button)

            self.calibrate_sliders()

    def toggle_emb_visibility(self, emb_name):
        exp = self.model.get_curr_experiment()
        if emb_name in self.model.to_remove[exp.name]:
            self.model.to_remove[exp.name].remove(emb_name)
        else:
            self.model.to_remove[exp.name].add(emb_name)

    def paint_graphs(self):
        self.single_graph_frame = QFrame()
        self.bottom_layout.addWidget(self.single_graph_frame)
        single_graph_layout = QHBoxLayout()
        self.single_graph_frame.setLayout(single_graph_layout)

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

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.sidebar)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(200)
        single_graph_layout.addWidget(scroll_area)
        # Sidebar end

        # Graph start
        self.plot_widget = InteractivePlotWidget()
        single_graph_layout.addWidget(self.plot_widget)
        self.plot_widget.hide()

        self.plot_widget.add_peak_fired.connect(self.add_peak)
        self.plot_widget.remove_peak_fired.connect(self.remove_peak)
        # Graph end

        # All traces start
        self.graph_scroll = QScrollArea()
        self.graph_scroll.setWidgetResizable(True)
        self.bottom_layout.addWidget(self.graph_scroll)

        self.graph_container = QWidget()
        self.graph_layout = QVBoxLayout()
        self.graph_container.setLayout(self.graph_layout)
        self.graph_scroll.setWidget(self.graph_container)

        self.graph_scroll.hide()
        # All traces end

    def paint_menu(self):
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

        display_movie_action = QAction("View embryo movie", self)
        display_movie_action.triggered.connect(self.display_embryo_movie)
        plot_menu.addAction(display_movie_action)

        display_plots_action = QAction("View plots", self)
        display_plots_action.triggered.connect(self.display_plots)
        plot_menu.addAction(display_plots_action)

        display_comp_plots_action = QAction("View comparison plots", self)
        display_comp_plots_action.triggered.connect(self.display_compare_plots)
        plot_menu.addAction(display_comp_plots_action)

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
            self.plot_all_traces()
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
        # TODO: this will remove more than one peak if they fall within wlen
        removed = trace.peak_idxes[target].tolist()
        trace.to_remove.extend(removed)
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
        peak = local_peak_at(x, trace.order_zero_savgol[window], wlen)
        trace.to_add.append(peak)
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

    def update_all_embs(self):
        """Calculates and paints again all peaks."""
        self.detect_peaks_all()
        self.render_trace()
        self.repaint_peaks()

    def detect_peaks_all(self):
        """Recalculates peak indices for all embryos.

        Persists peak detection params in `peak_detection_params.json`."""
        order_zero_min = self.order_zero_slider.value()
        order_one_min = self.order_one_slider.value()
        mpd = self.mpd_slider.value()
        prominence = self.prominence_slider.value()
        rel_height = self.width_slider.value()

        exp = self.model.get_curr_experiment()

        for emb in exp.embryos.values():
            emb.trace.detect_peaks(
                mpd,
                order_zero_min,
                order_one_min,
                prominence,
            )

        save_detection_params(
            pd_params_path=exp.pd_params_path,
            mpd=mpd,
            order0_min=order_zero_min,
            order1_min=order_one_min,
            prominence=prominence,
            rel_height=rel_height,
        )

    def open_directory(self):
        self.add_experiment_action.setEnabled(True)
        self.compare_experiment_action.setEnabled(True)
        self.model.set_initial_state()
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
        self.plot_all_traces()

    def plot_all_traces(self):
        group = self.model.get_filtered_group()
        self.scatter_items = {}
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

                self.scatter_items[emb.name] = scatter_plot_item

                plot_widget.plot(time, dff)
                if self.model.has_combined_experiments():
                    plot_widget.setTitle(f"{exp_name} - {emb.name}")
                else:
                    plot_widget.setTitle(emb.name)
                self.graph_layout.addWidget(plot_widget)

    def repaint_peaks(self):
        """Repaints the scatter items in the multi trace view."""
        exp = self.model.get_curr_experiment()
        for emb_name, scatter in self.scatter_items.items():
            if emb_name not in self.model.to_remove:
                trace = exp.embryos[emb_name].trace
                scatter.setData(trace.peak_times, trace.peak_amplitudes)

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
        self.order_zero_slider.slider.sliderPressed.connect(self.started_dragging)
        self.order_zero_slider.slider.sliderReleased.connect(self.stopped_dragging)
        self.order_one_slider.setValue(pd_params["order1_min"])
        self.order_one_slider.set_custom_slot(self.repaint_curr_emb)
        self.order_one_slider.slider.sliderPressed.connect(self.started_dragging)
        self.order_one_slider.slider.sliderReleased.connect(self.stopped_dragging)
        self.prominence_slider.setValue(pd_params["prominence"])
        self.prominence_slider.set_custom_slot(self.repaint_curr_emb)
        self.prominence_slider.slider.sliderPressed.connect(self.started_dragging)
        self.prominence_slider.slider.sliderReleased.connect(self.stopped_dragging)
        # it's too costly to repaint all peak widths, so I wont update them as we drag
        # the width_slider
        self.width_slider.setValue(pd_params["rel_height"])

    def started_dragging(self):
        self.is_dragging_slider = True

    def stopped_dragging(self):
        self.is_dragging_slider = False
        self.render_trace()

    def create_brushes_table(self):
        colormap = pg.colormap.get("PAL-relaxed_bright")
        colors = colormap.getLookupTable(nPts=6)
        return [QBrush(QColor(*color)) for color in colors]

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

        peak_times = trace.peak_times
        peak_amps = trace.peak_amplitudes
        # paint peaks
        brushes = [self.brushes[i % len(self.brushes)] for i in range(len(peak_times))]
        scatter_plot_item = pg.ScatterPlotItem(
            peak_times, peak_amps, size=8, brush=brushes, pen=QPen(Qt.PenStyle.NoPen)
        )
        # paint manual data
        if trace.to_add or trace.to_remove:
            ttad = np.array([*trace.to_add, *trace.to_remove])
            manual_times = time[ttad]
            manual_amps = dff[ttad]
            manual_scatter = pg.ScatterPlotItem(
                manual_times, manual_amps, size=10, brush=QColor("cyan")
            )
            self.plot_widget.addItem(manual_scatter)
        # paint trace
        self.plot_widget.addItem(scatter_plot_item)
        self.plot_widget.plot(time, dff)

        if self.model.has_combined_experiments():
            self.plot_widget.setTitle(f"{exp_name} - {emb_name}")
        else:
            self.plot_widget.setTitle(emb_name)

        if not self.show_peak_widths or self.is_dragging_slider:
            return
        # paint peak widths
        rel_height = self.width_slider.value()
        trace.compute_peak_bounds(rel_height)
        peak_bounds = trace.peak_bounds_indices.flatten()
        if peak_bounds.size == 0:
            return

        peak_bound_times = time[peak_bounds]
        brushes = [
            self.brushes[i % len(self.brushes)] for i in range(len(peak_bound_times))
        ]
        for i, idx in enumerate(peak_bound_times):
            if self.color_mode:
                il = pg.InfiniteLine(
                    idx,
                    movable=self.moveable_width_bars,
                    pen=pg.mkPen(brushes[i // 2].color()),
                )
            else:
                il = pg.InfiniteLine(idx, movable=self.moveable_width_bars)
            il.peak_index = i
            if self.moveable_width_bars:
                if i % 2 == 0:
                    il.addMarker("<|")
                else:
                    il.addMarker("|>")
            il.sigPositionChangeFinished.connect(self.change_peak_bounds)
            self.plot_widget.addItem(il)

    def change_peak_bounds(self, il_obj):
        trace = self.model.get_curr_trace()

        row, col = divmod(il_obj.peak_index, 2)
        trace.peak_bounds_indices[row, col] = il_obj.getXPos() // 6
        peak_bounds = trace.peak_bounds_indices[row].tolist()
        # cast values to int / str because that will be json dumped
        peak_bounds = [int(pb) for pb in peak_bounds]
        # indirectly the row represents the peak_index that this il is associated to
        peak_index = str(trace.peak_idxes[row])
        self.save_peak_pos(peak_bounds, peak_index)

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

    def clear_manual_data(self):
        """Removes all manual data from pd_params.json file.

        All manual data is stored under the 'embryos' key."""
        exp = self.model.get_curr_experiment()

        for emb in exp.embryos.values():
            emb.trace.to_add = []
            emb.trace.to_remove = []

        with open(exp.pd_params_path, "r") as f:
            config = json.load(f)

        del config["embryos"]

        with open(exp.pd_params_path, "w") as f:
            json.dump(config, f, indent=4)

        self.update_all_embs()
        self.show_notification("Clear manual data", "Manually added data was removed.")


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
