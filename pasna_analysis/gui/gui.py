from functools import partial
from pathlib import Path
import sys

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThreadPool
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

from pasna_analysis.gui import (
    ComparePlotWindow,
    ExperimentParamsDialog,
    FixedSidebar,
    ImageSequenceViewer,
    ImageWindow,
    InteractivePlotWidget,
    LabeledSlider,
    Model,
    PlotWindow,
    RemovableSidebar,
    Worker,
)

from pasna_analysis import Config


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = Model()
        self.moveable_width_bars = False
        self.show_peak_widths = False
        self.is_dragging_slider = False
        self.color_mode = False
        self.brushes = [QBrush(pg.mkColor("m"))]
        self.threadpool = QThreadPool()
        self.use_dev_time = False
        self.filtered_dff = None
        self.display_filtered_dff = False

        self.setWindowTitle("Pasna Analysis")
        self.setGeometry(100, 100, 1200, 600)

        self.paint_menu()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)

        self.placeholder = QLabel(
            "To get started, open a directory with pasnascope output."
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.placeholder)

    def render_next_trace(self, forward):
        if self.model.curr_exp is None:
            return
        curr_emb = self.model.curr_emb_name
        embs = self.model.get_filtered_embs(self.model.curr_exp)
        emb_names = list(embs.keys())
        curr_emb_index = emb_names.index(curr_emb)
        if forward:
            next_idx = (curr_emb_index + 1) % len(emb_names)
        else:
            next_idx = (curr_emb_index - 1) % len(emb_names)
        next_emb = emb_names[next_idx]
        self.render_trace(next_emb)

    def _open_directory(self, is_new_group, should_reset_model=False):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory:
            return
        directory = Path(directory)

        config = Config(directory)
        exp_params = config.get_exp_params()
        dff_strategy = config.data["pd_params"].get("dff_strategy", "")

        group_name = None
        if is_new_group:
            group_name, ok = QInputDialog.getText(self, "New Group", "Group Name:")
            if not ok:
                return
            if not group_name:
                group_name = f"group{len(self.model.groups) + 1}"

        dialog_params = {**exp_params, "dff_strategy": dff_strategy}
        dialog = ExperimentParamsDialog(
            dialog_params, exp_path=config.exp_path, parent=self
        )
        if not dialog.exec():
            return

        dialog_values = dialog.get_values()
        exp_params = {k: v for k, v in dialog_values.items() if k in exp_params}
        pd_params = {"dff_strategy": dialog_values["dff_strategy"]}
        new_config = {"exp_params": exp_params, "pd_params": pd_params}
        config.update_params(new_config)

        config.save_params()

        if should_reset_model:
            self.model.set_initial_state()

        if group_name:
            self.model.add_group(group_name)

        try:
            self.placeholder.setText("Loading data..")
            self.placeholder.repaint()
        except RuntimeError:
            pass

        worker = Worker(
            self.model.create_experiment,
            config=config,
            group_name=group_name,
        )
        worker.signals.result.connect(self.update_UI)
        worker.signals.error.connect(self.handle_open_err)
        self.threadpool.start(worker)

    def handle_open_err(self, err: Exception):
        try:
            self.placeholder.setText(
                "To get started, open a directory with pasnascope output."
            )
        except RuntimeError:
            pass
        self.show_error_message(str(err))

    def update_UI(self):
        self.clear_layout()
        self.add_experiment_action.setEnabled(True)
        self.compare_experiment_action.setEnabled(True)
        self.paint_main_view()
        self.render_trace()
        self.plot_all_traces()

    def open_directory(self):
        self._open_directory(is_new_group=True, should_reset_model=True)

    def compare_experiments(self):
        self._open_directory(is_new_group=True)

    def add_experiment(self):
        self._open_directory(is_new_group=False)

    def change_group(self, i):
        self.model.set_curr_group(self.group_combo_box.itemText(i))

        self.clear_layout(self.bottom_layout)
        self.paint_graphs()
        self.render_trace()
        self.plot_all_traces()

    def display_plots(self):
        group = self.model.get_filtered_group()
        curr_trace = self.model.get_curr_trace()
        self.pw = PlotWindow(group, self.model.curr_group, curr_trace)
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

        label_text = self.model.curr_group
        return QLabel(label_text)

    def paint_top_app_bar(self):
        top_app_bar_content = self.get_group_selector()

        if isinstance(top_app_bar_content, QComboBox):
            self.group_combo_box = top_app_bar_content

        self.top_app_bar.addWidget(top_app_bar_content)
        self.top_app_bar.addStretch()

        self.show_ifft_btn = QPushButton("Show filtered dff")
        self.show_ifft_btn.clicked.connect(self.toggle_display_filtered_dff)
        self.top_app_bar.addWidget(self.show_ifft_btn)

        self.color_mode_btn = QPushButton("Colored Peaks")
        self.color_mode_btn.clicked.connect(self.toggle_color_mode)
        self.top_app_bar.addWidget(self.color_mode_btn)

        self.toggle_graph_btn = QPushButton("View all traces")
        self.toggle_graph_btn.setCheckable(True)
        self.toggle_graph_btn.clicked.connect(self.toggle_graph_view)
        self.top_app_bar.addWidget(self.toggle_graph_btn)

        if len(self.model.groups) == 1 and not self.model.has_combined_experiments():
            self.toggle_width_view_btn = QPushButton("View widths")
            self.toggle_width_view_btn.setCheckable(True)
            self.toggle_width_view_btn.clicked.connect(self.toggle_width_view)
            self.top_app_bar.addWidget(self.toggle_width_view_btn)

            self.moveable_width_btn = QPushButton("Adjust widths")
            self.moveable_width_btn.setCheckable(True)
            self.moveable_width_btn.clicked.connect(self.toggle_moveable_widths)
            self.top_app_bar.addWidget(self.moveable_width_btn)

            self.clear_manual_data_btn = QPushButton("Clear manual data")
            self.clear_manual_data_btn.clicked.connect(self.clear_manual_data)
            self.top_app_bar.addWidget(self.clear_manual_data_btn)

        self.toggle_dev_time_btn = QPushButton("Dev time")
        self.toggle_dev_time_btn.clicked.connect(self.toggle_dev_time)
        self.top_app_bar.addWidget(self.toggle_dev_time_btn)

    def toggle_dev_time(self):
        self.use_dev_time = not self.use_dev_time
        if self.use_dev_time:
            self.toggle_dev_time_btn.setText("Time")
        else:
            self.toggle_dev_time_btn.setText("Dev time")
        self.update_all_embs()

    def toggle_display_filtered_dff(self):
        self.display_filtered_dff = not self.display_filtered_dff
        if self.display_filtered_dff:
            self.show_ifft_btn.setText("Hide filtered dff")
        else:
            self.show_ifft_btn.setText("Show filtered dff")
        self.render_trace()

    def toggle_color_mode(self):
        self.color_mode = not self.color_mode
        if self.color_mode:
            self.color_mode_btn.setText("Single color peaks")
            self.brushes = self.create_brushes_table()
        else:
            self.color_mode_btn.setText("Colored peaks")
            self.brushes = [QBrush(pg.mkColor("m"))]
        self.render_trace()

    def toggle_moveable_widths(self, check):
        self.moveable_width_bars = check
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

    def toggle_width_view(self, checked):
        self.show_peak_widths = checked
        self.render_trace()

    def paint_controls(self):
        # Sliders are only avaialable if a single experiment is open
        if len(self.model.groups) == 1 and not self.model.has_combined_experiments():
            self.freq_slider = LabeledSlider(
                "Frequency cutoff",
                min_value=0.0001,
                max_value=0.005,
                initial_value=0.0025,
                step_size=0.0001,
            )

            self.rel_h_slider = LabeledSlider(
                "Relative height",
                min_value=0.3,
                max_value=1,
                initial_value=0.9,
                step_size=0.01,
            )

            self.top_layout.addWidget(self.freq_slider)
            self.top_layout.addWidget(self.rel_h_slider)

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
            accepted_embs = set(self.model.get_filtered_emb_numbers(exp.name))
            removed_embs = set(self.model.to_remove[exp.name])
            self.sidebar = RemovableSidebar(
                self.render_trace,
                accepted_embs,
                removed_embs,
            )
            self.sidebar.emb_visibility_toggled.connect(self.toggle_emb_visibility)
        else:
            group = self.model.get_curr_group()
            exp_to_embs = {}
            for exp_name, exp in group.items():
                exp_to_embs[exp_name] = self.model.get_filtered_embs(exp_name).keys()
            self.sidebar = FixedSidebar(exp_to_embs, self.render_trace)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.sidebar)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(220)
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

        next_emb_action = QAction(self)
        next_emb_action.setShortcut(QKeySequence("Ctrl+N"))
        next_emb_action.triggered.connect(partial(self.render_next_trace, forward=True))
        menu_bar.addAction(next_emb_action)

        prev_emb_action = QAction(self)
        prev_emb_action.setShortcut(QKeySequence("Ctrl+P"))
        prev_emb_action.triggered.connect(
            partial(self.render_next_trace, forward=False)
        )
        menu_bar.addAction(prev_emb_action)

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

    def save_peak_pos(self, peak_widths, peak_index):
        emb_name = self.model.curr_emb_name
        self.model.save_peak_widths(emb_name, peak_widths, peak_index)

    def add_peak(self, x, y):
        exp = self.model.get_curr_experiment()
        trace = self.model.get_curr_trace()
        emb_name = self.model.curr_emb_name

        if self.use_dev_time:
            dev_time = exp.embryos[emb_name].lin_developmental_time()
            idx = np.searchsorted(dev_time, x) - 1
            x = int(idx)
        else:
            x = int(x) * 10

        new_peak, new_peaks = self.model.add_peak(x, emb_name, trace)

        trace.to_add.append(new_peak)
        trace.peak_idxes = new_peaks

        self.render_trace()

    def remove_peak(self, x, y):
        exp = self.model.get_curr_experiment()
        trace = self.model.get_curr_trace()
        emb_name = self.model.curr_emb_name

        if self.use_dev_time:
            dev_time = exp.embryos[emb_name].lin_developmental_time()
            idx = np.searchsorted(dev_time, x) - 1
            x = int(idx)
        else:
            x = int(x) * 10

        removed_peaks, new_peaks = self.model.remove_peak(x, emb_name, trace)

        trace.to_remove.extend(removed_peaks)
        trace.peak_idxes = new_peaks

        self.render_trace()

    def repaint_curr_emb(self):
        """Repaints peaks for the trace currently being displayed.

        This funciton runs after any of the peak detection params sliders change."""

        pd_params = self.collect_slider_params()

        if pd_params is None:
            pd_params = self.model.config.get_pd_params()

        trace = self.model.get_curr_trace()
        trace.detect_peaks(pd_params["freq"])
        self.render_trace()

    def update_all_embs(self):
        """Calculates and paints again all peaks."""
        self.detect_peaks_all()
        self.render_trace()
        self.plot_all_traces()

    def collect_slider_params(self):
        # on 'combined exp' mode, the top_layout that hold the slider will be removed
        if not self.top_layout:
            return None

        freq = self.freq_slider.value()
        peak_width = self.rel_h_slider.value()

        return {"freq": freq, "peak_width": peak_width}

    def detect_peaks_all(self):
        """Recalculates peak indices for all embryos.

        Persists peak detection params in `peak_detection_params.json`."""
        for exp_name, exp in self.model.get_curr_group().items():
            pd_params = self.collect_slider_params()

            if pd_params is None:
                pd_params = self.model.config.get_pd_params()

            to_remove = list(self.model.to_remove[exp_name])

            for emb in exp.embryos.values():
                emb.trace.detect_peaks(pd_params["freq"])

            new_data = {"pd_params": pd_params, "exp_params": {"to_remove": to_remove}}
            self.model.update_config(new_data)

    def plot_all_traces(self):
        self.clear_layout(self.graph_layout)
        group = self.model.get_filtered_group()
        for exp_name, exp in group.items():
            for emb in exp.embryos.values():
                plot_widget = pg.PlotWidget()
                plot_widget.setMinimumHeight(200)

                trace = emb.trace
                if self.use_dev_time:
                    dev_time = emb.lin_developmental_time()
                    time = dev_time[: trace.trim_idx]
                else:
                    time = trace.time[: trace.trim_idx] / 60
                dff = trace.dff[: trace.trim_idx]

                peak_amps = trace.peak_amplitudes
                peak_times = time[trace.peak_idxes]

                if trace.to_add or trace.to_remove:
                    ttad = np.array([*trace.to_add, *trace.to_remove])
                    manual_times = time[ttad]
                    manual_amps = dff[ttad]
                    manual_scatter = pg.ScatterPlotItem(
                        manual_times, manual_amps, size=10, brush=QColor("cyan")
                    )
                    plot_widget.addItem(manual_scatter)

                scatter_plot_item = pg.ScatterPlotItem(
                    size=8,
                    brush=pg.mkColor("m"),
                )
                scatter_plot_item.setData(peak_times, peak_amps)

                plot_widget.addItem(scatter_plot_item)

                plot_widget.plot(time, dff)
                if self.model.has_combined_experiments():
                    plot_widget.setTitle(f"{exp_name} - {emb.name}")
                else:
                    plot_widget.setTitle(emb.name)
                self.graph_layout.addWidget(plot_widget)

    def calibrate_sliders(self):
        """Adjusts the sliders based on pd_params.json.

        The sliders should not be available when more than one experiment is loaded."""
        if self.model.has_combined_experiments():
            return

        pd_params = self.model.config.get_pd_params()

        for sld, name in (
            (self.freq_slider, "freq"),
            (self.rel_h_slider, "peak_width"),
        ):
            sld.setValue(pd_params[name])
            sld.set_custom_slot(self.repaint_curr_emb)
            sld.slider.sliderPressed.connect(self.started_dragging)
            sld.slider.sliderReleased.connect(self.stopped_dragging)

        # it's too costly to repaint all peak widths, so I wont update them as we drag
        # the width_slider
        self.rel_h_slider.setValue(pd_params["peak_width"])

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
        """Renders the currently selected trace.

        Omitted values will be replaced by currently selected values in `self.model`.
        For example, passing only an emb_name will select that emb from the current experiment.
        """
        if exp_name is None:
            exp = self.model.get_curr_experiment()
        else:
            exp = self.model.get_experiment(exp_name)

        if emb_name is None:
            emb_name = self.model.curr_emb_name
        else:
            self.model.curr_emb_name = emb_name

        self.plot_widget.clear()
        self.plot_widget.show()

        embryo = exp.embryos[emb_name]
        trace = embryo.trace
        if self.use_dev_time:
            dev_time = embryo.lin_developmental_time()
            time = dev_time[: trace.trim_idx]
        else:
            time = trace.time[: trace.trim_idx] / 60
        dff = trace.dff[: trace.trim_idx]

        peak_times = time[trace.peak_idxes]
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

        if self.display_filtered_dff and trace.filtered_dff is not None:
            self.plot_widget.plot(time, trace.filtered_dff, pen=pg.mkPen("palegreen"))

        if self.model.has_combined_experiments():
            self.plot_widget.setTitle(f"{exp.name} - {emb_name}")
        else:
            self.plot_widget.setTitle(emb_name)

        # paint peak widths
        rel_height = self.rel_h_slider.value()
        trace.compute_peak_bounds(rel_height)
        peak_bounds = trace.peak_bounds_indices.flatten()
        if peak_bounds.size == 0:
            return

        op = trace.detect_oscillations()
        op_amps = trace.dff[op]
        op_times = time[op]
        op_plot_items = pg.ScatterPlotItem(
            op_times, op_amps, size=8, brush=QColor("orange")
        )
        self.plot_widget.addItem(op_plot_items)

        if not self.show_peak_widths or self.is_dragging_slider:
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
        exp = self.model.get_curr_experiment()
        emb_name = self.model.curr_emb_name
        emb = exp.embryos[emb_name]

        row, col = divmod(il_obj.peak_index, 2)

        if self.use_dev_time:
            dev_time = emb.lin_developmental_time()
            idx = np.searchsorted(dev_time, il_obj.getXPos()) - 1
            x = int(idx)
        else:
            x = il_obj.getXPos() * 10

        trace.peak_bounds_indices[row, col] = x
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
        self.model.clear_manual_data()

        self.update_all_embs()
        self.show_notification("Clear manual data", "Manually added data was removed.")


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
