from functools import partial
from pathlib import Path
import sys
import traceback
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtGui import QAction, QBrush, QColor, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pasna_analysis.gui import (
    ClickableViewBox,
    ComparePlotWindow,
    ExperimentParamsDialog,
    FixedSidebar,
    GraphSwitcher,
    ImageSequenceViewer,
    ImageWindow,
    JsonViewer,
    InteractivePlotWidget,
    LabeledSlider,
    Model,
    PlotWindow,
    PhaseBoundariesWindow,
    RemovableSidebar,
    Worker,
)

from pasna_analysis import Config


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = Model()
        self.moveable_width_bars = False
        self.view_peak_widths = False
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

    def _get_directory(self) -> Path | None:
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory:
            return
        return Path(directory)

    def _get_group_name(self, is_new_group: bool) -> str:
        return (
            f"group{len(self.model.groups) + 1}"
            if is_new_group
            else self.model.selected_group.name
        )

    def _show_experiment_dialog(
        self, config: Config, group_name: str, on_accepted: Callable[[dict], None]
    ):
        exp_params = config.get_exp_params()
        pd_params = config.get_pd_params()
        dff_strategy = pd_params.get("dff_strategy", "")

        dialog_params = {
            "group_name": group_name,
            **exp_params,
            "dff_strategy": dff_strategy,
        }
        self.exp_params_dialog = ExperimentParamsDialog(
            dialog_params, exp_path=config.rel_path, parent=self
        )

        self.exp_params_dialog.accepted.connect(
            lambda: on_accepted(self.exp_params_dialog.get_values())
        )
        self.exp_params_dialog.rejected.connect(lambda: None)

        self.exp_params_dialog.open()

    # TODO: it should be easier to update Config from the GUI
    def _update_config(self, config: Config, dialog_values):
        exp_params = config.get_exp_params()
        new_exp_params = {k: v for k, v in dialog_values.items() if k in exp_params}
        pd_params = {"dff_strategy": dialog_values["dff_strategy"]}
        new_config = {"exp_params": new_exp_params, "pd_params": pd_params}
        config.update_params(new_config)

        config.save_params()

    def _start_experiment_worker(self, config: Config, group_name: str):
        worker = Worker(
            self.model.create_experiment,
            config=config,
            group_name=group_name,
        )
        worker.signals.result.connect(self.update_UI)
        worker.signals.error.connect(self.handle_open_err)
        self.threadpool.start(worker)

    def _present_loading(self):
        try:
            self.placeholder.setText("Loading data..")
            self.placeholder.repaint()
        except RuntimeError:
            pass

    def _open_directory(self, is_new_group: bool, should_reset_model: bool = False):
        directory = self._get_directory()
        if not directory:
            return

        try:
            config = Config(directory)
            group_name = self._get_group_name(is_new_group)

            def on_dialog_accepted(dialog_values):
                group_name = dialog_values["group_name"]
                self._update_config(config, dialog_values)

                if should_reset_model:
                    self.model.set_initial_state()

                self._present_loading()

                self._start_experiment_worker(config, group_name)

            self._show_experiment_dialog(
                config, group_name, on_accepted=on_dialog_accepted
            )

        except Exception as e:
            traceback.print_exc()
            self.show_error_message(str(e))

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
        self.view_pd_action.setEnabled(True)
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
        group_name = self.group_combo_box.itemText(i)
        group = self.model.get_group_by_name(group_name)
        if group is None:
            return
        self.model.select_group(group)

        self.clear_layout(self.bottom_layout)
        self.paint_graphs()
        self.render_trace()
        self.plot_all_traces()

    def display_plots(self):
        curr_group = self.model.selected_group
        embs = list(emb for exp_name, emb in curr_group.iter_all_embryos())
        curr_trace = self.model.selected_trace
        self.pw = PlotWindow(embs, curr_group.name, curr_trace)
        self.pw.show()

    def display_compare_plots(self):
        groups = self.model.groups
        self.cpw = ComparePlotWindow(groups)
        self.cpw.show()

    def display_phase_boundaries(self):
        exp = self.model.selected_experiment
        traces = [e.trace for e in exp.embryos]

        current_trace = self.model.selected_trace
        current_trace_idx = 0
        for i, trace in enumerate(traces):
            if trace.name == current_trace:
                current_trace_idx = i
                break

        has_dsna = self.model.has_dsna()
        self.pbw = PhaseBoundariesWindow(traces, current_trace_idx, has_dsna)
        self.pbw.save_bounds_signal.connect(self.save_trace_phases)
        self.pbw.show()

    def save_trace_phases(self, emb_name, new_phases):
        if "phase1_end" in new_phases:
            self.model.save_phase1_end_idx(emb_name, new_phases["phase1_end"])
        if "dsna_start" in new_phases:
            self.model.save_dsna_start(emb_name, new_phases["dsna_start"])
        self.render_trace()

    def display_embryo_movie(self):
        exp = self.model.selected_experiment
        embryos = exp.all_embryos()
        try:
            self.viewer = ImageSequenceViewer(exp.directory, embryos)
        except FileNotFoundError as e:
            self.show_error_message(str(e))
            return
        self.viewer.show()

    def display_field_of_view(self):
        exp = self.model.selected_experiment
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
                select_group.insertItem(i, group.name)

            return select_group

        label_text = self.model.selected_group.name
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
            self.toggle_view_width_btn = QPushButton("View widths")
            self.toggle_view_width_btn.setCheckable(True)
            self.toggle_view_width_btn.clicked.connect(self.toggle_view_width)
            self.top_app_bar.addWidget(self.toggle_view_width_btn)

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
        self.render_trace()
        self.plot_all_traces()

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
        # to adjust widths `view_peak_widths` has to be True
        if check:
            self.toggle_view_width_btn.setChecked(True)
            self.view_peak_widths = True
        self.render_trace()

    def toggle_view_width(self, checked):
        self.view_peak_widths = checked
        # if peak widths are hidden widths cant be adjusted either
        if not checked:
            self.moveable_width_btn.setChecked(False)
            self.moveable_width_bars = False
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

    def toggle_emb_visibility(self, emb_name, should_remove):
        self.model.toggle_emb_visibility(emb_name, should_remove)
        if should_remove:
            self.render_trace()

    def paint_graphs(self):
        self.single_graph_frame = QFrame()
        self.bottom_layout.addWidget(self.single_graph_frame)
        single_graph_layout = QHBoxLayout()
        self.single_graph_frame.setLayout(single_graph_layout)

        # Sidebar start
        exp = self.model.selected_experiment
        if not self.model.has_combined_experiments():
            accepted_embs = set([e.name for e in exp.embryos])
            removed_embs = set(exp.to_remove)
            self.sidebar = RemovableSidebar(
                self.select_embryo, accepted_embs, removed_embs, exp.name
            )
            self.sidebar.emb_visibility_toggled.connect(self.toggle_emb_visibility)
        else:
            exp_to_embs = {}
            group = self.model.selected_group
            for exp_name, exp in group.experiments.items():
                exp_to_embs[exp_name] = [e.name for e in exp.embryos]
            self.sidebar = FixedSidebar(exp_to_embs, self.select_embryo)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.sidebar)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(220)
        single_graph_layout.addWidget(scroll_area)
        # Sidebar end

        # Graph start
        self.plot_widget = InteractivePlotWidget()
        self.plot_channels = pg.PlotWidget()
        self.graph_switcher = GraphSwitcher([self.plot_widget, self.plot_channels])
        single_graph_layout.addWidget(self.graph_switcher)

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

        self.view_pd_action = QAction("View pd_params data", self)
        self.view_pd_action.triggered.connect(self.display_json_data)
        self.view_pd_action.setEnabled(False)
        file_menu.addAction(self.view_pd_action)

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

        display_phase_bounds_action = QAction("View phase boundaries", self)
        display_phase_bounds_action.triggered.connect(self.display_phase_boundaries)
        plot_menu.addAction(display_phase_bounds_action)

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

    def display_json_data(self):
        config_data = self.model.get_config_data()
        if config_data is None:
            return
        self.json_window = JsonViewer(config_data)
        self.json_window.update_config_signal.connect(self.update_from_json_viewer)
        self.json_window.show()

    def update_from_json_viewer(self, new_data):
        self.model.update_config(new_data)
        # reset the current experiment to use the new config data
        self.model.reset_current_experiment()
        self.update_UI()

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
        emb_name = self.model.selected_embryo
        self.model.save_peak_widths(emb_name, peak_widths, peak_index)

    def add_peak(self, x, y):
        embryo = self.model.selected_embryo
        trace = embryo.trace
        emb_name = embryo.name

        if self.use_dev_time:
            dev_time = embryo.lin_developmental_time
            idx = np.searchsorted(dev_time, x) - 1
            x = int(idx)
        else:
            x = int(x) * 10

        new_peak, new_peaks = self.model.add_peak(x, emb_name, trace)

        trace.to_add.append(new_peak)
        trace.peak_idxes = new_peaks

        self.render_trace()

    def remove_peak(self, x, y):
        embryo = self.model.selected_embryo
        trace = embryo.trace
        emb_name = embryo.name

        if self.use_dev_time:
            dev_time = embryo.lin_developmental_time
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
            pd_params = self.model.get_pd_params()

        trace = self.model.selected_trace
        trace.detect_peaks(pd_params["freq"])
        trace.compute_peak_bounds(pd_params["peak_width"])
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
        pd_params = self.collect_slider_params()
        self.model.calc_peaks_all_embs(pd_params)

    def plot_all_traces(self):
        self.clear_layout(self.graph_layout)

        exps_and_embryos = list(self.model.selected_group.iter_all_embryos())
        for exp_name, emb in exps_and_embryos:
            callback = partial(self.select_embryo_from_multi_view, emb.name, exp_name)
            plot_widget = ClickableViewBox(callback)
            plot_widget.setMinimumHeight(200)

            trace = emb.trace
            if self.use_dev_time:
                dev_time = emb.lin_developmental_time
                time = dev_time[: trace.trim_idx]
            else:
                time = trace.time[: trace.trim_idx] / 60
            dff = trace.dff[: trace.trim_idx]

            plot_widget.plot(time, dff)

            if self.model.has_combined_experiments():
                plot_widget.setTitle(f"{exp_name} - {emb.name}")
            else:
                plot_widget.setTitle(emb.name)

            self.graph_layout.addWidget(plot_widget)

            if trace.to_add or trace.to_remove:
                ttad = np.array([*trace.to_add, *trace.to_remove])
                manual_times = time[ttad]
                manual_amps = dff[ttad]
                manual_scatter = pg.ScatterPlotItem(
                    manual_times, manual_amps, size=10, brush=QColor("cyan")
                )
                plot_widget.addItem(manual_scatter)

            if len(trace.peak_idxes) == 0:
                continue
            peak_times = time[trace.peak_idxes]

            scatter_plot_item = pg.ScatterPlotItem(
                size=8,
                brush=pg.mkColor("m"),
            )
            peak_amps = trace.peak_amplitudes
            scatter_plot_item.setData(peak_times, peak_amps)

            plot_widget.addItem(scatter_plot_item)

    def calibrate_sliders(self):
        """Adjusts the sliders based on pd_params.json.

        The sliders should not be available when more than one experiment is loaded."""
        if self.model.has_combined_experiments():
            return

        pd_params = self.model.get_pd_params()

        for sld, name in (
            (self.freq_slider, "freq"),
            (self.rel_h_slider, "peak_width"),
        ):
            sld.setValue(pd_params[name])
            sld.set_custom_slot(self.repaint_curr_emb)
            sld.slider.sliderPressed.connect(self.started_dragging)
            sld.slider.sliderReleased.connect(self.stopped_dragging)

        # it's too costly to repaint all peak widths, so they wont be
        # updated while we drag the width_slider
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

    def select_embryo_from_multi_view(self, emb_name, exp_name):
        self.toggle_graph_view(False)
        self.toggle_graph_btn.setChecked(False)
        self.select_embryo(emb_name, exp_name)

    def select_embryo(self, emb_name, exp_name):
        exp = self.model.selected_group.experiments[exp_name]
        emb = exp.get_embryo(emb_name)
        self.model.select_experiment(exp)
        self.model.select_embryo(emb)
        self.render_trace()

    def render_next_trace(self, forward: bool):
        self.model.move_to_next_emb(forward)
        self.render_trace()

    def render_trace(self):
        """Render data about the currently selected embryo."""
        trace, time, trimmed_time, dff = self.model.get_trace_context(self.use_dev_time)
        emb_name = self.model.selected_embryo.name
        exp_name = self.model.selected_experiment.name

        self._clear_current_plot()
        self._plot_raw_trace(trimmed_time, dff)
        self._plot_filtered_trace(trimmed_time, trace)
        self._plot_manual_annotations(trimmed_time, trace, dff)
        self._plot_peaks(trimmed_time, trace)
        self._plot_active_and_struct_channels(time, trace)
        self._setup_trim_line(time, trace)
        self._setup_dsna_line(trimmed_time, trace)
        self._plot_peak_widths(trimmed_time, trace)
        self._plot_detected_oscillations(trimmed_time, trace, dff)
        self._set_plot_titles(emb_name, exp_name)

    def _clear_current_plot(self):
        self.plot_widget.clear()
        self.plot_channels.clear()

    def _plot_raw_trace(self, time, dff):
        self.plot_widget.plot(time, dff)

    def _plot_filtered_trace(self, time, trace):
        if trace.filtered_dff is None:
            return
        if self.display_filtered_dff or self.is_dragging_slider:
            self.plot_widget.plot(time, trace.filtered_dff, pen=pg.mkPen("palegreen"))

    def _plot_manual_annotations(self, time, trace, dff):
        if not trace.to_add and not trace.to_remove:
            return
        indices = np.array([*trace.to_add, *trace.to_remove])
        manual_times = time[indices]
        manual_amps = dff[indices]
        scatter = pg.ScatterPlotItem(
            manual_times, manual_amps, size=10, brush=QColor("cyan")
        )
        self.plot_widget.addItem(scatter)

    def _plot_peaks(self, time, trace):
        if len(trace.peak_idxes) == 0:
            return

        peak_times = time[trace.peak_idxes]
        peak_amps = trace.peak_amplitudes

        brushes = [self.brushes[i % len(self.brushes)] for i in range(len(peak_times))]

        scatter = pg.ScatterPlotItem(
            x=peak_times,
            y=peak_amps,
            size=8,
            brush=brushes,
            pen=pg.mkPen(None),
        )
        self.plot_widget.addItem(scatter)

    def _plot_active_and_struct_channels(self, time, trace):
        self.plot_channels.plot(
            time, trace.active, name="Active", pen=pg.mkPen("limegreen")
        )
        self.plot_channels.plot(
            time, trace.struct, name="Structural", pen=pg.mkPen("firebrick")
        )
        self.plot_channels.addLegend()

    def _setup_dsna_line(self, time, trace):
        if not self.model.has_dsna():
            return

        try:
            freq = self.freq_slider.value()
        except RuntimeError:
            freq = trace.pd_params["freq"]

        dsna_start = trace.get_dsna_start(freq)

        dsna_line = pg.InfiniteLine(
            time[dsna_start],
            movable=True,
            pen=pg.mkPen("chartreuse", cosmetic=True),
        )
        dsna_line.addMarker("<|>")
        dsna_line.sigPositionChangeFinished.connect(self.change_dsna_start)
        self.plot_widget.addItem(dsna_line)

    def _setup_trim_line(self, time, trace):
        is_single_exp = not self.model.has_combined_experiments()

        trim_line = pg.InfiniteLine(
            time[trace.trim_idx],
            movable=is_single_exp,
            pen=pg.mkPen("darkorange", cosmetic=True),
        )

        if is_single_exp:
            trim_line.addMarker("<|>")

        trim_line.sigPositionChangeFinished.connect(self.change_trim_idx)
        self.plot_channels.addItem(trim_line)

    def _plot_peak_widths(self, time, trace):
        if self.is_dragging_slider or not self.view_peak_widths:
            return
        if trace.peak_bounds_indices.size == 0:
            return

        peak_bounds = trace.peak_bounds_indices.flatten()
        bound_times = time[peak_bounds]
        brushes = [self.brushes[i % len(self.brushes)] for i in range(len(bound_times))]

        for i, time_value in enumerate(bound_times):
            color = brushes[i // 2].color() if self.color_mode else "gold"
            line = pg.InfiniteLine(
                time_value, movable=self.moveable_width_bars, pen=pg.mkPen(color)
            )
            line.line_index = i
            if self.moveable_width_bars:
                line.addMarker("<|") if i % 2 == 0 else line.addMarker("|>")
            line.sigPositionChangeFinished.connect(self.change_peak_bounds)
            self.plot_widget.addItem(line)

    def _plot_detected_oscillations(self, time, trace, dff):
        op = trace.detect_oscillations()
        if op is None:
            print("No oscillations found")
            return
        op_times = time[op]
        op_amps = dff[op]
        scatter = pg.ScatterPlotItem(op_times, op_amps, size=8, brush=QColor("orange"))
        self.plot_widget.addItem(scatter)

    def _set_plot_titles(self, emb_name, exp_name):
        if self.model.has_combined_experiments():
            exp_name = exp_name or self.model.selected_experiment
            title = f"{exp_name} - {emb_name}"
        else:
            title = emb_name
        self.plot_widget.setTitle(title)
        self.plot_channels.setTitle(title)

    def change_dsna_start(self, il_obj):
        trace = self.model.selected_trace
        emb = self.model.selected_embryo

        if self.use_dev_time:
            dev_time = emb.lin_developmental_time
            idx = np.searchsorted(dev_time, il_obj.getXPos()) - 1
            x = int(idx)
        else:
            x = int(il_obj.getXPos() * 10)

        res = QMessageBox.question(
            self,
            "Confirm Update",
            "Update dSNA start?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )

        prev_dsna_start = trace.dsna_start
        if self.use_dev_time:
            prev_value = dev_time[prev_dsna_start]
        else:
            prev_value = prev_dsna_start / 10

        if res == QMessageBox.StandardButton.Cancel:
            il_obj.setValue(prev_value)
            return

        self.model.save_dsna_start(emb.name, x)

        pd_params = self.model.get_pd_params()

        trace.detect_peaks(pd_params["freq"])
        trace.compute_peak_bounds(pd_params["peak_width"])

        self.render_trace()

    def change_trim_idx(self, il_obj):
        trace = self.model.selected_trace
        emb = self.model.selected_embryo

        if self.use_dev_time:
            dev_time = emb.lin_developmental_time
            idx = np.searchsorted(dev_time, il_obj.getXPos()) - 1
            x = int(idx)
        else:
            x = int(il_obj.getXPos() * 10)

        res = QMessageBox.question(
            self,
            "Confirm Update",
            "Update trim index?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )

        if res == QMessageBox.StandardButton.Cancel:
            if self.use_dev_time:
                prev_value = dev_time[trace.trim_idx]
            else:
                prev_value = trace.trim_idx / 10
            il_obj.setValue(prev_value)
            return

        trace.trim_idx = x

        self.model.save_trim_idx(x)

        pd_params = self.model.get_pd_params()

        trace.detect_peaks(pd_params["freq"])
        trace.compute_peak_bounds(pd_params["peak_width"])

        self.render_trace()

    def change_peak_bounds(self, il_obj):
        emb = self.model.selected_embryo

        # since each peak has two bounds, we can recover the peak index
        # and bound index from the line number
        peak_index, bound_index = divmod(il_obj.line_index, 2)

        if self.use_dev_time:
            dev_time = emb.lin_developmental_time
            idx = np.searchsorted(dev_time, il_obj.getXPos()) - 1
            new_line_pos = int(idx)
        else:
            new_line_pos = int(il_obj.getXPos() * 10)

        self.model.update_peak_widths(peak_index, bound_index, new_line_pos)

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
