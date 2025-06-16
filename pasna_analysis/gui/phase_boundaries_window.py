import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget
import pyqtgraph as pg

from pasna_analysis import Trace, TracePhases


class PhaseBoundariesWindow(QWidget):
    save_bounds_signal = pyqtSignal(str, dict)

    def __init__(self, traces: list[Trace], current_trace: int, has_dsna: bool):
        super().__init__()

        self.traces = traces
        self.has_dsna = has_dsna
        self.current_trace = current_trace

        self.setWindowTitle("Phase Boundaries")

        next_trace_action = QAction(self)
        next_trace_action.setShortcut(QKeySequence("Ctrl+N"))
        next_trace_action.triggered.connect(self.next_trace)
        self.addAction(next_trace_action)

        layout = QVBoxLayout()

        self.top_btns = QHBoxLayout()

        self.save_changes_btn = QPushButton("Save changes")
        self.save_changes_btn.clicked.connect(self.save_changes)
        self.top_btns.addWidget(self.save_changes_btn)

        layout.addLayout(self.top_btns)

        self.plot_widget = pg.PlotWidget()

        layout.addWidget(self.plot_widget)

        self.setLayout(layout)

        self.phase2_line = None
        self.dsna_line = None
        self.paint_window()

    def paint_window(self):
        self.render_current_trace()
        self.phase2_start()
        if self.has_dsna:
            self.dsna_start()

    def next_trace(self):
        self.current_trace = (self.current_trace + 1) % len(self.traces)
        self.paint_window()

    def render_current_trace(self):
        self.plot_widget.clear()

        trace = self.traces[self.current_trace]
        self.plot_widget.plot(
            trace.time[: trace.trim_idx],
            trace.dff[: trace.trim_idx],
            name="Dff",
            pen=pg.mkPen("whitesmoke"),
        )
        self.plot_widget.setTitle(trace.name)

    def phase2_start(self):
        """Plot an InfiniteLine that splits phase 1 to phase 2.

        The line is plotted between peaks, so all peaks to the left belong to
        phase 1 and peaks to the right to phase 2."""
        if self.phase2_line is not None:
            self.plot_widget.removeItem(self.phase2_line)

        trace = self.traces[self.current_trace]
        trace_phases = TracePhases(trace)
        phase1_end = trace_phases.get_phase1_end()

        if phase1_end == trace.peak_idxes[-1]:
            break_line = phase1_end
        else:
            peak_idxes = trace.peak_idxes
            break_line = (peak_idxes[phase1_end] + peak_idxes[phase1_end + 1]) // 2

        self.phase2_line = pg.InfiniteLine(
            trace.time[break_line],
            movable=True,
            pen=pg.mkPen("tomato", cosmetic=True),
        )
        self.phase2_line.addMarker("<|>")
        self.phase2_line.sigPositionChangeFinished.connect(self.change_phase1_end)
        self.plot_widget.addItem(self.phase2_line)

    def find_last_peak_index(self, time: int, trace: Trace) -> int:
        """Find the last peak index that happended before `time`.

        Parameters:
            time(int):
                Time used to filter the indices. Has the same units as `trace.time`.
            trace(Trace):
                Trace object used to extract peak indices and time information.

        Return:
            int:
                Highest peak index that happend before `time`.
        """
        new_idx = -1
        for i, peak_time in enumerate(trace.peak_times):
            if peak_time > time:
                new_idx = i - 1
                break
        return new_idx

    def change_phase1_end(self, il_obj):
        previous_pos = il_obj.startPosition
        bound_time = int(il_obj.getXPos())

        curr_trace = self.traces[self.current_trace]

        new_idx = self.find_last_peak_index(bound_time, curr_trace)
        if new_idx == -1:
            self.phase2_line.setX(previous_pos.x())

    def change_dsna_start(self, il_obj):
        previous_pos = il_obj.startPosition
        bound_time = int(il_obj.getXPos())

        curr_trace = self.traces[self.current_trace]

        new_idx = self.find_last_peak_index(bound_time, curr_trace) + 1
        if new_idx == -1:
            self.dsna_line.setX(previous_pos.x())

    def dsna_start(self):
        if self.dsna_line is not None:
            self.plot_widget.removeItem(self.dsna_line)

        trace = self.traces[self.current_trace]
        trace_phases = TracePhases(trace)
        start = trace_phases.get_dsna_start()

        if start == trace.peak_idxes[-1]:
            break_line = start
        else:
            peak_idxes = trace.peak_idxes
            break_line = (peak_idxes[start] + peak_idxes[start - 1]) // 2

        self.dsna_line = pg.InfiniteLine(
            trace.time[break_line],
            movable=True,
            pen=pg.mkPen("fuchsia", cosmetic=True),
        )
        self.dsna_line.addMarker("<|>")
        self.dsna_line.sigPositionChangeFinished.connect(self.change_dsna_start)
        self.plot_widget.addItem(self.dsna_line)

    def get_index(self, x):
        trace = self.traces[self.current_trace]
        return np.searchsorted(trace.time, x, side="left")

    def save_changes(self):
        curr_trace = self.traces[self.current_trace]
        updated_bounds = {}
        if self.phase2_line is not None:
            phase1_end_pos = int(self.phase2_line.getXPos())
            new_idx = self.find_last_peak_index(phase1_end_pos, curr_trace)
            updated_bounds["phase1_end"] = new_idx
        if self.dsna_line is not None:
            dsna_start_pos = int(self.dsna_line.getXPos())
            new_idx = self.find_last_peak_index(dsna_start_pos, curr_trace) + 1
            updated_bounds["dsna_start"] = new_idx

        if updated_bounds:
            emb_name = curr_trace.name
            self.save_bounds_signal.emit(emb_name, updated_bounds)
