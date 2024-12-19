from pathlib import Path
import sys

from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QLabel,
    QApplication,
    QMainWindow,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QSlider,
)
import pyqtgraph as pg

from pasna_analysis import DataLoader, Experiment
from pasna_analysis.interactive_find_peaks import (
    get_initial_values,
    save_detection_params,
)


class InteractivePlotWidget(pg.PlotWidget):
    mouse_clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMouseTracking(True)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if ev.button() == Qt.MouseButton.LeftButton:
            plot_item = self.getPlotItem()
            vb = plot_item.vb
            mouse_point = vb.mapSceneToView(QPointF(ev.pos()))
            x, y = mouse_point.x(), mouse_point.y()

            modifiers = QApplication.keyboardModifiers()
            if modifiers and Qt.KeyboardModifier.ShiftModifier:
                print("SHIFT held down")
            print(x, y)
            self.mouse_clicked.emit(x, y)


class FloatSlider(QSlider):
    def __init__(self, min_value, max_value, initial_value, step_size=0.1, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)

        # Store the original min and max values as floating-point numbers
        self._min_value = min_value
        self._max_value = max_value
        self._step_size = step_size

        # Convert the float range to integers for the slider
        self.setRange(int(min_value / step_size), int(max_value / step_size))
        self.setSingleStep(
            int(step_size / step_size)
        )  # Ensure each tick represents the step size

        self.setValue(initial_value)

    def setValue(self, value):
        """Set the value as a float."""
        # Scale the value to fit the integer range of the slider
        value_int = int((value - self._min_value) / self._step_size)
        super().setValue(value_int)

    def value(self):
        """Get the value as a float."""
        # Convert the integer value back to the float scale
        return self._min_value + super().value() * self._step_size

    def setRange(self, min_value, max_value):
        """Set the range of the slider as floats."""
        self._min_value = min_value
        self._max_value = max_value
        super().setRange(
            int(min_value / self._step_size), int(max_value / self._step_size)
        )


class LabeledSlider(QWidget):
    def __init__(
        self,
        name,
        min_value,
        max_value,
        initial_value,
        step_size=None,
        custom_slot=None,
        parent=None,
    ):
        super().__init__(parent)

        # Layout for the slider, label, and value
        self.layout = QVBoxLayout()

        if step_size is None:
            # Create and set up the slider
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(min_value, max_value)
            self.slider.setValue(initial_value)
            # self.slider.valueChanged.connect(self.update_value_label)
        else:
            self.slider = FloatSlider(min_value, max_value, step_size, initial_value)
        self.slider.valueChanged.connect(self.update_value_label)

        if custom_slot:
            self.slider.valueChanged.connect(custom_slot)

        # Name label
        self.name_label = QLabel(name)

        # Value label to show current value of slider
        self.value_label = QLabel(str(initial_value))

        # Add widgets to layout
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.value_label)

        # Set the layout of this widget
        self.setLayout(self.layout)

    def update_value_label(self, value):
        # Update the value label when the slider's value changes
        if type(self.slider) == FloatSlider:
            self.value_label.setText(f"{self.slider.value():.2f}")
        else:
            self.value_label.setText(str(value))

    def value(self):
        return self.slider.value()

    def setValue(self, value):
        return self.slider.setValue(value)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pasna Analysis")
        self.setGeometry(100, 100, 1200, 600)

        # Menu start
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open Directory", self)
        open_action.triggered.connect(self.open_directory)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        # Menu end

        # Main layout start
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        # Main layout end

        # Top layout start (sliders)
        self.top_layout = QHBoxLayout()
        self.layout.addLayout(self.top_layout)

        # For the first paint, we have no access to the last used initial values
        # so we just initialize them with defaults:
        # TODO: extract the default values to somewhere else
        self.mpd_slider = LabeledSlider(
            "Minimum peak distance", 40, 600, 70, custom_slot=self.repaint_curr_emb
        )
        self.order_zero_slider = LabeledSlider(
            "Order 0 min", 0, 0.5, 0.06, 0.005, custom_slot=self.repaint_curr_emb
        )
        self.order_one_slider = LabeledSlider(
            "Order 1 min", 0, 0.1, 0.006, 0.0005, custom_slot=self.repaint_curr_emb
        )
        self.prominence_slider = LabeledSlider(
            "Prominence", 0, 1, 0.06, 0.005, custom_slot=self.repaint_curr_emb
        )

        self.top_layout.addWidget(self.mpd_slider)
        self.top_layout.addWidget(self.order_zero_slider)
        self.top_layout.addWidget(self.order_one_slider)
        self.top_layout.addWidget(self.prominence_slider)

        self.button = QPushButton("Apply Changes")
        self.button.clicked.connect(self.detect_peaks_all)
        self.top_layout.addWidget(self.button)
        # Top layout end (sliders)

        # Bottom layout start: sidebar and graph container
        self.bottom_layout = QHBoxLayout()
        self.layout.addLayout(self.bottom_layout)
        # Bottom layout end

        # Sidebar start
        self.sidebar = QWidget()
        self.sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(self.sidebar_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.sidebar)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(150)
        self.bottom_layout.addWidget(self.scroll_area)
        # Sidebar end

        # Graph start
        self.plot_widget = InteractivePlotWidget()
        self.bottom_layout.addWidget(self.plot_widget)
        self.plot_widget.hide()

        self.scatter_plot_item = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkColor("m"),
        )
        self.plot_widget.addItem(self.scatter_plot_item)
        # Graph end

    def repaint_curr_emb(self):
        """Repaints peaks for the trace currently being displayed.

        This funciton runs after any of the peak detection params sliders change."""
        order_zero_min = self.order_zero_slider.value()
        order_one_min = self.order_one_slider.value()
        mpd = self.mpd_slider.value()
        prominence = self.prominence_slider.value()

        curr_trace = self.exp.traces[self.curr_emb_name]
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

        for trace in self.exp.traces.values():
            trace.detect_peaks(
                mpd,
                order_zero_min,
                order_one_min,
                prominence,
            )

        pd_params_path = self.directory / "peak_detection_params.json"
        save_detection_params(
            pd_params_path,
            mpd,
            order_zero_min,
            order_one_min,
            prominence,
        )

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
        )
        self.directory = Path(directory)
        try:
            self.exp = Experiment(
                DataLoader(self.directory),
                first_peak_threshold=0,
                to_exclude=[],
                dff_strategy="local_minima",
            )

            self.curr_emb_name = self.exp.embryos[0].name

            self.add_sidebar_buttons([emb.name for emb in self.exp.embryos])
            self.calibrate_sliders()
            self.render_trace(self.curr_emb_name)
        except FileNotFoundError:
            print(f"Could not read data from {self.directory}")

    def calibrate_sliders(self):
        initial_values = get_initial_values(
            self.directory / "peak_detection_params.json"
        )
        self.mpd_slider.setValue(initial_values["mpd"])
        self.order_zero_slider.setValue(initial_values["order0_min"])
        self.order_one_slider.setValue(initial_values["order1_min"])
        self.prominence_slider.setValue(initial_values["prominence"])

    def add_sidebar_buttons(self, emb_names):
        for emb_name in emb_names:
            btn = QPushButton(emb_name)
            btn.clicked.connect(lambda checked, name=emb_name: self.render_trace(name))
            self.sidebar_layout.addWidget(btn)

        # Add spacer at the bottom
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.sidebar_layout.addWidget(spacer)

    def render_trace(self, emb_name):
        self.curr_emb_name = emb_name
        self.plot_widget.clear()
        self.plot_widget.show()

        trace = self.exp.traces[emb_name]
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
