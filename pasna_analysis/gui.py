from pathlib import Path
import sys

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
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, pyqtSignal, QPointF

import pyqtgraph as pg

from pasna_analysis import Experiment, DataLoader


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
        self, name, min_value, max_value, initial_value, step_size=None, parent=None
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

        self.mpd_slider = LabeledSlider("Minimum peak distance", 40, 600, 70)
        self.order_zero_slider = LabeledSlider("Order 0 min", 0, 0.5, 0.2, 0.01)
        self.order_one_slider = LabeledSlider("Order 1 min", 0, 0.1, 0.05, 0.002)
        self.prominence_slider = LabeledSlider("Prominence", 0, 2, 0.4, 0.01)

        self.top_layout.addWidget(self.mpd_slider)
        self.top_layout.addWidget(self.order_zero_slider)
        self.top_layout.addWidget(self.order_one_slider)
        self.top_layout.addWidget(self.prominence_slider)

        self.button = QPushButton("Apply Changes")
        self.button.clicked.connect(self.handle_button_click)
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

    def handle_button_click(self):
        # TODO: call detect_peaks to update plots
        slider_values = [
            self.slider1.value(),
            self.slider2.value(),
            self.slider3.value(),
            self.slider4.value(),
        ]
        print(slider_values)
        print("Should udpate peak detection params")

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
        )
        directory = Path(directory)
        if directory:
            self.exp = Experiment(
                DataLoader(directory),
                first_peak_threshold=0,
                to_exclude=[],
                dff_strategy="local_minima",
            )

            self.add_sidebar_buttons([emb.name for emb in self.exp.embryos])

            self.render_trace(self.exp.embryos[0].name)

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
