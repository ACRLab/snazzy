from pathlib import Path

import numpy as np
import tifffile
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from snazzy_analysis import Embryo


class ImageWindow(QWidget):
    def __init__(self, exp_name, image_path):
        super().__init__()
        self.setWindowTitle("All embryos")
        self.setGeometry(200, 200, 600, 400)

        title = QLabel(self)
        title.setText(f"Experiment: {exp_name}")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel(self)
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            raise FileNotFoundError(f"File not found:\n{image_path}")
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(label)
        self.setLayout(layout)


def normalize_16bit_to_8bit(img: np.ndarray, lower_p=0.25, upper_p=99.75) -> np.ndarray:
    """Convert a nparray from 16 bit to 8 bit.

    Outliers are clipped to match the upper and lower percentiles."""
    min_val = np.percentile(img, lower_p)
    max_val = np.percentile(img, upper_p)
    img = np.clip(img, min_val, max_val)
    normalized_img = (img - min_val) / (max_val - min_val) * 255
    return normalized_img.astype(np.uint8)


class ImageSequenceViewer(QWidget):
    def __init__(self, directory: Path, embryos: list[Embryo]):
        super().__init__()
        self.directory = directory

        embs_path = self.directory.joinpath("embs")
        if not embs_path.exists():
            raise FileNotFoundError(
                f"Embryo movies should be saved in {embs_path}. Could not find this directory."
            )

        self.dff_traces = {e.name: e.trace.dff for e in embryos}

        self.setWindowTitle("Image Sequence Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.images = None
        self.init_file_selector()

    def init_file_selector(self):
        self.selector_label = QLabel("Select a file:")
        self.combo_box = QComboBox()
        embs_path = self.directory.joinpath("embs")
        file_names = [str(f) for f in embs_path.iterdir() if "ch1.tif" in f.name]
        self.combo_box.addItems(file_names)
        self.open_button = QPushButton("Open Viewer")

        self.layout.addWidget(self.selector_label)
        self.layout.addWidget(self.combo_box)
        self.layout.addWidget(self.open_button)

        self.open_button.clicked.connect(self.load_file)

    def load_file(self):
        selected_file = self.combo_box.currentText()
        full_path = Path(selected_file)
        self.data = self.dff_traces[full_path.stem[:-4]]

        img = tifffile.imread(full_path)
        self.images = normalize_16bit_to_8bit(img)

        self.clear_file_selector()

        self.init_viewer()

    def clear_file_selector(self):
        """Clear file selector UI components."""
        self.selector_label.deleteLater()
        self.combo_box.deleteLater()
        self.open_button.deleteLater()

    def init_viewer(self):
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.images.shape[0] - 1)
        self.slider.valueChanged.connect(self.slider_changed)

        self.plot_widget = pg.PlotWidget()
        self.signal_plot = self.plot_widget.plot(self.data)
        self.indicator = pg.InfiniteLine(pos=0, angle=90, movable=False, pen="r")
        self.plot_widget.addItem(self.indicator)
        self.layout.addWidget(self.plot_widget)

        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.slider)
        self.layout.addLayout(control_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.current_frame = 0
        self.is_playing = False

        self.play_button.clicked.connect(self.start_playback)
        self.pause_button.clicked.connect(self.pause_playback)

        self.display_frame(0)

    def display_frame(self, frame_index):
        image = self.images[frame_index]

        height, width = image.shape
        q_image = QImage(
            image.data, width, height, width, QImage.Format.Format_Grayscale8
        )
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap)
        self.indicator.setPos(frame_index)
        self.slider.setValue(frame_index)

    def slider_changed(self, value):
        if not self.is_playing:
            self.current_frame = value
            self.display_frame(self.current_frame)

    def start_playback(self):
        if not self.timer.isActive():
            self.is_playing = True
            self.timer.start(10)

    def pause_playback(self):
        self.is_playing = False
        self.timer.stop()

    def next_frame(self):
        """Advance to next frame.

        After the last frame is reached, cycles back to the start."""
        self.current_frame = (self.current_frame + 1) % len(self.images)
        self.display_frame(self.current_frame)
