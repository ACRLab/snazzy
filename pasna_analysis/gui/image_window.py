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


class ImageWindow(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("All embryos")
        self.setGeometry(200, 200, 600, 400)

        label = QLabel(self)
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)


class ImageSequenceViewer(QWidget):
    def __init__(self, directory, dff_traces):
        super().__init__()
        self.directory = directory
        self.dff_traces = dff_traces

        self.setWindowTitle("Image Sequence Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.images = None
        self.init_file_selector()

    def init_file_selector(self):
        """Initialize the file selector UI."""
        self.selector_label = QLabel("Select a file:")
        self.combo_box = QComboBox()
        embs_path = self.directory / "embs"
        file_options = [str(f) for f in embs_path.iterdir() if "ch1.tif" in f.name]
        self.combo_box.addItems(file_options)
        self.open_button = QPushButton("Open Viewer")

        self.layout.addWidget(self.selector_label)
        self.layout.addWidget(self.combo_box)
        self.layout.addWidget(self.open_button)

        self.open_button.clicked.connect(self.load_file)

    def load_file(self):
        """Load selected file and initialize viewer."""
        selected_file = self.combo_box.currentText()
        full_path = Path(selected_file)
        self.data = self.dff_traces[full_path.stem[:-4]]

        if not full_path.exists():
            error_msg = QLabel(f"Error: File not found - {full_path}")
            error_msg.setStyleSheet("color: red;")
            self.layout.addWidget(error_msg)
            return

        img = tifffile.imread(full_path)
        normalized_img = (img - img.min()) / (img.max() - img.min()) * 255
        self.images = normalized_img.astype(np.uint8)

        self.clear_file_selector()

        self.init_viewer()

    def clear_file_selector(self):
        """Clear file selector UI components."""
        self.selector_label.deleteLater()
        self.combo_box.deleteLater()
        self.open_button.deleteLater()

    def init_viewer(self):
        """Initialize the full image sequence viewer UI."""
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
        """Display the specified frame."""
        image = self.images[frame_index]

        height, width = image.shape
        q_image = QImage(image.data, width, height, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap)
        self.indicator.setPos(frame_index)
        self.slider.setValue(frame_index)

    def slider_changed(self, value):
        """Handle slider changes."""
        if not self.is_playing:
            self.current_frame = value
            self.display_frame(self.current_frame)

    def start_playback(self):
        """Start playing the image sequence."""
        if not self.timer.isActive():
            self.is_playing = True
            self.timer.start(10)

    def pause_playback(self):
        """Pause playback."""
        self.is_playing = False
        self.timer.stop()

    def next_frame(self):
        """Advance to the next frame."""
        self.current_frame = (self.current_frame + 1) % len(self.images)
        self.display_frame(self.current_frame)
