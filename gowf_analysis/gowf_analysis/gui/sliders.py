from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class FloatSlider(QSlider):
    def __init__(self, min_value, max_value, initial_value, step_size=0.1):
        super().__init__(Qt.Orientation.Horizontal)

        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size

        self.setRange(int(min_value / step_size), int(max_value / step_size))
        self.setSingleStep(int(step_size / step_size))

        self.setValue(initial_value / step_size)

    def setValue(self, value):
        value_int = int(value / self.step_size)
        super().setValue(value_int)

    def value(self):
        return super().value() * self.step_size

    def setRange(self, min_value, max_value):
        super().setRange(min_value, max_value)


class LabeledSlider(QWidget):
    def __init__(
        self,
        name,
        min_value,
        max_value,
        initial_value,
        step_size=None,
        custom_slot=None,
    ):
        super().__init__()

        self.layout = QVBoxLayout()

        if step_size is None:
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(min_value, max_value)
            self.slider.setValue(initial_value)
        else:
            self.slider = FloatSlider(min_value, max_value, initial_value, step_size)
        self.slider.valueChanged.connect(self.update_value_label)

        if custom_slot:
            self.slider.valueChanged.connect(custom_slot)

        self.name_label = QLabel(name)
        self.value_label = QLabel(str(initial_value))

        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.value_label)

        self.setLayout(self.layout)

    def update_value_label(self, value):
        if type(self.slider) == FloatSlider:
            self.value_label.setText(f"{self.slider.value():.4f}")
        else:
            self.value_label.setText(str(value))

    def value(self):
        return self.slider.value()

    def setValue(self, value):
        self.slider.setValue(value)

    def set_custom_slot(self, slot):
        self.slider.valueChanged.connect(slot)
