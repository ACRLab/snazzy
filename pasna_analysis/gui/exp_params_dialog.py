from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from pasna_analysis import utils


def convert_value(value: str, target_type: type | str):
    """Convert string input to the specified type, or raise ValueError.

    If target_type is a list, will parse value as a list of integers."""
    if target_type == bool:
        val = value.strip().lower()
        if val not in ["true", "false"]:
            raise ValueError(f"Expected a boolean value, got: {val}")
        return True if val == "true" else False
    elif target_type == int:
        return int(value)
    elif target_type == float:
        return float(value)
    elif target_type == Path:
        return utils.convert_to_relative_path(Path(value), "data")
    elif target_type == "emb_list":
        return [f"emb{x.strip()}" for x in value.strip("[]").split(",") if x.strip()]
    else:
        return value


class ExperimentParamsDialog(QDialog):
    def __init__(self, properties, types, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Experiment parameters")
        self.setFixedWidth(480)
        self.properties = properties
        self.types = types
        self.inputs = {}

        layout = QVBoxLayout()
        for key, value in properties.items():
            row = QHBoxLayout()
            label = QLabel(key)
            input_field = QLineEdit(str(value))
            self.inputs[key] = input_field
            row.addWidget(label)
            row.addWidget(input_field)
            layout.addLayout(row)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def showEvent(self, event):
        super().showEvent(event)
        if self.parent():
            parent_geometry = self.parent().frameGeometry()
            dialog_geometry = self.frameGeometry()
            dialog_geometry.moveCenter(parent_geometry.center())
            self.move(dialog_geometry.topLeft())

    def get_values(self):
        result = {}
        for k, field in self.inputs.items():
            raw = field.text().strip()
            expected_type = self.types.get(k, str)
            result[k] = convert_value(raw, expected_type)
        return result
