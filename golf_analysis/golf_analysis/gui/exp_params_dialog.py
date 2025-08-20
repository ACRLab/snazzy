from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from golf_analysis import utils


def convert_value(value: str, field_name: str):
    """Convert fields for the corresponding type based on field name."""
    if field_name == "first_peak_threshold":
        return int(value)
    elif field_name == "to_exclude" or field_name == "to_remove":
        return [f"emb{x.strip()}" for x in value.strip("[]").split(",") if x.strip()]
    else:
        return value


class ExperimentParamsDialog(QDialog):
    """Present Experiment params that can be changed before creating an Experiment.

    Embryos are presented as embryo ids, to make the input easier to change.
    Internally, pasna_analysis uses embryo names, so when data is coming in / going
    out it has to alternate between emb_names and emb_ids.
    """

    def __init__(self, properties, exp_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Experiment parameters")
        self.setFixedWidth(540)

        self.combo_keys = {"dff_strategy": ["baseline", "local_minima"]}
        self.inputs = {}
        self.adjust_emb_names(properties)

        layout = QVBoxLayout()
        layout.addLayout(self._create_exp_path_row(exp_path))
        for row_layout in self._create_property_rows(properties):
            layout.addLayout(row_layout)
        layout.addLayout(self._create_button_row())
        self.setLayout(layout)

    def _create_exp_path_row(self, exp_path):
        layout = QHBoxLayout()
        label = QLabel(f"Experiment path: {exp_path}")
        layout.addWidget(label)
        return layout

    def _create_property_rows(self, properties):
        layouts = []
        for key, value in properties.items():
            row = QHBoxLayout()
            label = QLabel(key)
            input_field = self._create_input_field(key, value)
            self.inputs[key] = input_field
            row.addWidget(label)
            row.addWidget(input_field)
            layouts.append(row)
        return layouts

    def _create_input_field(self, key, value):
        if isinstance(value, bool):
            input_field = QCheckBox()
            input_field.setChecked(value)
        elif key in self.combo_keys:
            input_field = QComboBox()
            input_field.addItems(self.combo_keys[key])
            if value in self.combo_keys[key]:
                input_field.setCurrentText(value)
        else:
            input_field = QLineEdit(str(value))
        return input_field

    def _create_button_row(self):
        layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(ok_button)
        layout.addWidget(cancel_button)
        return layout

    def adjust_emb_names(self, properties):
        """Convert the embryo names to embryo ids, so its easier to add / remove embryos."""
        for key in ("to_remove", "to_exclude"):
            if key in properties:
                if properties[key]:
                    # keeping this check for compatibility:
                    # previous versions of pd_params used to save ids instead of emb_names
                    if properties[key][0].isdigit():
                        properties[key] = sorted(
                            [int(emb_id) for emb_id in properties[key]]
                        )
                    else:
                        properties[key] = sorted(
                            [utils.emb_id(emb_name) for emb_name in properties[key]]
                        )

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
            if isinstance(field, QCheckBox):
                result[k] = field.isChecked()
            elif isinstance(field, QComboBox):
                result[k] = field.currentText()
            else:
                raw = field.text().strip()
                result[k] = convert_value(raw, k)
        return result
