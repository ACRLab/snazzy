from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHeaderView,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

KEY_TYPES = {
    "exp_path": str,
    "exp_params": dict,
    "first_peak_threshold": int,
    "to_exclude": list,
    "to_remove": list,
    "has_transients": bool,
    "pd_params": dict,
    "freq": float,
    "trim_zscore": float,
    "dff_strategy": str,
    "peak_width": float,
    "manual_peaks": list,
    "manual_remove": list,
    "manual_widths": dict,
    "wlen": int,
    "embryos": dict,
}


class JsonViewer(QWidget):

    update_config_signal = pyqtSignal(object)

    def __init__(self, json_data):
        super().__init__()
        self.setWindowTitle("Configuration parameters")
        self.resize(600, 400)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Key", "Value"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )

        self.populate_tree(self.tree, json_data)

        self.save_btn = QPushButton("Save changes")
        self.save_btn.clicked.connect(self.collect_data)

        layout = QVBoxLayout()
        layout.addWidget(self.tree)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

    def populate_tree(self, tree_widget, json_data):
        for k, v in json_data.items():
            item = QTreeWidgetItem([k, ""])
            tree_widget.addTopLevelItem(item)
            editable = k in ("exp_params", "pd_params")
            self.add_children(item, v, editable=editable)

    def add_children(self, parent, value, editable=False):
        if isinstance(value, dict):
            for k, v in value.items():
                child = QTreeWidgetItem(
                    [str(k), str(v) if not isinstance(v, (dict, list)) else ""]
                )
                if editable and not isinstance(v, (dict, list)):
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsEditable)
                parent.addChild(child)
                self.add_children(child, v, editable)
        elif isinstance(value, list):
            for i, elem in enumerate(value):
                child = QTreeWidgetItem(
                    [f"[{i}]", str(elem) if not isinstance(elem, (dict, list)) else ""]
                )
                parent.addChild(child)
                self.add_children(child, elem, editable=False)
        else:
            parent.setText(1, str(value))
            if editable:
                parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsEditable)

    def collect_data(self):
        collected = {}
        for i in range(self.tree.topLevelItemCount()):
            top_item = self.tree.topLevelItem(i)
            key = top_item.text(0)
            collected[key] = self.read_item(top_item)
        self.update_config_signal.emit(collected)

    def read_item(self, item):
        if item.childCount() == 0:
            return self.parse_value(item.text(0), item.text(1))

        obj = {}
        for i in range(item.childCount()):
            child = item.child(i)
            key = child.text(0)
            obj[key] = self.read_item(child)

        # convert to list if keys are indices:
        if all(k.startswith("[") for k in obj.keys()):
            return [
                v
                for k, v in sorted(obj.items(), key=lambda x: int(x[0].strip("[]")))
                if v
            ]

        return obj

    def parse_value(self, key, value):
        # all lists in Config at the moment are list[int]
        if key.startswith("["):
            return int(value)
        if key not in KEY_TYPES:
            raise ValueError(f"Got a key that is not part of config params: {key}.")
        try:
            return KEY_TYPES[key](value)
        except ValueError:
            return value
