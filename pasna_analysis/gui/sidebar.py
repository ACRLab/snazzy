from collections.abc import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
)


def sort_by_emb_number(emb: str):
    if not emb.startswith("emb"):
        raise ValueError(f"Emb names must start with 'emb'. Received: {emb}")
    return int(emb[3:])


class FixedSidebar(QWidget):
    def __init__(
        self, exp_to_embs: dict[str, str], callback: Callable[[str, str | None], None]
    ):
        super().__init__()

        self.callback = callback

        main_layout = QVBoxLayout()

        self.populate_buttons(exp_to_embs, main_layout)
        main_layout.addStretch()
        self.setLayout(main_layout)

    def populate_buttons(self, exp_to_embs, layout):
        for exp in exp_to_embs:
            for emb in sorted(exp_to_embs[exp], key=sort_by_emb_number):
                row_layout = QHBoxLayout()
                row_layout.setSpacing(0)
                btn = QPushButton(f"{exp} - {emb}")
                btn.clicked.connect(
                    lambda checked, name=emb, exp=exp: self.callback(name, exp)
                )
                row_layout.addWidget(btn)
                layout.addLayout(row_layout)


class RemovableSidebar(QWidget):
    emb_visibility_toggled = pyqtSignal(str)

    def __init__(
        self,
        callback: Callable[[str, str | None], None],
        accepted_embs: set[str],
        removed_embs: set[str],
        pd_path: str,
    ):
        super().__init__()

        self.callback = callback
        self.pd_path = pd_path

        main_layout = QVBoxLayout()

        self.accepted_layout = QVBoxLayout()
        self.removed_layout = QVBoxLayout()

        main_layout.addLayout(self.accepted_layout)

        main_layout.addWidget(QLabel("Removed"))
        main_layout.addLayout(self.removed_layout)
        main_layout.addStretch()

        self.accepted_embs = accepted_embs
        self.removed_embs = removed_embs

        self.populate_buttons(self.accepted_embs, self.accepted_layout, True)
        self.populate_buttons(self.removed_embs, self.removed_layout, False)

        self.setLayout(main_layout)

    def populate_buttons(self, labels, layout, is_accepted):
        """Populate buttons into the given layout."""
        for label in sorted(labels, key=sort_by_emb_number):
            row_layout = QHBoxLayout()
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, name=label: self.callback(name, None))
            row_layout.addWidget(btn)

            btn2 = QPushButton("Remove" if is_accepted else "Accept")
            btn2.clicked.connect(
                lambda _, name=label, acc=is_accepted: self.toggle_button(name, acc)
            )
            row_layout.addWidget(btn2)

            layout.addLayout(row_layout)

    def toggle_button(self, label, is_accepted):
        """Toggle the button between accepted and removed categories."""
        if is_accepted:
            self.accepted_embs.remove(label)
            self.removed_embs.add(label)
        else:
            self.removed_embs.remove(label)
            self.accepted_embs.add(label)

        self.clear_layout(self.accepted_layout)
        self.clear_layout(self.removed_layout)
        self.populate_buttons(self.accepted_embs, self.accepted_layout, True)
        self.populate_buttons(self.removed_embs, self.removed_layout, False)

        self.emb_visibility_toggled.emit(label)

    def clear_layout(self, layout):
        """Clear all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())
