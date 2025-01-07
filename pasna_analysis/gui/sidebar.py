from collections.abc import Callable

from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (
    QWidget,
    QScrollArea,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
)


class RemovableSidebar(QWidget):
    def __init__(
        self,
        emb_names: list[str],
        callback: Callable[[str, str | None], None],
        pd_path: str,
    ):
        super().__init__()

        self.emb_names = emb_names
        self.callback = callback
        self.pd_path = pd_path

        # Main container and layouts
        # container = QWidget()
        main_layout = QVBoxLayout()

        # Create separate layouts for accepted and removed buttons
        self.accepted_layout = QVBoxLayout()
        self.removed_layout = QVBoxLayout()

        # Section labels
        main_layout.addWidget(QLabel("Accepted"), 0, Qt.AlignmentFlag.AlignTop)
        main_layout.addLayout(self.accepted_layout)

        main_layout.addWidget(QLabel("Removed"), 0, Qt.AlignmentFlag.AlignBottom)
        main_layout.addLayout(self.removed_layout)

        # Initialize buttons in the accepted category
        self.accepted_buttons = emb_names
        self.removed_buttons = []

        # Fill the accepted layout initially
        self.populate_buttons(self.accepted_buttons, self.accepted_layout, True)

        # Set up container
        self.setLayout(main_layout)
        # self.setWidget(container)
        # self.setWidgetResizable(True)

    def populate_buttons(self, labels, layout, is_accepted):
        """Populate buttons into the given layout."""
        for label in labels:
            row_layout = QHBoxLayout()
            row_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            # Create the first button (display label)
            btn = QPushButton(label)
            # btn.setEnabled(False)  # Display-only button
            btn.clicked.connect(lambda checked, name=label: self.callback(name, None))
            row_layout.addWidget(btn)

            # Create the second button (toggle button)
            btn2 = QPushButton("Remove" if is_accepted else "Accept")
            btn2.clicked.connect(
                lambda _, name=label, acc=is_accepted: self.toggle_button(name, acc)
            )
            row_layout.addWidget(btn2)

            layout.addLayout(row_layout)

    def toggle_button(self, label, is_accepted):
        """Toggle the button between accepted and removed categories."""
        if is_accepted:
            # Move from accepted to removed
            self.accepted_buttons.remove(label)
            self.removed_buttons.append(label)
        else:
            # Move from removed to accepted
            self.removed_buttons.remove(label)
            self.accepted_buttons.append(label)

        # Refresh layouts
        self.clear_layout(self.accepted_layout)
        self.clear_layout(self.removed_layout)
        self.populate_buttons(self.accepted_buttons, self.accepted_layout, True)
        self.populate_buttons(self.removed_buttons, self.removed_layout, False)

    def clear_layout(self, layout):
        """Clear all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())
