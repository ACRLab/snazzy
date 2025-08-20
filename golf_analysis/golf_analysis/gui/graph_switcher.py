from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSizePolicy


class GraphSwitcher(QWidget):

    def __init__(self, graph_widgets: list):
        super().__init__()

        self.graph_widgets = graph_widgets
        self.current_idx = 0

        self.main_layout = QHBoxLayout()
        self.graph_area = QVBoxLayout()

        for widget in self.graph_widgets:
            self.graph_area.addWidget(widget)
            widget.hide()

        if self.graph_widgets:
            self.graph_widgets[0].show()

        self.toggle_btn = QPushButton(">")
        self.toggle_btn.clicked.connect(self.toggle_graph)
        self.toggle_btn.setFixedWidth(20)
        self.toggle_btn.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        # Wrap graph area in a widget to apply stretch
        self.graph_container = QWidget()
        self.graph_container.setLayout(self.graph_area)
        self.graph_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.main_layout.addWidget(self.graph_container, stretch=1)
        self.main_layout.addWidget(self.toggle_btn, stretch=0)
        self.setLayout(self.main_layout)

    def toggle_graph(self):
        if not self.graph_widgets:
            return

        self.graph_widgets[self.current_idx].hide()

        self.current_idx = (self.current_idx + 1) % len(self.graph_widgets)

        self.graph_widgets[self.current_idx].show()
