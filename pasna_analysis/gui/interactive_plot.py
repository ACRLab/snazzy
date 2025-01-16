from PyQt6.QtCore import pyqtSignal, QPointF, Qt
from PyQt6.QtWidgets import QApplication
import pyqtgraph as pg


class InteractivePlotWidget(pg.PlotWidget):
    add_peak_fired = pyqtSignal(float, float)
    remove_peak_fired = pyqtSignal(float, float)

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

            modifier = QApplication.keyboardModifiers()
            if modifier == Qt.KeyboardModifier.ShiftModifier:
                self.add_peak_fired.emit(x, y)
            elif modifier == Qt.KeyboardModifier.ControlModifier:
                self.remove_peak_fired.emit(x, y)

    def get_items(self):
        """Returns the list of `QGraphicsItem` associated to this PlotWidget."""
        return super().items()
