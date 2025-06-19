import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor


class CustomViewBox(pg.ViewBox):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def mouseClickEvent(self, ev):
        if ev.button() == pg.QtCore.Qt.MouseButton.LeftButton:
            self.callback()
            ev.accept()


class ClickableViewBox(pg.PlotWidget):
    def __init__(self, callback, *args, **kwargs):
        # PlotWidget cannot receive mouse clicks, we need a ViewBox for that
        self.custom_viewbox = CustomViewBox(callback)
        super().__init__(*args, viewBox=self.custom_viewbox, **kwargs)
        self.setStyleSheet("border: 2px solid #000000;")

    def mouseMoveEvent(self, ev):
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet("border: 2px solid goldenrod;")

    def leaveEvent(self, ev):
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.setStyleSheet(("border: 2px solid #000000;"))
