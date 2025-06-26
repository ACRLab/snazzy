import sys
import traceback

from PyQt6.QtCore import pyqtSignal, QObject, QRunnable


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(object)
    result = pyqtSignal(object)


class Worker(QRunnable):
    finished = pyqtSignal()

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            value = sys.exc_info()[1]
            self.signals.error.emit(value)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
