from copy import deepcopy
import traceback
import sys

from PyQt6.QtCore import pyqtSignal, QObject, QRunnable

from pasna_analysis import Experiment, Trace
from pasna_analysis.gui.peak_finder import PeakFinder


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


class Model:
    def __init__(self):
        self.pf = PeakFinder()
        self.set_initial_state()

    def set_initial_state(self):
        self.groups: dict[str, dict[str, Experiment]] = {}
        self.curr_group = None
        self.to_remove: dict[str, set] = {}
        self.curr_exp = None
        self.curr_emb_name = None

    def create_experiment(self, directory, group_name):
        exp = Experiment(
            directory,
            first_peak_threshold=0,
            to_exclude=[],
            dff_strategy="local_minima",
        )
        self.add_experiment(exp, group_name)
        pd_params = self.pf.get_pd_params(exp.pd_params_path)
        self.to_remove[exp.name] = set(pd_params.get("to_remove", []))
        return exp

    def add_experiment(self, experiment: Experiment, group: str):
        if group is None:
            group = self.curr_group

        if experiment in self.groups[group]:
            raise ValueError("Experiment already added to this group.")

        self.groups[group][experiment.name] = experiment
        self.to_remove[experiment.name] = set()

        if self.curr_exp is None:
            self.curr_exp = experiment.name

        if self.curr_emb_name is None:
            emb_name = next(iter(experiment.embryos))
            self.curr_emb_name = emb_name

    def get_filtered_groups(self):
        groups = deepcopy(self.groups)
        for group_name, group in groups.items():
            for exp_name, exp in group.items():
                exp.embryos = self.get_filtered_embs(exp_name, group_name)
        return groups

    def get_filtered_embs(self, exp_name, group_name=None):
        exp = self.get_experiment(exp_name, group_name)
        if exp_name not in self.to_remove:
            return exp.embryos
        return {
            emb_name: emb
            for emb_name, emb in exp.embryos.items()
            if emb_name not in self.to_remove[exp_name]
        }

    def get_filtered_group(self):
        group = deepcopy(self.groups[self.curr_group])
        for exp_name, exp in group.items():
            exp.embryos = self.get_filtered_embs(exp_name)
        return group

    def set_curr_group(self, group=str):
        if group not in self.groups:
            raise ValueError("Group not found.")
        self.curr_group = group

        self.curr_exp = next(iter(self.groups[group]))
        curr_exp = self.get_curr_experiment()

        self.curr_emb_name = next(iter(curr_exp.embryos))

    def get_curr_experiment(self) -> Experiment:
        return self.groups[self.curr_group][self.curr_exp]

    def get_experiment(self, exp_name, group_name=None) -> Experiment:
        if group_name is None:
            curr_group = self.get_curr_group()
        else:
            curr_group = self.groups[group_name]
        return curr_group[exp_name]

    def get_curr_group(self) -> dict[str, Experiment]:
        return self.groups[self.curr_group]

    def get_curr_trace(self) -> Trace:
        exp = self.get_curr_experiment()
        return exp.embryos[self.curr_emb_name].trace

    def add_group(self, group):
        self.groups[group] = {}
        if self.curr_group is None:
            self.curr_group = group

    def has_combined_experiments(self):
        return len(self.get_curr_group()) > 1
