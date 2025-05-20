from copy import deepcopy
import sys
import traceback

from PyQt6.QtCore import pyqtSignal, QObject, QRunnable

from pasna_analysis import Config, Embryo, Experiment, Trace, utils
from pasna_analysis.gui import PeakMatcher


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
        self.pm = PeakMatcher()
        self.config: Config | None = None
        self.set_initial_state()

    def __str__(self):
        group_names = list(self.groups.keys())
        to_remove_count = {k: len(v) for k, v in self.to_remove.items()}
        return (
            f"Model(\n"
            f"  groups: {group_names} groups\n"
            f"  curr_group: {self.curr_group}\n"
            f"  to_remove: {to_remove_count}\n"
            f"  curr_exp: {self.curr_exp}\n"
            f"  curr_emb_name: {self.curr_emb_name}\n"
            f")"
        )

    def update_config(self, new_data):
        """Updates the config data for the current experiment."""
        exp = self.get_curr_experiment()
        exp.config.update_params(new_data)
        exp.config.save_params()

    def save_trim_idx(self, idx):
        """Updates trim index of the current embryo."""
        emb_name = self.curr_emb_name
        self.config.save_manual_peak_data(emb_name, manual_trim_idx=idx)

    def save_peak_widths(self, emb_name, peak_widths, peak_index):
        corrected_peaks = self.config.get_corrected_peaks(emb_name)
        peak_key = str(peak_index)

        if corrected_peaks:
            manual_widths = corrected_peaks["manual_widths"]
            manual_widths[peak_key] = peak_widths
        else:
            manual_widths = {peak_key: peak_widths}

        self.config.save_manual_peak_data(emb_name, manual_widths=manual_widths)

    def add_peak(self, x, emb_name, trace, wlen=10):
        # load corrected data to reconcile with the new add
        corrected_peaks = self.config.get_corrected_peaks(emb_name)
        manual_remove = [] if not corrected_peaks else corrected_peaks["manual_remove"]

        new_peak, new_peaks, removed_peaks = self.pm.add_peak(
            x, trace, manual_remove, wlen
        )
        # update corrected peaks
        if corrected_peaks:
            corrected_peaks["manual_peaks"].append(new_peak)
            added_peaks = list(set(corrected_peaks["manual_peaks"]))
        else:
            added_peaks = [new_peak]

        self.config.save_manual_peak_data(
            emb_name, added_peaks=added_peaks, removed_peaks=removed_peaks, wlen=wlen
        )
        return new_peak, new_peaks

    def remove_peak(self, x, emb_name, trace, wlen=10):
        # load corrected data to reconcile with the new add
        corrected_peaks = self.config.get_corrected_peaks(emb_name)
        manual_add = [] if not corrected_peaks else corrected_peaks["manual_peaks"]
        manual_widths = (
            None if not corrected_peaks else corrected_peaks["manual_widths"]
        )

        removed, new_arr, added_peaks, peak_width_to_remove = self.pm.remove_peak(
            x, trace, manual_add, manual_widths
        )
        # update corrected peaks
        if corrected_peaks:
            to_remove = corrected_peaks["manual_remove"]
            removed = list(set(to_remove + removed))
            corrected_peaks["manual_remove"] = removed
        if manual_widths and peak_width_to_remove:
            del manual_widths[str(peak_width_to_remove)]

        self.config.save_manual_peak_data(
            emb_name,
            added_peaks=added_peaks,
            removed_peaks=removed,
            manual_widths=manual_widths,
        )
        return removed, new_arr

    def clear_manual_data(self):
        exp = self.get_curr_experiment()

        for emb in exp.embryos.values():
            emb.trace.to_add = []
            emb.trace.to_remove = []

        if "embryos" in self.config.data:
            self.config.data["embryos"] = {}

        self.config.save_params()

    def set_initial_state(self):
        self.groups: dict[str, dict[str, Experiment]] = {}
        self.curr_group = None
        self.to_remove: dict[str, set] = {}
        self.curr_exp = None
        self.curr_emb_name = None

    def reset_current_experiment(self):
        exp_path = self.config.data["exp_path"]
        del self.groups[self.curr_group][self.curr_exp]

        new_exp = Experiment(exp_path=exp_path)

        self.config = new_exp.config
        self.curr_exp = None
        self.curr_emb_name = None
        self.add_experiment(new_exp, group=self.curr_group)
        exp_params = self.config.get_exp_params()
        self.to_remove[new_exp.name] = set(exp_params.get("to_remove", []))

    def create_experiment(self, config: Config, group_name: str):
        self.config = config
        exp = Experiment(config.data["exp_path"], config)

        self.add_experiment(exp, group_name)
        exp_params = self.config.get_exp_params()
        self.to_remove[exp.name] = set(exp_params.get("to_remove", []))

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
            if utils.emb_id(emb_name) not in self.to_remove[exp_name]
        }

    def get_filtered_emb_numbers(self, exp_name, group_name=None):
        embs = self.get_filtered_embs(exp_name, group_name)
        return [utils.emb_id(emb_name) for emb_name in embs.keys()]

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

    def get_curr_embryo(self) -> Embryo:
        exp = self.get_curr_experiment()
        return exp.embryos[self.curr_emb_name]

    def get_curr_trace(self) -> Trace:
        exp = self.get_curr_experiment()
        return exp.embryos[self.curr_emb_name].trace

    def add_group(self, group):
        self.groups[group] = {}
        if self.curr_group is None:
            self.curr_group = group

    def has_combined_experiments(self):
        return len(self.get_curr_group()) > 1
