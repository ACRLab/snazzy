from snazzy_analysis import Config, Embryo, Experiment, utils
from snazzy_analysis.gui import PeakMatcher


class ExperimentModel:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.to_remove = self.get_removed_embryos()
        self.selected_embryo = self.embryos[0]

    @property
    def embryos(self):
        """List of filtered embryos for this experiment."""
        return [e for e in self.all_embryos() if e.name not in self.to_remove]

    def __getattr__(self, name):
        return getattr(self.experiment, name)

    def get_embryo(self, emb_name):
        for embryo in self.all_embryos():
            if embryo.name == emb_name:
                return embryo
        raise ValueError(f"Could not find {emb_name} in experiment {self.name}.")

    def get_emb_ids(self):
        return [e.get_id() for e in self.embryos]

    def all_embryos(self):
        return self.experiment.get_all_embryos()

    def get_removed_embryos(self):
        manual_remove = self.experiment.exp_params.get("to_remove", set())
        if self.experiment.filtered_out is not None:
            removed_embryos = manual_remove.union(self.experiment.filtered_out)
        return removed_embryos

    def mark_as_accepted(self, emb_name):
        self.to_remove.remove(emb_name)

    def mark_as_removed(self, emb_name):
        self.to_remove.add(emb_name)


class GroupModel:
    def __init__(self, name: str):
        self.name = name
        self.experiments: dict[str, ExperimentModel] = {}

    def add_experiment(self, exp: ExperimentModel):
        if exp.name in self.experiments:
            raise ValueError("Experiment already added to this group.")
        self.experiments[exp.name] = exp

    def remove_experiment(self, exp: ExperimentModel):
        if exp.name in self.experiments:
            del self.experiments[exp.name]

    def iter_all_embryos(self):
        """Yield tuples of exp_name, embryo for all valid embryos in a Group."""
        for exp_name, exp in self.experiments.items():
            for emb in exp.embryos:
                yield exp_name, emb


class Model:
    def __init__(self):
        self.pm = PeakMatcher()
        self.set_initial_state()

    def __str__(self):
        group_names = [g.name for g in self.groups]
        to_remove_count = {
            exp.name: len(exp.to_remove)
            for g in self.groups
            for exp in g.experiments.values()
        }
        return (
            f"Model(\n"
            f"  groups: {group_names} groups\n"
            f"  curr_group: {self.selected_group.name}\n"
            f"  to_remove: {to_remove_count}\n"
            f"  curr_exp: {self.selected_experiment.name}\n"
            f"  curr_emb_name: {self.selected_embryo.name}\n"
            f")"
        )

    @property
    def selected_trace(self):
        if self.selected_experiment is None:
            return None
        return self.selected_experiment.selected_embryo.trace

    @property
    def selected_embryo(self):
        if self.selected_experiment is None:
            return None
        return self.selected_experiment.selected_embryo

    def set_initial_state(self):
        self.groups: list[GroupModel] = []
        self.selected_group: GroupModel = None
        self.selected_experiment: ExperimentModel = None

    def create_experiment(self, config: Config, group_name: str):
        exp = Experiment(config.exp_path, config)

        if not exp.embryos:
            first_peak_threshold = config.get_exp_params()["first_peak_threshold"]
            raise AttributeError(
                f"Could not find any embryos with first peak after {first_peak_threshold} minutes."
            )

        config.save_params()
        return self.add_experiment(ExperimentModel(exp), group_name)

    def add_experiment(self, experiment: ExperimentModel, group_name: str):
        group = self.get_group_by_name(group_name)
        if group is None:
            group = self.create_group(group_name)
            self.add_group(group)

        group.add_experiment(experiment)

        self.select_group(group)
        self.select_experiment(experiment)
        self.select_embryo(experiment.embryos[0])

    def create_group(self, group_name: str) -> GroupModel:
        for g in self.groups:
            if g.name == group_name:
                return
        group = GroupModel(group_name)
        self.add_group(group)
        return group

    def add_group(self, group: GroupModel):
        for g in self.groups:
            if g.name == group.name:
                return
        self.groups.append(group)

    def select_group(self, group: GroupModel):
        if self.selected_group == group:
            return
        self.selected_group = group
        if group.experiments:
            self.select_experiment(next(iter(group.experiments.values())))

    def select_experiment(self, experiment: ExperimentModel):
        if self.selected_experiment == experiment:
            return
        self.selected_experiment = experiment

    def select_embryo(self, embryo: Embryo):
        if self.selected_experiment.selected_embryo == embryo:
            return
        self.selected_experiment.selected_embryo = embryo

    def get_group_by_name(self, name: str) -> GroupModel | None:
        for group in self.groups:
            if group.name == name:
                return group
        return None

    def update_config(self, new_data):
        """Updates the config data for the current experiment."""
        exp = self.selected_experiment
        exp.config.update_params(new_data)
        exp.config.save_params()

    def save_trim_idx(self, idx):
        """Updates trim index of the current embryo."""
        exp = self.selected_experiment
        emb_name = exp.selected_embryo.name
        exp.config.save_manual_peak_data(emb_name, manual_trim_idx=idx)

    def save_phase1_end_idx(self, emb_name, idx):
        exp = self.selected_experiment
        exp.config.save_manual_peak_data(emb_name, manual_phase1_end=idx)

    def save_dsna_start(self, emb_name, idx):
        exp = self.selected_experiment
        exp.config.save_manual_peak_data(emb_name, manual_dsna_start=idx)

    def update_peak_widths(self, peak_index, line_index, new_line_pos):
        emb = self.selected_experiment.selected_embryo

        peak_bounds = emb.trace.peak_bounds_indices[peak_index]
        peak_bounds[line_index] = new_line_pos

        peak_bounds = peak_bounds.tolist()
        peak_index = int(emb.trace.peak_idxes[peak_index])
        self.save_peak_widths(emb.name, peak_bounds, peak_index)

    def save_peak_widths(self, emb_name, peak_widths, peak_index):
        exp = self.selected_experiment
        corrected_peaks = exp.config.get_corrected_peaks(emb_name)
        peak_key = str(peak_index)

        if corrected_peaks:
            manual_widths = corrected_peaks["manual_widths"]
            manual_widths[peak_key] = peak_widths
        else:
            manual_widths = {peak_key: peak_widths}

        exp.config.save_manual_peak_data(emb_name, manual_widths=manual_widths)

    def calc_peaks_all_embs(self, pd_params=None):
        """Calculates peaks for all embryos in a given experiment.

        Persists the parameters used to calculate peaks in pd_params.json.

        There's no need to calculate all peaks for all experiments in a Group because
        the GUI does not support updating combined experiments.
        If there are combined experiments, the GUI will only present them."""
        exp = self.selected_experiment
        if pd_params is None:
            pd_params = self.get_pd_params()

        for emb in exp.all_embryos():
            emb.trace.detect_peaks(pd_params["freq"])
            emb.trace.compute_peak_bounds(pd_params["peak_width"])

        to_remove = self.selected_experiment.to_remove
        self.update_config(
            {"pd_params": pd_params, "exp_params": {"to_remove": to_remove}}
        )

    def add_peak(self, x, emb_name, trace, wlen=2):
        # load corrected data to reconcile with the new add
        exp = self.selected_experiment
        corrected_peaks = exp.config.get_corrected_peaks(emb_name)
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

        exp.config.save_manual_peak_data(
            emb_name, added_peaks=added_peaks, removed_peaks=removed_peaks, wlen=wlen
        )
        return new_peak, new_peaks

    def remove_peak(self, x, emb_name, trace, wlen=2):
        # load corrected data to reconcile with the new add
        exp = self.selected_experiment
        corrected_peaks = exp.config.get_corrected_peaks(emb_name)
        manual_add = [] if not corrected_peaks else corrected_peaks["manual_peaks"]
        manual_widths = {} if not corrected_peaks else corrected_peaks["manual_widths"]
        wlen = wlen if not corrected_peaks else corrected_peaks["wlen"]

        removed, new_arr, added_peaks, filtered_peak_widths = self.pm.remove_peak(
            x, trace, manual_add, manual_widths, wlen
        )
        # update corrected peaks
        if corrected_peaks:
            to_remove = corrected_peaks["manual_remove"]
            removed = list(set(to_remove + removed))
            corrected_peaks["manual_remove"] = removed
            corrected_peaks["manual_widths"] = filtered_peak_widths

        exp.config.save_manual_peak_data(
            emb_name,
            added_peaks=added_peaks,
            removed_peaks=removed,
            manual_widths=filtered_peak_widths,
        )
        return removed, new_arr

    def is_emb_accepted(self, emb_id):
        emb_name = f"emb{emb_id}"
        exp = self.selected_experiment
        return emb_name in exp.embryos

    def toggle_emb_visibility(self, emb_name, should_remove):
        exp = self.selected_experiment
        if not should_remove:
            exp.mark_as_accepted(emb_name)
        else:
            if emb_name == self.selected_embryo.name:
                self.render_next_embryo()
            exp.mark_as_removed(emb_name)

        self.update_emb_visibility_in_config()
        return emb_name in exp.to_remove

    def update_emb_visibility_in_config(self):
        exp = self.selected_experiment
        new_data = {"exp_params": {"to_remove": exp.to_remove}}
        self.update_config(new_data)

    def render_next_embryo(self):
        next_exp, next_emb = self.get_next_emb_name(forward=True)
        experiment = self.selected_group.experiments[next_exp]
        embryo = experiment.get_embryo(next_emb)
        self.select_experiment(experiment)
        self.select_embryo(embryo)

    def clear_manual_data_by_embryo(self, emb_name):
        exp = self.selected_experiment

        target_emb = None
        for emb in exp.all_embryos():
            if emb.name == emb_name:
                target_emb = emb
        if target_emb is None:
            raise ValueError(f"Cannot find {emb_name} in selected experiment.")

        target_emb.trace.to_add = []
        target_emb.trace.to_remove = []

        if "embryos" in exp.config.data:
            emb_data = exp.config.data["embryos"]
            if emb_name in emb_data:
                del emb_data[emb_name]

        exp.config.save_params()

    def clear_all_manual_data(self):
        exp = self.selected_experiment

        for emb in exp.embryos:
            emb.trace.to_add = []
            emb.trace.to_remove = []

        if "embryos" in exp.config.data:
            exp.config.data["embryos"] = {}

        exp.config.save_params()

    def reset_current_experiment(self):
        exp = self.selected_experiment
        config = exp.config

        group = self.selected_group
        group.remove_experiment(exp)

        self.create_experiment(config, group.name)

    def get_trace_context(self, use_dev_time: bool = False):
        exp = self.selected_experiment

        embryo = exp.selected_embryo
        trace = embryo.trace

        if use_dev_time:
            time = embryo.lin_developmental_time
        else:
            time = trace.time / 60

        dff = trace.dff[: trace.trim_idx]

        return trace, time, time[: trace.trim_idx], dff

    def has_dsna(self):
        exp = self.selected_experiment
        exp_params = exp.config.get_exp_params()
        return exp_params.get("has_dsna", False)

    def has_combined_experiments(self):
        return len(self.selected_group.experiments) > 1

    def get_pd_params(self):
        exp = self.selected_experiment
        return exp.config.get_pd_params()

    def get_config_data(self):
        exp = self.selected_experiment
        if exp is None:
            return None
        return exp.config.load_data()

    def get_next_emb_name(self, forward: bool) -> tuple[str, str]:
        """Return the next valid exp_name and emb_name of the currenlty selected group.

        If an emb marked as to_remove is selected, returns the first valid embryo.

        Parameters:
            forward(bool):
                If True returns the next embryo, otherwise the previous embryo.
        Returns:
            next_values(tuple[str, str]):
                next_emb_name, next_exp_name
        """
        if self.select_experiment is None:
            return

        exp_and_embs = [
            (exp_name, emb.name)
            for exp_name, emb in self.selected_group.iter_all_embryos()
        ]

        exp_and_embs.sort(key=lambda e: (e[0], utils.emb_id(e[1])))

        try:
            exp = self.selected_experiment
            curr_emb_index = exp_and_embs.index((exp.name, exp.selected_embryo.name))
        except ValueError:
            return exp_and_embs[0]

        if forward:
            next_idx = (curr_emb_index + 1) % len(exp_and_embs)
        else:
            next_idx = (curr_emb_index - 1) % len(exp_and_embs)

        return exp_and_embs[next_idx]

    def move_to_next_emb(self, forward):
        exp_name, emb_name = self.get_next_emb_name(forward)

        experiment = self.selected_group.experiments[exp_name]
        embryo = experiment.get_embryo(emb_name)

        self.select_experiment(experiment)
        self.select_embryo(embryo)

    def get_index_from_time(self, time) -> int:
        """Calculates signal index based on time.

        Relies on the fact that the acquisition rate is constant.

        Parameters:
            time (float): time in minutes.
        """
        exp = self.selected_experiment
        exp_params = exp.config.get_exp_params()

        return int(time * 60) // exp_params["acquisition_period"]
