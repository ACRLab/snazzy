from snazzy_analysis import Config, Embryo, Dataset, utils
from snazzy_analysis.gui import PeakMatcher


class DatasetModel:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.to_remove = self.get_removed_embryos()
        self.selected_embryo = self.embryos[0]

    @property
    def embryos(self):
        """List of filtered embryos for this dataset."""
        return [e for e in self.all_embryos() if e.name not in self.to_remove]

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def get_embryo(self, emb_name):
        for embryo in self.all_embryos():
            if embryo.name == emb_name:
                return embryo
        raise ValueError(f"Could not find {emb_name} in dataset {self.name}.")

    def get_emb_ids(self):
        return [e.get_id() for e in self.embryos]

    def all_embryos(self):
        return self.dataset.get_all_embryos()

    def get_removed_embryos(self):
        manual_remove = self.dataset.exp_params.get("to_remove", set())
        if self.dataset.filtered_out is not None:
            removed_embryos = manual_remove.union(self.dataset.filtered_out)
        return removed_embryos

    def mark_as_accepted(self, emb_name):
        self.to_remove.remove(emb_name)

    def mark_as_removed(self, emb_name):
        self.to_remove.add(emb_name)


class GroupModel:
    def __init__(self, name: str):
        self.name = name
        self.datasets: dict[str, DatasetModel] = {}

    def add_dataset(self, dataset: DatasetModel):
        if dataset.name in self.datasets:
            raise ValueError("Dataset already added to this group.")
        self.datasets[dataset.name] = dataset

    def remove_dataset(self, dataset: DatasetModel):
        if dataset.name in self.datasets:
            del self.datasets[dataset.name]

    def iter_all_embryos(self):
        """Yield tuples of dataset_name, embryo for all valid embryos in a Group."""
        for dataset_name, dataset in self.datasets.items():
            for emb in dataset.embryos:
                yield dataset_name, emb


class Model:
    def __init__(self):
        self.pm = PeakMatcher()
        self.set_initial_state()

    def __str__(self):
        group_names = [g.name for g in self.groups]
        to_remove_count = {
            dataset.name: len(dataset.to_remove)
            for g in self.groups
            for dataset in g.datasets.values()
        }
        return (
            f"Model(\n"
            f"  groups: {group_names} groups\n"
            f"  curr_group: {self.selected_group.name}\n"
            f"  to_remove: {to_remove_count}\n"
            f"  curr_dataset: {self.selected_dataset.name}\n"
            f"  curr_emb_name: {self.selected_embryo.name}\n"
            f")"
        )

    @property
    def selected_trace(self):
        if self.selected_dataset is None:
            return None
        return self.selected_dataset.selected_embryo.trace

    @property
    def selected_embryo(self):
        if self.selected_dataset is None:
            return None
        return self.selected_dataset.selected_embryo

    def set_initial_state(self):
        self.groups: list[GroupModel] = []
        self.selected_group: GroupModel = None
        self.selected_dataset: DatasetModel = None

    def create_dataset(self, config: Config, group_name: str):
        dataset = Dataset(config.dataset_path, config)

        if not dataset.embryos:
            first_peak_threshold = config.get_exp_params()["first_peak_threshold"]
            raise AttributeError(
                f"Could not find any embryos with first peak after {first_peak_threshold} minutes."
            )

        config.save_params()
        return self.add_dataset(DatasetModel(dataset), group_name)

    def add_dataset(self, dataset: DatasetModel, group_name: str):
        group = self.get_group_by_name(group_name)
        if group is None:
            group = self.create_group(group_name)
            self.add_group(group)

        group.add_dataset(dataset)

        self.select_group(group)
        self.select_dataset(dataset)
        self.select_embryo(dataset.embryos[0])

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
        if group.datasets:
            self.select_dataset(next(iter(group.datasets.values())))

    def select_dataset(self, dataset: DatasetModel):
        if self.selected_dataset == dataset:
            return
        self.selected_dataset = dataset

    def select_embryo(self, embryo: Embryo):
        if self.selected_dataset.selected_embryo == embryo:
            return
        self.selected_dataset.selected_embryo = embryo

    def get_group_by_name(self, name: str) -> GroupModel | None:
        for group in self.groups:
            if group.name == name:
                return group
        return None

    def update_config(self, new_data):
        """Updates the config data for the current dataset."""
        dataset = self.selected_dataset
        dataset.config.update_params(new_data)
        dataset.config.save_params()

    def save_trim_idx(self, idx):
        """Updates trim index of the current embryo."""
        dataset = self.selected_dataset
        emb_name = dataset.selected_embryo.name
        dataset.config.save_manual_peak_data(emb_name, manual_trim_idx=idx)

    def save_phase1_end_idx(self, emb_name, idx):
        dataset = self.selected_dataset
        dataset.config.save_manual_peak_data(emb_name, manual_phase1_end=idx)

    def save_dsna_start(self, emb_name, idx):
        dataset = self.selected_dataset
        dataset.config.save_manual_peak_data(emb_name, manual_dsna_start=idx)

    def update_peak_widths(self, peak_index, line_index, new_line_pos):
        emb = self.selected_dataset.selected_embryo

        peak_bounds = emb.trace.peak_bounds_indices[peak_index]
        peak_bounds[line_index] = new_line_pos

        peak_bounds = peak_bounds.tolist()
        peak_index = int(emb.trace.peak_idxes[peak_index])
        self.save_peak_widths(emb.name, peak_bounds, peak_index)

    def save_peak_widths(self, emb_name, peak_widths, peak_index):
        dataset = self.selected_dataset
        corrected_peaks = dataset.config.get_corrected_peaks(emb_name)
        peak_key = str(peak_index)

        if corrected_peaks:
            manual_widths = corrected_peaks["manual_widths"]
            manual_widths[peak_key] = peak_widths
        else:
            manual_widths = {peak_key: peak_widths}

        dataset.config.save_manual_peak_data(emb_name, manual_widths=manual_widths)

    def calc_peaks_all_embs(self, pd_params=None):
        """Calculates peaks for all embryos in a given dataset.

        Persists the parameters used to calculate peaks in pd_params.json.

        There's no need to calculate all peaks for all datasets in a Group because
        the GUI does not support updating combined datasets.
        If there are combined datasets, the GUI will only present them."""
        dataset = self.selected_dataset
        if pd_params is None:
            pd_params = self.get_pd_params()

        for emb in dataset.all_embryos():
            emb.trace.detect_peaks(pd_params["freq"])
            emb.trace.compute_peak_bounds(pd_params["peak_width"])

        to_remove = self.selected_dataset.to_remove
        self.update_config(
            {"pd_params": pd_params, "exp_params": {"to_remove": to_remove}}
        )

    def add_peak(self, x, emb_name, trace, wlen=2):
        # load corrected data to reconcile with the new add
        dataset = self.selected_dataset
        corrected_peaks = dataset.config.get_corrected_peaks(emb_name)
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

        dataset.config.save_manual_peak_data(
            emb_name, added_peaks=added_peaks, removed_peaks=removed_peaks, wlen=wlen
        )
        return new_peak, new_peaks

    def remove_peak(self, x, emb_name, trace, wlen=2):
        # load corrected data to reconcile with the new add
        dataset = self.selected_dataset
        corrected_peaks = dataset.config.get_corrected_peaks(emb_name)
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

        dataset.config.save_manual_peak_data(
            emb_name,
            added_peaks=added_peaks,
            removed_peaks=removed,
            manual_widths=filtered_peak_widths,
        )
        return removed, new_arr

    def is_emb_accepted(self, emb_id):
        emb_name = f"emb{emb_id}"
        dataset = self.selected_dataset
        return emb_name in dataset.embryos

    def toggle_emb_visibility(self, emb_name, should_remove):
        dataset = self.selected_dataset
        if not should_remove:
            dataset.mark_as_accepted(emb_name)
        else:
            if emb_name == self.selected_embryo.name:
                self.render_next_embryo()
            dataset.mark_as_removed(emb_name)

        self.update_emb_visibility_in_config()
        return emb_name in dataset.to_remove

    def update_emb_visibility_in_config(self):
        dataset = self.selected_dataset
        new_data = {"exp_params": {"to_remove": dataset.to_remove}}
        self.update_config(new_data)

    def render_next_embryo(self):
        next_exp, next_emb = self.get_next_emb_name(forward=True)
        dataset = self.selected_group.datasets[next_exp]
        embryo = dataset.get_embryo(next_emb)
        self.select_dataset(dataset)
        self.select_embryo(embryo)

    def clear_manual_data_by_embryo(self, emb_name):
        dataset = self.selected_dataset

        target_emb = None
        for emb in dataset.all_embryos():
            if emb.name == emb_name:
                target_emb = emb
        if target_emb is None:
            raise ValueError(f"Cannot find {emb_name} in selected dataset.")

        target_emb.trace.to_add = []
        target_emb.trace.to_remove = []

        if "embryos" in dataset.config.data:
            emb_data = dataset.config.data["embryos"]
            if emb_name in emb_data:
                del emb_data[emb_name]

        dataset.config.save_params()

    def clear_all_manual_data(self):
        dataset = self.selected_dataset

        for emb in dataset.embryos:
            emb.trace.to_add = []
            emb.trace.to_remove = []

        if "embryos" in dataset.config.data:
            dataset.config.data["embryos"] = {}

        dataset.config.save_params()

    def reset_current_dataset(self):
        dataset = self.selected_dataset
        config = dataset.config

        group = self.selected_group
        group.remove_dataset(dataset)

        self.create_dataset(config, group.name)

    def get_trace_context(self, use_dev_time: bool = False):
        dataset = self.selected_dataset

        embryo = dataset.selected_embryo
        trace = embryo.trace

        if use_dev_time:
            time = embryo.lin_developmental_time
        else:
            time = trace.time / 60

        dff = trace.dff[: trace.trim_idx]

        return trace, time, time[: trace.trim_idx], dff

    def has_dsna(self):
        dataset = self.selected_dataset
        exp_params = dataset.config.get_exp_params()
        return exp_params.get("has_dsna", False)

    def has_combined_datasets(self):
        return len(self.selected_group.datasets) > 1

    def get_pd_params(self):
        dataset = self.selected_dataset
        return dataset.config.get_pd_params()

    def get_config_data(self):
        dataset = self.selected_dataset
        if dataset is None:
            return None
        return dataset.config.load_data()

    def get_next_emb_name(self, forward: bool) -> tuple[str, str]:
        """Return the next valid dataset_name and emb_name of the currenlty selected group.

        If an emb marked as to_remove is selected, returns the first valid embryo.

        Parameters:
            forward(bool):
                If True returns the next embryo, otherwise the previous embryo.
        Returns:
            next_values(tuple[str, str]):
                next_emb_name, next_dataset_name
        """
        if self.select_dataset is None:
            return

        datasets_and_embs = [
            (dataset_name, emb.name)
            for dataset_name, emb in self.selected_group.iter_all_embryos()
        ]

        datasets_and_embs.sort(key=lambda e: (e[0], utils.emb_id(e[1])))

        try:
            dataset = self.selected_dataset
            curr_emb_index = datasets_and_embs.index(
                (dataset.name, dataset.selected_embryo.name)
            )
        except ValueError:
            return datasets_and_embs[0]

        if forward:
            next_idx = (curr_emb_index + 1) % len(datasets_and_embs)
        else:
            next_idx = (curr_emb_index - 1) % len(datasets_and_embs)

        return datasets_and_embs[next_idx]

    def move_to_next_emb(self, forward):
        dataset_name, emb_name = self.get_next_emb_name(forward)

        dataset = self.selected_group.datasets[dataset_name]
        embryo = dataset.get_embryo(emb_name)

        self.select_dataset(dataset)
        self.select_embryo(embryo)

    def get_index_from_time(self, time) -> int:
        """Calculates signal index based on time.

        Relies on the fact that the acquisition rate is constant.

        Parameters:
            time (float): time in minutes.
        """
        dataset = self.selected_dataset
        exp_params = dataset.config.get_exp_params()

        return int(time * 60) // exp_params["acquisition_period"]
