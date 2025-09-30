from pathlib import Path

from snazzy_analysis import Config, DataLoader, Embryo, utils


class Experiment:
    """Encapsulates data about all embryos for a given experiment.

    Attributes
    ----------
    exp_path: Path
        Path with `pasnascope` output.
    config: Config | None
        Config obj. If not provided will look for config data saved as json in
        the exp_path. If not found, a file with default params will be created.
    kwargs:
        See `self._parse_kwargs` for list of valid keys passed as kwargs.
    """

    def __init__(
        self,
        exp_path: str | Path,
        config: Config | None = None,
        **kwargs,
    ):
        exp_path = Path(exp_path)
        self.directory = exp_path
        self.config = config if config is not None else Config(exp_path)

        if kwargs:
            self._parse_kwargs(kwargs)

        self.exp_params = self.config.get_exp_params()

        self.data_loader = DataLoader(exp_path)
        # persist config to file if it only exists in memory
        if not self.config.config_path.exists():
            self.config.initialize_config_file()

        self.name = self.data_loader.name
        self.filtered_out = set(self.exp_params.get("to_remove", []))
        self._embryos = self._create_embryos()

    @property
    def embryos(self) -> list[Embryo]:
        """Embryos which first peak happens after first peak threshold."""
        return [
            emb for emb in self._embryos.values() if emb.name not in self.filtered_out
        ]

    def get_all_embryos(self) -> list[Embryo]:
        """All embryos calculated"""
        return [emb for emb in self._embryos.values()]

    def get_embryo(self, emb_name) -> Embryo:
        """Return Embryo based on name.

        Parameters:
            emb_name (str):
                Embryo name, eg: `emb12`
        """
        if emb_name not in self._embryos:
            raise ValueError(f"Cannot find {emb_name}.")
        return self._embryos[emb_name]

    def _parse_kwargs(self, kwargs):
        """Updates Config with valid kwargs.

        Attributes:
            kwargs: dict
                Valid keys are listed inside this function.
        """
        valid_params = {
            "has_transients",
            "to_exclude",
            "dff_strategy",
            "first_peak_threshold",
            "has_transients",
        }
        ignored_params = [kw for kw in kwargs if kw not in valid_params]
        if ignored_params:
            print(
                f"WARN: Some kwargs were ignored when creating a new Experiment: {ignored_params}."
            )
        exp_params_keys = self.config.default_params["exp_params"].keys()
        update_exp_params = {k: v for k, v in kwargs.items() if k in exp_params_keys}
        dff_strategy = kwargs.get("dff_strategy", None)

        to_update = {}
        if dff_strategy is not None:
            to_update["pd_params"] = {"dff_strategy": dff_strategy}
        if update_exp_params:
            to_update["exp_params"] = update_exp_params
        self.config.update_params(to_update)

    def _create_embryos(self) -> dict[str, Embryo]:
        embryos = {}

        emb_size_data = self.data_loader.load_csv(
            self.directory.joinpath("full-length.csv")
        )
        to_exclude = self.exp_params.get("to_exclude", [])

        for act_path, len_path in self.data_loader.get_data_path_pairs():
            emb_name = act_path.stem
            emb_id = utils.emb_id(emb_name)
            if emb_id in to_exclude:
                continue

            act_data = self.data_loader.load_csv(act_path)
            len_data = self.data_loader.load_csv(len_path)
            emb = Embryo(act_data, len_data, emb_size_data, emb_name, self.config)

            try:
                first_peak_threshold = self.exp_params.get("first_peak_threshold", 0)
                if emb.trace.peak_times[0] <= first_peak_threshold * 60:
                    print(
                        f"First peak detected before {first_peak_threshold} mins.",
                        f"Skipping {emb.name}..",
                    )
                    self.filtered_out.add(emb.name)
            except (ValueError, IndexError):
                print(f"No peaks detected for {emb.name}. Skipping..")
                self.filtered_out.add(emb.name)
            embryos[emb.name] = emb

        if self.filtered_out:
            self.config.update_params({"exp_params": {"to_remove": self.filtered_out}})

        return embryos
