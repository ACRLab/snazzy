from pathlib import Path

from pasna_analysis import Config, DataLoader, Embryo, utils


class Experiment:
    """Encapsulates data about all embryos for a given experiment.

    Attributes
    ----------
    exp_path: Path
        Path with `pasnascope` output.
    config: Config | None
        Config obj. If not provided will be created if a peak_detection_params.json file is found. If not, a file based on default params will be created.
    kwargs:
        See `self._parse_kwargs` for list of valid keys passed as kwargs.
    """

    def __init__(
        self,
        exp_path: str | Path,
        config: Config | None = None,
        verbose: bool = True,
        **kwargs,
    ):
        exp_path = Path(exp_path)
        self.directory = exp_path
        self.config = config if config is not None else Config(exp_path)

        self.exp_params = self.config.get_exp_params()

        if kwargs:
            self._parse_kwargs(kwargs)

        data = DataLoader(exp_path)
        if not self.config.config_path.exists():
            self.config.initialize_config_file()

        self.name = data.name
        self.act_paths = data.activities()
        self.len_paths = data.lengths()
        self.embryos = self._get_embryos()

        if verbose:
            print(f"Parameters used for experiment {self.name}:")
            print("-" * 50)
            print(self.config)

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
        update_exp_params = {k: v for k, v in kwargs.items() if k in self.exp_params}
        dff_strategy = kwargs.get("dff_strategy", None)

        to_update = {}
        if dff_strategy is not None:
            to_update["pd_params"] = {"dff_strategy": dff_strategy}
        if update_exp_params:
            to_update["exp_params"] = update_exp_params
        self.config.update_params(to_update)

    def _get_embryos(self) -> dict[str, Embryo]:
        """Returns all embryos where the first episode happend before `first_peak_threshold` minutes."""
        embryos = {}

        exp_params = self.config.get_exp_params()
        to_exclude = exp_params.get("to_exclude", [])
        for act_path, len_path in zip(self.act_paths, self.len_paths):
            emb_name = act_path.stem
            emb_id = utils.emb_id(emb_name)
            if emb_id in to_exclude:
                continue

            emb = Embryo(act_path, len_path, self.config)

            try:
                first_peak_threshold = exp_params.get("first_peak_threshold", 0)
                if emb.trace.peak_times[0] <= first_peak_threshold * 60:
                    print(
                        f"First peak detected before {first_peak_threshold} mins.",
                        f"Skipping {emb.name}..",
                    )
                    continue
            except (ValueError, IndexError):
                print(f"No peaks detected for {emb.name}. Skipping..")
            embryos[emb_name] = emb

        return embryos
