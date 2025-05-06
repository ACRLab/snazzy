from pathlib import Path
from typing import Optional

from pasna_analysis import DataLoader, Embryo, utils


class Experiment:
    """Encapsulates data about all embryos for a given experiment.

    Attributes
    ----------
    exp_path: Path
        Path with `pasnascope` output.
    first_peak_threshold: int
        Minimum time, in minutes, for the first peak. Embryos with peaks before that
    will be ignored.
    to_exclude: Optional[list[int]]
        List of embryo ids to be excluded. To exclude emb1 pass [1], for example.
    dff_strategy: baseline | local_minima
        How to compute the dff baseline.
    has_transients: boolean
        If an experiment has transients, early peaks will be skipped.
    """

    def __init__(
        self,
        exp_path: Path,
        first_peak_threshold=30,
        to_exclude: Optional[list[int]] = None,
        dff_strategy="baseline",
        has_transients=True,
    ):
        self.to_exclude = [] if to_exclude is None else to_exclude
        data = DataLoader(exp_path)
        self.directory = exp_path
        self.name = data.name
        self.pd_params_path = data.pd_params_path
        self.pd_params = None

        self.first_peak_threshold = first_peak_threshold
        self.dff_strategy = dff_strategy
        self.has_transients = has_transients

        self.act_paths = data.activities()
        self.len_paths = data.lengths()
        self.embryos = self._get_embryos()

    def _get_embryos(self) -> dict[str, Embryo]:
        embryos = {}

        for act_path, len_path in zip(self.act_paths, self.len_paths):
            emb_name = act_path.stem
            emb_id = utils.emb_id(emb_name)
            if emb_id in self.to_exclude:
                continue

            emb = Embryo(
                act_path,
                len_path,
                self.dff_strategy,
                self.has_transients,
                self.pd_params_path,
                self.pd_params,
            )

            try:
                if emb.trace.get_first_peak_time() <= self.first_peak_threshold * 60:
                    print(
                        f"First peak detected before {self.first_peak_threshold} mins.",
                        f"Skipping {emb.name}..",
                    )
                    continue
            except (ValueError, IndexError):
                print(f"No peaks detected for {emb.name}. Skipping..")
            embryos[emb_name] = emb

        return embryos

    def set_pd_params(self, pd_params):
        pd_params_keys = ["mpd", "order0_min", "order1_min", "prominence"]
        if any(param not in pd_params_keys for param in pd_params):
            raise ValueError("Missing params in pd_params file")
        self.pd_params = pd_params
        self.embryos = self._get_embryos()
