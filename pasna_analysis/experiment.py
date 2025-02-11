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
        has_transients=False,
    ):
        if to_exclude is None:
            to_exclude = []

        data = DataLoader(exp_path)
        self.directory = exp_path
        self.name = data.name
        self.pd_params_path = data.pd_params_path

        self.first_peak_threshold = first_peak_threshold
        self.dff_strategy = dff_strategy
        self.has_transients = has_transients

        act_paths = data.activities()
        len_paths = data.lengths()
        self.embryos = self._get_embryos(act_paths, len_paths, to_exclude)

    def _get_embryos(
        self, act_paths: list[Path], len_paths: list[Path], to_exclude: list[int]
    ) -> dict[str, Embryo]:
        embryos = {}

        for act_path, len_path in zip(act_paths, len_paths):
            emb_name = act_path.stem
            emb_id = utils.emb_id(emb_name)
            if emb_id in to_exclude:
                continue

            emb = Embryo(
                act_path,
                len_path,
                self.dff_strategy,
                self.has_transients,
                self.pd_params_path,
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
