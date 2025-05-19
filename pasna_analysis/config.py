import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from pasna_analysis import utils


@dataclass
class ExpParams:
    """
    Exp params.

    Attributes
    ----------
    first_peak_threshold: int
        Minimum time, in minutes, for the first peak. Embryos with peaks before that will be ignored.
    to_exclude: Optional[list[int]]
        List of embryo ids to be excluded. To exclude emb1 pass [1], for example.
    dff_strategy: baseline | local_minima
        How to compute the dff baseline.
    has_transients: boolean
        If an experiment has transients, early peaks will be skipped.
    """

    first_peak_threshold: int
    to_exclude: List[int]
    to_remove: List[int]
    has_transients: bool


@dataclass
class PDParams:
    peak_width: float
    freq: float
    dff_strategy: str
    trim_zscore: float = 0.35


@dataclass
class EmbryoParams:
    wlen: int
    manual_peaks: List[int]
    manual_remove: List[int]
    manual_widths: Dict[str, Any]


@dataclass
class ConfigObj:
    exp_path: str
    exp_params: ExpParams
    pd_params: PDParams
    embryos: Dict[str, EmbryoParams] = field(default_factory=dict)


def config_decoder(d):
    if "to_exclude" in d and "first_peak_threshold" in d:
        return ExpParams(**d)
    elif "peak_width" in d and "freq" in d:
        return PDParams(**d)
    elif "wlen" in d and "manual_peaks" in d:
        return EmbryoParams(**d)
    elif "exp_path" in d and "exp_params" in d and "pd_params" in d:
        return asdict(
            ConfigObj(
                exp_path=d["exp_path"],
                exp_params=d["exp_params"],
                pd_params=d["pd_params"],
                embryos=d["embryos"],
            )
        )
    return d


class Config:

    def __init__(self, exp_path: Path):
        self.exp_path = utils.convert_to_relative_path(exp_path, "data")
        self.config_path = self.exp_path / "peak_detection_params.json"

        self.default_params = {
            "exp_params": {
                "first_peak_threshold": 30,
                "to_exclude": [],  # excluded from an Experiment, won't be in the GUI
                "to_remove": [],  # shows in the GUI marked as removed
                "has_transients": True,
            },
            "pd_params": {
                "dff_strategy": "baseline",
                "peak_width": 0.92,
                "freq": 0.0025,
                "trim_zscore": 0.35,
            },
            "embryos": {},
        }
        self.default_params["exp_path"] = str(self.exp_path)

        self.data = self.load_data()

    def update_params(self, new_data: dict):
        valid_keys = self.default_params.keys()

        for key, value in new_data.items():
            if key not in valid_keys:
                raise KeyError(f"Invalid config key: {key}")

            if isinstance(value, dict) and isinstance(self.data.get(key), dict):
                self.data[key].update(value)
            else:
                self.data[key] = value

    def save_params(self):
        with open(self.config_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def load_data(self):
        if not self.config_path.exists():
            self.initialize_config_file()
            return self.default_params
        else:
            return self.read_from_file()

    def read_from_file(self):
        with open(self.config_path, "r") as f:
            data = json.load(f, object_hook=config_decoder)
        return data

    def initialize_config_file(self):
        with open(self.config_path, "w") as f:
            json.dump(self.default_params, f, indent=4)

    def get_pd_params(self):
        return self.data.get("pd_params", self.default_params["pd_params"])

    def get_exp_params(self):
        return self.data.get("exp_params", self.default_params["exp_params"])

    def get_corrected_peaks(self, emb_name):
        try:
            return self.data["embryos"][emb_name]
        except KeyError:
            return None

    def save_manual_peak_data(
        self,
        emb_name,
        added_peaks=None,
        removed_peaks=None,
        manual_widths=None,
        wlen=None,
    ):
        if "embryos" not in self.data:
            self.data["embryos"] = {}

        if not emb_name in self.data["embryos"]:
            self.data["embryos"][emb_name] = {
                "wlen": 10,
                "manual_peaks": [],
                "manual_remove": [],
                "manual_widths": {},
            }

        if added_peaks is not None:
            self.data["embryos"][emb_name]["manual_peaks"] = added_peaks
        if removed_peaks is not None:
            self.data["embryos"][emb_name]["manual_remove"] = removed_peaks
        if manual_widths is not None:
            self.data["embryos"][emb_name]["manual_widths"] = manual_widths
        if wlen is not None:
            self.data["embryos"][emb_name]["wlen"] = wlen

        self.save_params()
