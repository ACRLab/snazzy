import json
from pathlib import Path
import pprint
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from gowf_analysis import utils


class PdParamsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


class ExpParams(BaseModel):
    """
    Exp params.

    Attributes
    ----------
    first_peak_threshold: int
        Minimum time, in minutes, for the first peak. Embryos with peaks before that will be ignored.
    to_exclude: list[str] | None
        List of embryo names to be excluded. These embryos won't even be created.
    to_remove: list[str] | None
        List of embryo names to be removed. These will be created and show up in the GUI as removed.
    dff_strategy: "baseline" | "local_minima"
        How to compute the dff baseline.
    has_transients: boolean
        If an experiment has transients, early peaks will be skipped.
    has_dsna: boolean
        If an experiment has embryos with dSNA, automatically calculates dSNA start and ignores peaks
        that happen after that point.
    """

    first_peak_threshold: int = 30
    to_exclude: list[str] = Field(default_factory=list)
    to_remove: list[str] = Field(default_factory=list)
    has_transients: bool = True
    has_dsna: bool = False


class PDParams(BaseModel):
    peak_width: float = 0.98
    freq: float = 0.0025
    dff_strategy: str = "local_minima"
    baseline_window_size: int = 81
    trim_zscore: float = 0.35
    ISI_factor: float = 4
    low_amp_threshold: float = 0.1
    fft_height: float = 0.04
    fft_prominence: float = 0.03
    local_thres_window_size: int = 300
    local_thres_value: float = 75
    local_thres_method: str = "percentile"
    port_peaks_window_size: int = 30
    port_peaks_thres: float = 70


class EmbryoParams(BaseModel):
    wlen: int = 30
    manual_peaks: list[int] = Field(default_factory=list)
    manual_remove: list[int] = Field(default_factory=list)
    manual_widths: dict[str, Any] = Field(default_factory={})
    manual_trim_idx: int = -1
    manual_phase1_end: int = -1
    manual_dsna_start: int = -1


class ConfigObj(BaseModel):
    # exp_path: str | None
    exp_params: ExpParams = Field(default_factory=ExpParams)
    pd_params: PDParams = Field(default_factory=PDParams)
    embryos: dict[str, EmbryoParams] = Field(default_factory=dict)


class Config:
    """
    Configuration data from Experiment class.

    Falls back to default values if any values are missing. The default values
    are specified in the BaseModel subclasses above.

    Attributes:
        exp_path (Path):
            Path to the `peak_detection_params.json` file.
            If not found, will hold the default params in memory.
        rel_root_path (str):
            Name of the directory used to determine the relative path.
    """

    def __init__(self, exp_path: Path, rel_root_path="data"):
        self.exp_path = exp_path
        self.rel_path = utils.convert_to_relative_path(exp_path, rel_root_path)
        self.config_path = self.exp_path / "peak_detection_params.json"

        self.default_params = ConfigObj().dict()

        self.data = self.load_data()
        self.data["exp_path"] = str(self.rel_path)

    def __str__(self):
        return pprint.pformat(self.data, sort_dicts=False)

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
            json.dump(self.data, f, cls=PdParamsEncoder, indent=4)

    def load_data(self):
        if not self.config_path.exists():
            return self.default_params
        else:
            return self.read_from_file()

    def read_from_file(self):
        """Reads Config data from file.

        Missing values are added using default values. If the file has invalid
        data it will be ignored and the default values will be used.
        """
        with open(self.config_path, "r") as f:
            data = json.load(f)
            try:
                return ConfigObj(**data).dict()
            except ValidationError as e:
                print(
                    "WARN: Could not read `peak_detection_params.json`. Using default values."
                )
                print(e)
                return self.default_params

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

    def get_corrected_dsna_start(self, emb_name):
        try:
            return self.data["embryos"][emb_name]["manual_dsna_start"]
        except KeyError:
            return None

    def save_manual_peak_data(
        self,
        emb_name,
        wlen=None,
        added_peaks=None,
        removed_peaks=None,
        manual_widths=None,
        manual_trim_idx=None,
        manual_phase1_end=None,
        manual_dsna_start=None,
    ):
        if "embryos" not in self.data:
            self.data["embryos"] = {}

        if not emb_name in self.data["embryos"]:
            self.data["embryos"][emb_name] = {
                "wlen": 10,
                "manual_peaks": [],
                "manual_remove": [],
                "manual_widths": {},
                "manual_trim_idx": -1,
                "manual_phase1_end": -1,
                "manual_dsna_start": -1,
            }

        if wlen is not None:
            self.data["embryos"][emb_name]["wlen"] = wlen
        if added_peaks is not None:
            self.data["embryos"][emb_name]["manual_peaks"] = added_peaks
        if removed_peaks is not None:
            self.data["embryos"][emb_name]["manual_remove"] = removed_peaks
        if manual_widths is not None:
            self.data["embryos"][emb_name]["manual_widths"] = manual_widths
        if manual_trim_idx is not None:
            self.data["embryos"][emb_name]["manual_trim_idx"] = manual_trim_idx
        if manual_phase1_end is not None:
            self.data["embryos"][emb_name]["manual_phase1_end"] = manual_phase1_end
        if manual_dsna_start is not None:
            self.data["embryos"][emb_name]["manual_dsna_start"] = manual_dsna_start

        self.save_params()
