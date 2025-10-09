import json
from pathlib import Path
import pprint
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class PdParamsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


SET_KEYS = {"to_remove", "to_exclude"}


def set_decoder(obj):
    """Convert specific list-valued keys back to sets."""
    for key in SET_KEYS:
        if key in obj and isinstance(obj[key], list):
            obj[key] = set(obj[key])
    return obj


class ExpParams(BaseModel):
    """
    Params related to a Dataset.

    Attributes
    ----------
    first_peak_threshold: int
        Minimum time, in minutes, for the first peak. Embryos with peaks before that will be ignored.
    to_exclude: list[str] | None
        List of embryo ids to be excluded. These embryos won't even be created.
    to_remove: list[str] | None
        List of embryo ids to be removed. These will be created and show up in the GUI as removed.
    has_transients: boolean
        If a dataset has transients, early peaks will be skipped.
    acquisition_period: int
        The time (in seconds) interval between acquiring two successive frames in a given channel.
    """

    first_peak_threshold: int = 30
    to_exclude: set[str] = Field(default_factory=set)
    to_remove: set[str] = Field(default_factory=set)
    has_transients: bool = True
    has_dsna: bool = False
    acquisition_period: int = 6


class PDParams(BaseModel):
    """Parameters used in peak detection.

    While this is a long list of parameters, the default values work well for many DFF traces.
    `freq` and `peak_width` might require some tuning and the GUI has sliders for changing them.
    They are exposed here to avoid hiding scattered magic values in the peak detection code and
    to make them easier to change if necessary.

    Attributes
    ----------
    peak_width: float
        Value between 0 and 1 (inclusive), used to calculate peak width.
        Passed to `scipy.signal.peak_widths`, see `rel_height` in that function for detailed description.
        Default value is 0.98.
    freq: float
        Frequency cutoff used to find peaks.
        See `trace.calculate_peaks` for details.
        Default value is 0.0025.
    dff_strategy: str
        Method used to calculate F0, and therefore dff, since dff = (F - F0) / F0.
        Options are 'local_minima' (default) and 'baseline'.
        See `trace.average_n_local_window` and `trace.compute_baseline` for method descriptions.
    baseline_window_size: int
        Number of points used to calculate F0.
        Should be an odd number.
        Defaults to 81.
    trim_zscore: float
        Z-value threshold used to determine hatching.
        See `trace.trim_data` for more details.
        Defaults to 0.35.
    ISI_factor: float
        The inter-spike interval (ISI) is used to ignore early ramps in signal that are misindentified as bursts.
        If the interval between the first two peaks is greater than `average interval * ISI_factor`, that peak is ignored.
        See `trace.remove_transients` for details.
    low_amp_threshold: float
        Ignores peaks that have amplitude lower than `low_amp_threshold * max_peak_amplitude`.
        Defaults to 0.1.
    fft_height: float
        Minimum amplitude to detect peaks in the low-passed filtered signal.
        Defaults to 0.04.
    fft_prominence: float
        Complements fft_height for detecting peaks in the freq domain.
        See `trace.calculate_peaks` for details.
        Defaults to 0.03.
    local_thres_window_size: int
        Window size used to filter peaks by local threshold.
        See `trace.filter_peaks_by_local_threshold` for details.
        Defaults to 300.
    local_thres_value: float
        Percentage of the local maximum, used to filter peaks.
        See `trace.filter_peaks_by_local_threshold` for details.
        Defaults to 75.
    port_peaks_window_size: int
        Window size to port peaks from filtered signal to dff signal.
        Defaults to 30.
    port_peaks_thres: float
        Minimum value as a percentage of the maximum peak within `port_peaks_window_size` required when porting a peak to dff values.
        Defaults to 70.
    """

    peak_width: float = 0.98
    freq: float = 0.0025
    dff_strategy: str = "local_minima"
    baseline_window_size: int = 81
    trim_zscore: float = 0.35
    ISI_factor: float = 3.0
    low_amp_threshold: float = 0.1
    fft_height: float = 0.04
    fft_prominence: float = 0.03
    local_thres_window_size: int = 300
    local_thres_value: float = 75.0
    local_thres_method: str = "percentile"
    port_peaks_window_size: int = 30
    port_peaks_thres: float = 70.0


class EmbryoParams(BaseModel):
    """Values that can be manually changed using the GUI, for a single Embryo.

    Attributes:
    -----------
        wlen: int
            window length (half size)used when adding or removing peaks.
            defaults to 2.
        manual_peaks: list[int]
            Indices marked as a manual peak.
        manual_remove: list[int]
            Indices where calculated peaks will be ignored within that index +- wlen.
        manual_widths: dict[str, Any]
            Maps indices to start and end coordinates of a peak.
            The index represents a peak.
        manual_trim_idx: int
            Index used as the trim_idx.
            All dff data after trim_idx is ignored.
    """

    wlen: int = 2
    manual_peaks: list[int] = Field(default_factory=list)
    manual_remove: list[int] = Field(default_factory=list)
    manual_widths: dict[str, Any] = Field(default_factory={})
    manual_trim_idx: int = -1
    manual_phase1_end: int = -1
    manual_dsna_start: int = -1


class ConfigObj(BaseModel):
    exp_params: ExpParams = Field(default_factory=ExpParams)
    pd_params: PDParams = Field(default_factory=PDParams)
    embryos: dict[str, EmbryoParams] = Field(default_factory=dict)


class Config:
    """
    Configuration data from Dataset class.

    Falls back to default values if any values are missing. The default values
    are specified in the BaseModel subclasses above.

    Attributes:
        dataset_path (Path):
            Path to the `peak_detection_params.json` file.
            If not found, will hold the default params in memory.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.config_path = dataset_path / "peak_detection_params.json"

        self.default_params = ConfigObj().dict()

        self.data = self.load_data()

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
            data = json.load(f, object_hook=set_decoder)
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
            json.dump(self.default_params, f, cls=PdParamsEncoder, indent=4)

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
                "wlen": 2,
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
