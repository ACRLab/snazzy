import json
from pathlib import Path

import numpy as np

from pasna_analysis import Trace


def local_peak_at(x, signal, wlen):
    local_max_index = np.argmax(signal)
    adjusted_max = local_max_index + x - wlen
    return int(adjusted_max)


class PeakFinder:

    def __init__(self):
        self.default_params = dict(
            order0_min=0.06,
            order1_min=0.006,
            mpd=70,
            prominence=0.2,
            peak_width=0.92,
            to_remove=[],
        )

    def initialize_config_file(self, config_path):
        with open(config_path, "w") as f:
            json.dump(self.default_params, f, indent=4)

    def get_pd_params(self, config_path: Path):
        """Returns peak detection params from config file.

        Creates an empty config file with default params if it doesn't exist."""
        if not config_path.exists():
            self.initialize_config_file(config_path)
            return self.default_params

        with open(config_path, "r") as f:
            data = json.load(f)

        if "to_remove" not in data:
            data["to_remove"] = []

        return {
            "order0_min": float(data["order0_min"]),
            "order1_min": float(data["order1_min"]),
            "mpd": int(data["mpd"]),
            "prominence": float(data["prominence"]),
            "peak_width": float(data["peak_width"]),
            "to_remove": list(data["to_remove"]),
        }

    def save_pd_params(self, config_path: Path, **kwargs):
        """Writes all keyword arguments to the file at config_path."""
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            for k, val in kwargs.items():
                config[k] = val
        else:
            config = kwargs
        with open(config_path, "w+") as f:
            json.dump(config, f, indent=4)

    def add_peak(self, x: int, emb_name: str, config_path: Path, trace: Trace, wlen=10):
        """Adds a new peak in the vicinity of `x`.

        The new peak is calculated as the local maximum near `x`, in a window of 2*wlen points.
        """
        window = slice(x - wlen, x + wlen)
        new_peak = local_peak_at(x, trace.order_zero_savgol[window], wlen)
        new_arr = np.append(trace.peak_idxes, new_peak)
        new_arr.sort()

        self.write_add_peak(config_path, emb_name, new_peak, wlen)

        return new_peak, new_arr

    def write_add_peak(self, config_path: Path, emb_name: str, x: int, wlen: int):
        """Writes manually added peaks to `config_path`.

        Reconciles the manual_remove and manual_peaks lists, since they can't have
        overlapping peaks."""

        with open(config_path, "r") as f:
            config = json.load(f)
        if "embryos" not in config:
            config["embryos"] = {}
        if not emb_name in config["embryos"]:
            config["embryos"][emb_name] = {
                "wlen": wlen,
                "manual_peaks": [],
                "manual_remove": [],
                "manual_widths": {},
            }

        to_add = config["embryos"][emb_name]["manual_peaks"]
        to_add.append(x)
        config["embryos"][emb_name]["manual_peaks"] = list(set(to_add))

        # if we add a peak in a place where there's a manually removed peak,
        # that manually removed entry must be erased
        if "manual_remove" in config["embryos"][emb_name]:
            to_remove = config["embryos"][emb_name]["manual_remove"]
            try:
                peak = next(p for p in to_remove if x - wlen <= p <= x + wlen)
                i = to_remove.index(peak)
                to_remove = to_remove[:i] + to_remove[i + 1 :]
                config["embryos"][emb_name]["manual_remove"] = to_remove
            except StopIteration:
                pass

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    def remove_peak(
        self, x: int, emb_name: str, config_path: Path, trace: Trace, wlen=10
    ) -> tuple[list[int], list[int]]:
        """Removes peaks that are within `wlen` of `x`."""
        target = (trace.peak_idxes >= x - wlen) & (trace.peak_idxes <= x + wlen)
        removed = trace.peak_idxes[target].tolist()
        new_arr = trace.peak_idxes[~target]

        self.write_remove_peak(config_path, emb_name, x, removed, wlen)

        return removed, new_arr

    def write_remove_peak(
        self, config_path: Path, emb_name: str, x: int, removed: list[int], wlen: int
    ):
        with open(config_path, "r") as f:
            config = json.load(f)
        if "embryos" not in config.keys():
            config["embryos"] = {}

        if not emb_name in config["embryos"]:
            config["embryos"][emb_name] = {
                "wlen": wlen,
                "manual_peaks": [],
                "manual_remove": [],
                "manual_widths": {},
            }
        to_remove = config["embryos"][emb_name]["manual_remove"]
        config["embryos"][emb_name]["manual_remove"] = list(set(to_remove + removed))
        # adjust manual_peaks now that a new peak was removed
        if "manual_peaks" in config["embryos"][emb_name]:
            to_add = config["embryos"][emb_name]["manual_peaks"]
            try:
                # FIXME: if you remove more than one element, this update method fails
                peak = next(p for p in to_add if x - wlen <= p <= x + wlen)
                i = to_add.index(peak)
                to_add = to_add[:i] + to_add[i + 1 :]
                config["embryos"][emb_name]["manual_peaks"] = to_add
            except StopIteration:
                pass
        if "manual_widths" in config["embryos"][emb_name]:
            widths = config["embryos"][emb_name]["manual_widths"]
            try:
                peak = next(int(p) for p in widths if x - wlen <= int(p) <= x + wlen)
                del config["embryos"][emb_name]["manual_widths"][str(peak)]
            except StopIteration:
                pass

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    def save_peak_widths(self, config_path, emb_name, peak_widths, peak_index):
        print(f"Inside save_peak_widths, peak index is: {peak_index}")
        with open(config_path, "r") as f:
            config = json.load(f)
        if "embryos" not in config:
            config["embryos"] = {}

        if not emb_name in config["embryos"]:
            config["embryos"][emb_name] = {
                "wlen": 10,
                "manual_peaks": [],
                "manual_remove": [],
                "manual_widths": {},
            }

        config["embryos"][emb_name]["manual_widths"][peak_index] = peak_widths

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
