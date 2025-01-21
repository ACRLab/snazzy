"""Functions to support interactive find peaks via matplotlib."""

import json
from pathlib import Path

import numpy as np


def get_initial_values(pd_params_path: Path | None = None):
    """Restores initial peak detection parameters.

    If not found, a file with default params will be created."""
    if pd_params_path and pd_params_path.exists():
        with open(pd_params_path) as f:
            data = json.load(f)
            default_params = {
                "order0_min": float(data["order0_min"]),
                "order1_min": float(data["order1_min"]),
                "mpd": int(data["mpd"]),
                "prominence": float(data["prominence"]),
                "rel_height": float(data["rel_height"]),
            }
    else:
        default_params = dict(
            order0_min=0.06, order1_min=0.006, mpd=70, prominence=0.2, rel_height=0.92
        )
        with open(pd_params_path, "w+") as f:
            json.dump(default_params, f, indent=4)
    return default_params


def save_detection_params(pd_params_path: Path, **kwargs):
    if pd_params_path.exists():
        with open(pd_params_path, "r") as f:
            config = json.load(f)
        for k, val in kwargs.items():
            config[k] = val
    else:
        config = kwargs
    with open(pd_params_path, "w+") as f:
        json.dump(config, f, indent=4)


def local_peak_at(x, signal, wlen):
    local_max_index = np.argmax(signal)
    adjusted_max = local_max_index + x - wlen
    return int(adjusted_max)


def save_add_peak(emb_name, config, x, wlen):
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
    # adjust manual_remove now that a new peak was added
    if "manual_remove" in config["embryos"][emb_name]:
        to_remove = config["embryos"][emb_name]["manual_remove"]
        try:
            peak = next(p for p in to_remove if x - wlen <= p <= x + wlen)
            i = to_remove.index(peak)
            to_remove = to_remove[:i] + to_remove[i + 1 :]
            config["embryos"][emb_name]["manual_remove"] = to_remove
        except StopIteration:
            pass


def save_remove_peak(emb_name, config, removed, x, wlen):
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
        except StopIteration:
            return
        del config["embryos"][emb_name]["manual_widths"][str(peak)]
