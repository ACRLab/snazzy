from pathlib import Path
import json
import shutil
import sys

from gowf_analysis import utils


def update_dir(dir_path: Path):
    default_params = {
        "exp_params": {
            "first_peak_threshold": 30,
            "to_exclude": [],
            "to_remove": [],
            "has_transients": True,
        },
        "pd_params": {
            "dff_strategy": "local_minima",
            "peak_width": 0.98,
            "freq": 0.0025,
            "trim_zscore": 0.35,
        },
        "embryos": {},
    }

    pd_params = dir_path / "peak_detection_params.json"
    if not pd_params.exists():
        print(f"Could not find pd_params inside {dir_path}. Aborting..")
        return

    dest = dir_path / "peak_detection_params_old.json"
    shutil.copyfile(pd_params, dest)
    print("Wrote old data as `peak_detection_params_old.json`.")

    with open(pd_params, "r") as f:
        old_data = json.load(f)
        if "peak_width" in old_data:
            default_params["pd_params"]["peak_width"] = old_data["peak_width"]
        if "freq" in old_data:
            default_params["pd_params"]["freq"] = old_data["freq"]
        if "embryos" in old_data:
            default_params["embryos"] = old_data["embryos"]
        if "to_remove" in old_data:
            default_params["exp_params"]["to_remove"] = [
                int(x[3:]) for x in old_data["to_remove"]
            ]
        default_params["exp_path"] = str(
            utils.convert_to_relative_path(dir_path, "data")
        )

    with open(pd_params, "w") as f:
        json.dump(default_params, f, indent=4)
        print("Updated `peak_detection_params.json`.")


def update_dirs(file_path: Path):
    for file in file_path.iterdir():
        print(file)
        pd_params = file / "peak_detection_params.json"
        if file.is_dir() and pd_params.exists():
            update_dir(file)
        elif file.is_dir():
            update_dirs(file)


if __name__ == "__main__":
    try:
        rel_path = sys.argv[1]
    except IndexError:
        print("Usage: python3 convert_pd_params.py rel_path")
        exit(1)

    file_path = Path().cwd().joinpath(rel_path)

    update_dirs(file_path)
