import json
import shutil
import sys
from pathlib import Path

default_params = {
    "exp_params": {
        "first_peak_threshold": 30,
        "to_exclude": [],
        "to_remove": [],
        "has_transients": True,
    },
    "pd_params": {
        "dff_strategy": "baseline",
        "peak_width": 0.98,
        "freq": 0.0025,
    },
    "embryos": {},
}

if __name__ == "__main__":
    try:
        rel_path = sys.argv[1]
    except IndexError:
        print("Usage: python3 convert_pd_params.py rel_path")
        exit(1)

    file_path = Path().cwd().joinpath(rel_path)

    src = file_path / "peak_detection_params.json"
    if src.exists():
        dest = file_path / "peak_detection_params_old.json"
        shutil.copyfile(src, dest)
        print("Copied.")

        with open(src, "r") as f:
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
            default_params["exp_path"] = rel_path

        with open(src, "w") as f:
            json.dump(default_params, f, indent=4)
            print("wrote.")
