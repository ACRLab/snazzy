import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GroundTruthData:

    exp_date: str
    fly_line: str
    emb_name: str


@dataclass
class GroundTruthPointData(GroundTruthData):
    """Keep track of annotated peaks for a given embryo."""

    episode_idxes: list
    oscillation_idxes: list


@dataclass
class GroundTruthWindowData(GroundTruthData):
    """Keep track of peak boundaries for a given embryo."""

    episode_bounds: list
    oscillation_bounds: list


class PeakAnnotationParser:
    """Reads annotated peak data from a csv file.

    Expects the file name to follow the format: expDate_flyLine_embName_annotated.csv.
    Also expects csv files to have two columns, first for peak index and two for label.
    """

    def __init__(self, dir_path: str | Path, annot_type: str):
        self.dir_path = Path(dir_path)
        self.validate_annot_type(annot_type)
        if annot_type == "point":
            self.annotations = self.load_point_data()
        elif annot_type == "window":
            self.annotations = self.load_window_data()

    def validate_annot_type(self, annot_type):
        valid_types = ["window", "point"]
        if annot_type not in valid_types:
            raise ValueError(
                f"Invalid annot_type: {annot_type}, expected one of {valid_types}"
            )

    def load_point_data(self, suffix="annotated.csv"):
        csv_paths = [f for f in self.dir_path.iterdir() if f.name.endswith(suffix)]

        annotations = {}

        for csv_path in csv_paths:
            exp_date, fly_line, emb_name, _ = csv_path.name.split("_")
            episode_idxes = []
            oscillation_idxes = []
            with open(csv_path, newline="") as csv_file:
                rdr = csv.reader(csv_file)
                # skip header:
                next(rdr)
                for idx, label in rdr:
                    if label == "Episode":
                        episode_idxes.append(int(idx))
                    elif label == "Oscillation":
                        oscillation_idxes.append(int(idx))
                    elif label == "dSNA":
                        continue
                    else:
                        print(
                            f"WARN: got an unexpected label: {label}. File: {csv_path.name}"
                        )
            annotation = GroundTruthPointData(
                exp_date, fly_line, emb_name, episode_idxes, oscillation_idxes
            )
            key = f"{exp_date}_{fly_line}_{emb_name}"
            annotations[key] = annotation

        return annotations

    def load_window_data(self, suffix="windows.csv"):
        csv_paths = [f for f in self.dir_path.iterdir() if f.name.endswith(suffix)]

        annotations = {}

        for csv_path in csv_paths:
            exp_date, fly_line, emb_name, _ = csv_path.name.split("_")
            episode_bounds = []
            oscillation_bounds = []
            with open(csv_path, newline="") as csv_file:
                rdr = csv.reader(csv_file)
                next(rdr)
                for start, end, label in rdr:
                    if label == "Episode":
                        episode_bounds.append((int(start), int(end)))
                    elif label == "Oscillation":
                        oscillation_bounds.append((int(start), int(end)))
                    elif label == "dSNA" or label == "Baseline":
                        continue
                    else:
                        print(
                            f"WARN: got an unexpected label: {label}. File: {csv_path.name}"
                        )
            annotation = GroundTruthWindowData(
                exp_date, fly_line, emb_name, episode_bounds, oscillation_bounds
            )
            key = f"{exp_date}_{fly_line}_{emb_name}"
            annotations[key] = annotation

        return annotations

    def get_annotation_by_exp_name(self, exp_name: str) -> list[GroundTruthPointData]:
        """Returns all GroundTruthData relative to an `exp_name`.

        Parameters:
            exp_name: str in the format: `expDate_flyLine`

        Returns:
            annotations: list with all annotated data found for that dataset
        """
        if self.annotations is None:
            raise AttributeError(
                "Cannot read annotation data, first call `self.load_point_data` or `self.load_window_data`."
            )
        annotations = []
        for annot_name, annot in self.annotations.items():
            exp_date, fly_line, _ = annot_name.split("_")
            if f"{exp_date}_{fly_line}" == exp_name:
                annotations.append(annot)
        return annotations

    def get_exp_names(self) -> list[str]:
        """Returns the names of all datasets that have GT data."""
        if self.annotations is None:
            raise AttributeError(
                "Cannot read annotation data, first call `self.load_point_data` or `self.load_window_data`."
            )
        exp_names = set()

        for k in self.annotations.keys():
            name, _ = k.split("_emb")
            exp_names.add(name)

        return list(exp_names)
