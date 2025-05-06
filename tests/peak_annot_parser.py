import csv
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnnotationData:
    """Keep track of annotated peaks for a given embryo"""

    exp_date: str
    fly_line: str
    emb_name: str
    episode_idxes: list
    oscillation_idxes: list


class PeakAnnotationParser:
    """Reads annotated peak data from a csv file.

    Expects the file name to follow the format: expDate_flyLine_embName_annotated.csv.
    Also expects csv files to have two columns, first for peak index and two for label.
    """

    def __init__(self, dir_path):
        self.dir_path = Path(dir_path)
        self.annotations = self.load_data()

    def load_data(self, suffix="annotated.csv"):
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
            annotation = AnnotationData(
                exp_date, fly_line, emb_name, episode_idxes, oscillation_idxes
            )
            key = f"{exp_date}_{fly_line}_{emb_name}"
            annotations[key] = annotation

        return annotations

    def get_annotation_by_exp_name(self, exp_name) -> list[AnnotationData]:
        """Returns all AnnotationData relative to an `exp_name`.

        Parameters:
            exp_name: str in the format: `expDate_flyLine`

        Returns:
            annotations: list with all annotated data found for that experiment
        """
        annotations = []
        for annot_name, annot in self.annotations.items():
            exp_date, fly_line, emb_name = annot_name.split("_")
            if f"{exp_date}_{fly_line}" == exp_name:
                annotations.append(annot)
        return annotations

    def get_exp_names(self):
        exp_names = set()

        for k in self.annotations.keys():
            name, _ = k.split("_emb")
            exp_names.add(name)

        return list(exp_names)
