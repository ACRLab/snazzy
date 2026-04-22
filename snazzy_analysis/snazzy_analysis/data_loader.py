from pathlib import Path

import numpy as np

from snazzy_analysis import utils


class DataLoader:
    """Loads data from a dataset.

    Attributes
    ----------
    path: Path
        The path that contains `snazzy_processing` output. Must follow the folder\
        structure described in this project's README.
    """

    REQUIRED_DIRS = ["activity", "lengths"]
    REQUIRED_FILES = ["full-length.csv"]

    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = path.stem
        self.check_files()
        self.check_embs_match()

    def check_files(self):
        """Asserts that folder structure matches `snazzy_processing` output."""
        if not self.path.exists():
            raise ValueError(f"Path not found: {self.path}")
        paths = (
            (self.path / f)
            for f in DataLoader.REQUIRED_DIRS + DataLoader.REQUIRED_FILES
        )
        if not all(path.exists() for path in paths):
            raise ValueError(
                "Could not find expected files. Is this really a directory from `snazzy_processing`?"
            )

    def check_embs_match(self):
        """Each embryo must have a file in `activity` and `lengths` dirs."""
        for act_file, len_file in self.get_data_path_pairs():
            if act_file.name != len_file.name:
                raise ValueError(
                    "Could not process this dataset. Mismatch between embryo data in activity and length directory."
                )

    def get_data_path_pairs(self):
        """Iterator with pairs of activity and lenght filepaths."""
        return zip(
            self.get_filenames_sorted_by_emb_number("activity"),
            self.get_filenames_sorted_by_emb_number("lengths"),
        )

    def get_filenames_sorted_by_emb_number(self, dir_name: str) -> list[Path]:
        dir_path = self.path.joinpath(dir_name)
        ids_and_filenames = []
        for e in dir_path.iterdir():
            if e.suffix == ".csv":
                try:
                    ids_and_filenames.append((utils.emb_id(e), e))
                except ValueError:
                    print(f"Could not parse filename {e.name}. Skipping..")
                    continue

        return [f for _, f in sorted(ids_and_filenames)]

    def load_csv(self, csv_path: Path) -> np.ndarray:
        """Read csv content as a 2D nparray."""
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        # csv files with a single row are read as 1D, but rest of the code expects 2D
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return data
