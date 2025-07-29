from pathlib import Path

import numpy as np


class DataLoader:
    """Access data about the current experiment.

    Attributes
    ----------
    path: Path
        The path that contains `pasnascope` output. Must follow the folder\
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
        """Asserts that folder structure matches `pasnascope` output."""
        if not self.path.exists():
            raise ValueError(f"Path not found: {self.path}")
        paths = (
            (self.path / f)
            for f in DataLoader.REQUIRED_DIRS + DataLoader.REQUIRED_FILES
        )
        if not all(path.exists() for path in paths):
            raise ValueError(
                "Could not find expected files. Is this really a directory from `pasnascope`?"
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
        dir = self.path.joinpath(dir_name)
        return sorted([e for e in dir.iterdir()], key=self.get_emb_id)

    def get_emb_id(self, emb_path: Path) -> int:
        """Return an embryo id based on a filepath.

        Filepaths that represent embryo data have the format embXX.csv."""
        emb_name = emb_path.stem
        return int(emb_name[3:])

    def load_csv(self, csv_path: Path) -> np.ndarray:
        """Read csv content as a 2D nparray."""
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        # csv files with a single row are read as 1D, but rest of the code expects 2D
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return data
