from pathlib import Path

from pasna_analysis import utils


class DataLoader:
    """Used to access data about the current experiment.

    Attributes
    ----------
    path: Path
        The path that contains the `pasnascope` output. Must follow the folder\
        structure described in this project's README.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = path.stem
        self.check_files()

    def check_files(self):
        """Asserts that the expected file structure is found."""
        required_dirs = ["activity", "lengths"]
        required_files = ["full-length.csv"]
        if not self.path.exists():
            raise ValueError(f"Path not found: {self.path}")
        paths = ((self.path / f) for f in required_dirs + required_files)
        if not all(path.exists() for path in paths):
            raise ValueError(
                "Could not find expected files. Is this really a directory from `pasnascope`?"
            )

    def embryos(self) -> list[str]:
        """Returns a list of available embryos."""
        activity_dir = self.path.joinpath("activity")
        return sorted([e.stem for e in activity_dir.iterdir()], key=utils.emb_id)

    def activities(self) -> list[Path]:
        """Returns a list of activity csv files."""
        activity_dir = self.path.joinpath("activity")
        return sorted(list(activity_dir.iterdir()), key=utils.emb_id)

    def lengths(self) -> list[Path]:
        """Returns a list of VNC length csv files."""
        length_dir = self.path.joinpath("lengths")
        return sorted(list(length_dir.iterdir()), key=utils.emb_id)

    def get_embryo_files_by_id(self, id: int) -> tuple[Path]:
        """Returns a tuple with activity and length files for a given `id`.

        If no files are found, returns None for each file."""
        emb = f"emb{id}"
        a = next((e for e in self.activities() if e.stem == emb), None)
        l = next((e for e in self.lengths() if e.stem == emb), None)
        return a, l
