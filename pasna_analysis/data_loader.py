import json
from pathlib import Path


class DataLoader:
    """Used to access data about the current experiment.

    Params:
    -------
    path: Path
        The path that contains the `pasnascope` output. Must follow the folder\
        structure described in this project's README.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = path.stem
        self.pd_params_path = self.path.joinpath("peak_detection_params.json")
        self.check_files()

    def check_files(self):
        """Asserts that the expected file structure is found."""
        required_dirs = ["activity", "lengths"]
        required_files = ["full-length.csv"]
        paths = ((self.path / f) for f in required_dirs + required_files)
        assert all(
            path.exists() for path in paths
        ), "Could not find expected files. Is this really a directory from `pasnascope`?"

    def embryos(self) -> list[str]:
        """Returns a list of available embryos."""
        activity_dir = self.path.joinpath("activity")
        return sorted([e.stem for e in activity_dir.iterdir()], key=emb_id)

    def activities(self) -> list[Path]:
        """Returns a list of activity csv files."""
        activity_dir = self.path.joinpath("activity")
        return sorted(list(activity_dir.iterdir()), key=emb_id)

    def lengths(self) -> list[Path]:
        """Returns a list of VNC length csv files."""
        length_dir = self.path.joinpath("lengths")
        return sorted(list(length_dir.iterdir()), key=emb_id)

    def get_embryo_files_by_id(self, id: int) -> tuple[Path]:
        """Returns a tuple with activity and length files for a given `id`.

        If no files are found, returns None for each file."""
        emb = f"emb{id}"
        a = next((e for e in self.activities() if e.stem == emb), None)
        l = next((e for e in self.lengths() if e.stem == emb), None)
        return a, l

    def peak_detection_props(self) -> None | dict:
        """Reads peak detection params that were added by the user."""
        if self.peak_props_path.exists():
            print("Found peak detection params.json")
            with open(self.peak_props_path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError as e:
                    print("Could not read `peak_detection_params.json`.")
                    raise e


def emb_id(emb: Path | str) -> int:
    """Assumes that embryos are always named as emb + id, e.g: `emb21`."""
    if isinstance(emb, Path):
        emb = emb.stem
    return int(emb[3:])
