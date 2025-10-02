import numpy as np
from pathlib import Path


def split_in_bins(arr: np.ndarray, bins: int):
    """Return an array of bin indices for each arr element."""
    return np.digitize(arr, bins)


def emb_id(emb: Path | str) -> int:
    """Retun the number that identifies a given embryos.

    Assumes that embryos are always named as emb + id, e.g: `emb21`."""
    if isinstance(emb, Path):
        emb = emb.stem
    return int(emb[3:])


def emb_id_from_filename(emb_path: str) -> int:
    """Return the embryo id based on filename.

    Assumes that files are named as embXX-chY.

    Parameters:
        emb_path (str):
            Full path to embryo file, as a string.
    """
    emb_name = Path(emb_path).stem
    return int(emb_name.split("-")[0][3:])
