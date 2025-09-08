import numpy as np
from pathlib import Path


def split_in_bins(arr, bins):
    return np.digitize(arr, bins)


def emb_id(emb: Path | str) -> int:
    """Retuns the number that identifies a given embryos.

    Assumes that embryos are always named as emb + id, e.g: `emb21`."""
    if isinstance(emb, Path):
        emb = emb.stem
    return int(emb[3:])
