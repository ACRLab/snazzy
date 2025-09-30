from pathlib import Path

import numpy as np


def read(csv_path: Path, delimiter=",", skip_header=1, usecols=None) -> np.ndarray:
    """Read csv file as a numpy array.

    Parameters:
        csv_path (Path):
            Path to csv file.
        delimiter (str):
            Delimiter used in the csv file. Defaults to ','.
        skip_header (int):
            Number of header lines, that will be skipped.
        usecols: (list[int] | None):
            List with column numbers to read.
    """
    return np.genfromtxt(
        csv_path, delimiter=delimiter, skip_header=skip_header, usecols=usecols
    )


def write_files(
    csv_paths: list[Path], signals: np.ndarray, header: list[str], fmt="%.2f"
):
    """Write data as csv.

    Parameters:
        csv_paths (list[Path]):
            List of csv paths for files that will be created.
        signals (np.ndarray):
            Array with data to be saved as csv.
        header (list[str]):
            Column names.
        fmt (str):
            Format string. Refer to `np.savetxt` docs for details.
            Defaults to "%.2f".
    """
    for csv_path, signal in zip(csv_paths, signals):
        if csv_path.exists():
            print(f"File {csv_path.stem} already exists. Skipping..")
            continue
        write_file(csv_path, signal, header, fmt)


def write_file(csv_path: Path, signal: np.ndarray, header: list[str], fmt="%.2f"):
    """Write a csv file.

    Parameters:
        csv_path (Path):
            Path to save the csv file.
        signal (np.ndarray):
            Array with data to be saved as csv.
        header (list[str]):
            Column names.
        fmt (str):
            Format string. Refer to `np.savetxt` docs for details.
            Defaults to "%.2f".
    """
    header_line = ", ".join(header)
    np.savetxt(
        csv_path,
        signal,
        delimiter=",",
        header=header_line,
        comments="",
        fmt=fmt,
    )
