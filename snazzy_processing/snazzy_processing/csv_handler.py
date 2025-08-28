import numpy as np


def read(csv_path, delimiter=",", skip_header=1, usecols=None):
    return np.genfromtxt(
        csv_path, delimiter=delimiter, skip_header=skip_header, usecols=usecols
    )


def write_files(csv_paths, signals, header, fmt="%.2f"):
    for csv_path, signal in zip(csv_paths, signals):
        if csv_path.exists():
            print(f"File {csv_path.stem} already exists. Skipping..")
            continue
        write_file(csv_path, signal, header, fmt)


def write_file(csv_path, signal, header, fmt="%.2f"):
    header_line = ", ".join(header)
    np.savetxt(
        csv_path,
        signal,
        delimiter=",",
        header=header_line,
        comments="",
        fmt=fmt,
    )
