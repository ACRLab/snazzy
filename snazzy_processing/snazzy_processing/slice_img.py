from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

from nd2 import ND2File
from skimage.filters import threshold_triangle
from skimage.morphology import binary_closing, octagon
from skimage.exposure import equalize_hist
from tifffile import imread, imwrite, TiffFile
import numpy as np

from snazzy_processing import utils


def get_metadata(img_path: Path):
    """Returns image metadata, used to open it as a memory map."""
    with TiffFile(img_path) as tif:
        series = tif.series[0]
        shape = series.shape
        offset = series.dataoffset
        dtype = np.dtype(tif.byteorder + series.dtype.char)

    return offset, dtype, shape


def save_as_tiff(file: Path, dest_path: Path):
    """Converts an nd2 image to tiff.

    Does not overwrite the file if `dest_path` exists.

    Parameters:
        file (Path):
            Path to nd2 file.
        dest_path (Path):
            Path to save tiff file.
    """
    dest = Path(dest_path)
    if dest.exists():
        print(f"File '{dest.name}' already exists.")
        return
    with ND2File(file) as f:
        f.write_tiff(dest_path, progress=True)


def save_first_frames_as_tiff(file: Path, dest_path: Path, n: int):
    """Save the first frames as tif.

    Does not overwrite the file if `dest_path` exists.

    Parameters:
        file (Path):
            Path to tif or nd2 file.
        dest_path (Path):
            Path to save tif file.
        n (int):
            Number of frames to save.
    """
    dest = Path(dest_path)
    if dest.exists():
        print(f"File '{dest.name}' already exists.")
        return
    if file.suffix == ".tif" or file.suffix == ".tiff":
        # tiffile will slice over pages, so we need to reshape
        initial_frames = imread(file, key=slice(0, 2 * n))
        _, y, x = initial_frames.shape
        initial_frames = np.reshape(initial_frames, (n, 2, y, x))

        imwrite(dest_path, initial_frames)
    else:
        with ND2File(file) as f:
            darray = f.to_dask()
            initial_frames = darray[:n].compute()
            imwrite(dest_path, initial_frames)


def get_threshold(img: np.ndarray, thres_adjust=0) -> float:
    """Returns image threshold using the triangle method.

    Parameters:
        img (np.ndarray):
            2D np array.
        thres_adjust (int):
            Increment the calculated threshold by this amount.
            Makes it easy to manually adjust the threshold.
            Defaults to 0.
    """
    return threshold_triangle(img) + thres_adjust


def within_boundaries(r: int, c: int, rows: int, cols: int) -> bool:
    """Checks if `r` and `c` are valid coords in a `rows`x`cols` matrix."""
    return r >= 0 and r < rows and c >= 0 and c < cols


def mark_neighbors(img: np.ndarray, row: int, col: int, s: int):
    """Mark all points connected to (row, col) within the search box.

    Parameters:
        img (np.ndarray):
            2D np array.
        row (int):
            Row coordinate.
        col (int):
            Columns coordinate.
        s (int):
            Search box size.

    Returns:
        counter: amount of pixels marked as visited
        extremes: list with min and max pixel value for both dimensions
    """
    neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    rows, cols = img.shape
    dq = utils.CounterDeque(img.shape)

    for dr in range(s):
        for dc in range(s):
            if (
                within_boundaries(row + dr, col + dc, rows, cols)
                and img[row + dr, col + dc] == 1
            ):
                dq.append((row + dr, col + dc))

    while len(dq) > 0:
        r, c = dq.pop()
        if img[r, c] == 1:
            dq.update_extremes(r, c)
            # 255 represents a visited pixel
            img[r, c] = 255
            for dr, dc in neighbors:
                nr = r + dr
                nc = c + dc
                if within_boundaries(nr, nc, rows, cols):
                    if img[nr, nc] == 1:
                        dq.append((nr, nc))
    return dq.counter, dq.extremes


def get_bbox_boundaries(
    img: np.ndarray, s=25, n_cols=3, min_pixel_count=6000
) -> dict[int, list[int]]:
    """Calculate bounding boxes.

    BBoxes are only calculated for connected components with more than
    `min_pixel_count` points.

    Each bounding box is represented as the max and min values for both coordinates.

    Parameters:
        img (np.ndarray):
            2D np array
        s (int):
            Iteration step when searching for connected components.
            Defaults to 25.
        n_cols (int):
            Number of columns in the grid of embryos.
            Used to numerate the embryos.
        min_pixel_count (int):
            Minimum number of items in a connected region to consider it an embryo.
            Defaults to 6000.

    Returns:
        extremes (dict[int, list[int]]):
            List of bbox coordinates, sorted in F-order.
    """
    extremes = []
    rows, cols = img.shape
    # iterate over slices of size sxs
    for r in range(0, rows, s):
        for c in range(0, cols, s):
            slc = img[r : r + s, c : c + s]
            # pixel value of 255 means it is marked as visited
            if 255 in slc:
                continue
            if 1 in slc:
                counter, extreme = mark_neighbors(img, r, c, s)
                # minimum amount of pixels to be considered a VNC
                if counter > min_pixel_count:
                    extremes.append(extreme)
    extremes = sort_by_grid_pos(extremes, n_cols)
    return extremes


def increase_bbox(coords: dict[int, list[int]], w: int, h: int, shape: tuple):
    """Increases the bbox boundaries by w and h.

    Parameters:
        coords (dict[int, list[int]]):
            Extremes (output from `slice_img.calculate_slice_coordinates`).
        w (int):
            Number of pixels to increment in the bbox width (half each side).
        h (int):
            Number of pixels to increment in the bbox height (half each side).
        shape (tuple):
            Shape of the original image where the bboxes were calculated.

    Returns:
        new_coords (dict[int, list[int]]):
            New dict with expanded bboxes
    """
    new_coords = {}
    for emb_id, coord in coords.items():
        x0, x1, y0, y1 = coord
        r, c = shape
        new_x0 = max(x0 - h // 2, 0)
        new_x1 = min(x1 + h // 2, r)
        new_y0 = max(y0 - w // 2, 0)
        new_y1 = min(y1 + w // 2, c)
        new_coords[emb_id] = [new_x0, new_x1, new_y0, new_y1]
    return new_coords


def sort_by_grid_pos(extremes: list[list[int]], n_cols: int):
    """Sorts each boundary points list based on their position in the grid.

    Sorts by F-order (column-wise).

    Parameters:
        extremes (list[list[int]]):
            List of bbox coordinates
        n_cols (int):
            Number of columns to use to order embryos.

    Returns:
        sorted_bboxes (dict[int, list[int]]):
            Dict of bbox order to bbox coordinates.
    """
    centroids = [
        ((r0 + r1) // 2, (c0 + c1) // 2, i)
        for i, (r0, r1, c0, c1) in enumerate(extremes)
    ]
    # determine bin_size from the maximum column value
    bin_size = (max((c[1] for c in centroids)) // n_cols) + 1

    bins = [[] for _ in range(n_cols)]
    # add centroids to their respective bins
    for centroid in centroids:
        col = centroid[1]
        bin_idx = col // bin_size
        bins[bin_idx].append(centroid)

    # filter out possibly empty bins
    bins = [b for b in bins if len(b) > 0]

    # sort bins by row value
    for b in bins:
        b.sort(key=lambda b: b[0])

    # extracts each index
    indices = [b[2] for bin in bins for b in bin]
    return {k + 1: extremes[i] for k, i in enumerate(indices)}


def filter_by_embryos(
    extremes: dict[int, list[int]], selected_embryos: list[int]
) -> dict[int, list[int]]:
    """Filter the extremes dict, by only keeping the selected_embryos.

    Parameters:
        extremes (dict[int, list[int]]):
            List of bbox coordinates
        selected_embryos (int):
            List with embryos to keep, that match `extremes` keys.

    Returns:
        filtered_bboxes (dict[int, list[int]]):
            Dict of bbox order to bbox coordinates.
    """
    return {k: extremes[k] for k in selected_embryos if k in extremes}


def read_mmap(mmap_path: Path, num_frames=None):
    """Read mmaped-file contents."""
    offset, dtype, shape = get_metadata(mmap_path)
    if num_frames:
        shape = (num_frames, *shape[1:])
    return np.memmap(mmap_path, dtype, "r", offset, shape)


def create_tasks(extremes, channels, active_ch, dest, overwrite):
    """Wraps args to be submitted with `save_movie` calls to ThreadPoolExecutor.

    Parameters:
        extremes (dict[int, list[int]]):
            Dict that maps emb ids to bbox coordinates.
        channels (list[int]):
            List of channel numbers.
        active_ch (int):
            What channel number represents the active channel.
        dest (Path):
            Output path.
        overwrite (bool):
            If saved movies should overwrite existing ones or not.

    Returns:
        tasks:
            List of arguments used by `save_movie`.
    """
    tasks = []
    for id, extreme in extremes.items():
        x0, x1, y0, y1 = extreme
        for ch in range(channels):
            file_name = output_file_name(id, ch, active_ch)
            output = dest.joinpath(file_name)
            if output.exists() and not overwrite:
                print(
                    f"{file_name} already found. To overwrite the file, pass `overwrite=True`."
                )
            else:
                tasks.append((ch, x0, x1, y0, y1, output))
    return tasks


def submit_tasks(img: np.ndarray, tasks: list):
    """Run `save_movie` for all bbox coordinates args represented by tasks."""
    num_threads = os.cpu_count()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(save_movie, img, *task) for task in tasks]
        for future in futures:
            future.result()


def cut_movies(
    extremes,
    img_path,
    dest,
    embryos=None,
    active_ch=1,
    channels=2,
    overwrite=False,
):
    """Extracts movies from ch1 and ch2, based on the boundaries passed for
    each item of `extremes`.

    Parameters:
        extremes dict[int, list[int]]:
            `emb_number: [min_r, max_r, min_c, max_c]`.
        img_path (int):
            path to the raw image that will be cut.
        dest (Path):
            Directory where the movies will be saved.
        embryos (list[int]):
            List of embryo numbers. Used to select a subgroup of embryos.
            If not provided, all embryos will be processed.
        active_ch (1 | 2):
            Indicates the image active channel.
            Defaults to 1.
        channels (1 | 2):
            Number of channels imaged.
            Defaults to 2.
        overwrite (bool):
            Determine if movies should be overwritten."""
    if embryos:
        extremes = filter_by_embryos(extremes, embryos)
    if active_ch not in [1, 2]:
        raise ValueError(f"Active channel should be 1 or 2, got {active_ch}.")
    if channels not in [1, 2]:
        raise ValueError(f"Can only parse 1 or 2 channels, but got {channels}")

    img = read_mmap(img_path)

    dest_path = Path(dest)
    tasks = create_tasks(extremes, channels, active_ch, dest_path, overwrite)

    if len(tasks) == 0:
        return

    submit_tasks(img, tasks)


def output_file_name(id: int, ch: int, active_ch: int) -> str:
    """File name based on embryo id and ch number.

    Parameters:
        id (int):
            Embryo id.
        ch (int):
            Channel number
        active_ch (1 | 2):
            Specifies which is the active channel.
    """
    if active_ch != 1 and active_ch != 2:
        raise ValueError(f"Active channel should be 1 or 2, got {active_ch}.")
    ch_number = ch + 1 if active_ch == 1 else active_ch - ch
    return f"emb{id}-ch{ch_number}.tif"


def save_movie(
    img: np.ndarray, ch: int, x0: int, x1: int, y0: int, y1: int, output: Path
):
    """Write a slice of the movie based on the provided coordinates."""
    movie = img[:, ch, x0:x1, y0:y1]
    imwrite(output, movie)


def boundary_to_rect_coords(boundary: list[int]) -> list[int]:
    """Converts from `(x0, x1, y0, y1)` to `(x, y, w, h)`."""
    [x0, x1, y0, y1] = boundary
    return [x0, y0, y1 - y0, x1 - x0]


def calculate_slice_coordinates(
    img_path: Path, n_cols=3, thres_adjust=0
) -> dict[int, list[int]]:
    """Returns boundary points for all images in `img_path`.

    Parameters:
        img_path (Path):
            Absolute path to the raw data
        n_cols (int):
            Number of columns in the FOV grid, used to enforce the naming
            convention of the extracted embryos.
        thres_adjust (int):
            Increment the calculated threshold by this amount.
            Makes it easy to manually adjust the threshold.
            Defaults to 0.
    """
    binary_img = get_initial_binary_image(img_path, thres_adjust=thres_adjust)

    return get_bbox_boundaries(binary_img, s=25, n_cols=n_cols)


def get_initial_frames_from_mmap(img_path: Path, n=10):
    """Returns the first n frames from the file at `img_path`."""
    return read_mmap(img_path, num_frames=n)


def get_first_image(img_path: Path):
    """Returns the first image from for plotting.

    The channel 2 frames are averaged and equalized, for better visualization."""
    img = imread(img_path)
    print(img.shape)
    first_frame = np.average(img[:, 1, :, :], axis=0)
    return equalize_hist(first_frame)


def get_initial_binary_image(img_path: Path, n=10, thres_adjust=0):
    """Binarizes the first `n` slices of the img, which is read as a mmap."""
    img = get_initial_frames_from_mmap(img_path, n=n)

    frame = img[:, 1, :, :].copy()
    frame = np.max(frame, axis=0)

    thres = get_threshold(frame, thres_adjust=thres_adjust)
    frame[frame < thres] = 0
    frame[frame >= thres] = 1
    opened = np.zeros_like(frame, dtype=np.uint8)
    binary_closing(frame, footprint=octagon(5, 5), out=opened)
    return opened
