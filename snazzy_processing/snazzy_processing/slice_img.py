from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

from nd2 import ND2File
from skimage.filters import threshold_triangle
from skimage.morphology import binary_closing, octagon
from skimage.exposure import equalize_hist
from tifffile import imwrite, TiffFile
import numpy as np

from snazzy_processing import utils


def get_metadata(img_path):
    """Returns image metadata, used to open it as a memory map."""
    with TiffFile(img_path) as tif:
        series = tif.series[0]
        shape = series.shape
        offset = series.dataoffset
        dtype = np.dtype(tif.byteorder + series.dtype.char)

    return offset, dtype, shape


def save_as_tiff(file, dest_path):
    """Converts an nd2 image to tiff."""
    dest = Path(dest_path)
    if dest.exists():
        print(f"File '{dest.name}' already exists.")
        return
    with ND2File(file) as f:
        f.write_tiff(dest_path, progress=True)


def save_first_frames_as_tiff(file, dest_path, n):
    """Converts the first n slices of an nd2 image to tiff."""
    dest = Path(dest_path)
    if dest.exists():
        print(f"File '{dest.name}' already exists.")
        return
    with ND2File(file) as f:
        darray = f.to_dask()
        initial_frames = darray[:n].compute()
        imwrite(dest_path, initial_frames)


def get_threshold(img, thres_adjust=0):
    """Returns image threshold using the triangle method."""
    return threshold_triangle(img) + thres_adjust


def within_boundaries(i, j, r, c):
    """Checks if `i` and `j` are valid coords in a `r`x`c` matrix."""
    return i >= 0 and i < r and j >= 0 and j < c


def mark_neighbors(img, x, y, s):
    """Expands from the search box and marks points connected to any point
    within the search box.

    Returns:
        counter: amount of pixels marked as visited
        extremes: list with min and max pixel value for both dimensions
    """
    neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    r, c = img.shape
    dq = utils.CounterDeque(img.shape)

    for di in range(s):
        for dj in range(s):
            if within_boundaries(x + di, y + dj, r, c) and img[x + di, y + dj] == 1:
                dq.append((x + di, y + dj))
    while len(dq) > 0:
        i, j = dq.pop()
        if img[i, j] == 1:
            dq.update_extremes(i, j)
            # 255 represents a visited pixel
            img[i, j] = 255
            for ii, jj in neighbors:
                if within_boundaries(i + ii, j + jj, r, c):
                    if img[i + ii, j + jj] == 1:
                        dq.append((i + ii, j + jj))
    return dq.counter, dq.extremes


def get_bbox_boundaries(img, s=25, n_cols=3):
    """Gets bbox boundaries (max and min values for both coordinates)."""
    extremes = []
    r, c = img.shape
    # iterate over slices of size sxs
    for i in range(0, r, s):
        for j in range(0, c, s):
            slc = img[i : i + s, j : j + s]
            # pixel value of 255 means it is marked as visited
            if 255 in slc:
                continue
            if 1 in slc:
                counter, extreme = mark_neighbors(img, i, j, s)
                # minimum amount of pixels to be considered a VNC
                if counter > 6000:
                    extremes.append(extreme)
    extremes = sort_by_grid_pos(extremes, n_cols)
    return extremes


def increase_bbox(coords, w, h):
    """Increases the bbox boundaries by w and h.

    Args:
        coords: Extremes (output from `slice_img.calculate_slice_coordinates`)
        w: int Number of pixels to increment in the bbox width (half each side)
        h: int Number of pixels to increment in the bbox height (half each side)
    """
    new_coords = coords.copy()
    for k, coord in new_coords.items():
        x0, x1, y0, y1 = coord
        new_coords[k] = x0 - h // 2, x1 + h // 2, y0 - w // 2, y1 + w // 2
    return new_coords


def sort_by_grid_pos(extremes, n_cols):
    """Sorts each boundary points list based on their position in the grid.

    Sorts by F-order (column-wise)."""
    centroids = [
        ((r0 + r1) // 2, (c0 + c1) // 2, i)
        for i, (r0, r1, c0, c1) in enumerate(extremes)
    ]
    # determine bin_size from the maximum column value
    bin_size = (max((c for (r, c, i) in centroids)) // n_cols) + 1

    bins = [[] for _ in range(n_cols)]
    # add centroids to their respective bins
    for centroid in centroids:
        col = centroid[1]
        bin_idx = (col) // bin_size
        bins[bin_idx].append(centroid)

    # filter out possibly empty bins
    bins = [b for b in bins if len(b) > 0]
    # sort bins by row value
    for b in bins:
        b.sort(key=lambda b: b[0])
    # extracts each index
    indices = [b[2] for bin in bins for b in bin]
    # maps back the indices to each bounding box
    return {i: extremes[idx] for i, idx in enumerate(indices, 1)}


def filter_by_embryos(extremes, selected_embryos):
    """Filter the extremes dict, by only keeping the selected_embryos."""
    return {k: extremes[k] for k in selected_embryos if k in extremes}


def read_mmap(mmap_path, num_frames=None):
    offset, dtype, shape = get_metadata(mmap_path)
    if num_frames:
        shape = (num_frames, *shape[1:])
    return np.memmap(mmap_path, dtype, "r", offset, shape)


def create_tasks(extremes, frame_shape, pad, channels, active_ch, dest, overwrite):
    """"""
    tasks = []
    for id, extreme in extremes.items():
        x0, x1, y0, y1 = add_padding(extreme, frame_shape, pad)
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


def submit_tasks(img, tasks):
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
    pad=20,
    overwrite=False,
):
    """Extracts movies from ch1 and ch2, based on the boundaries passed for
    each item of `extremes`.

    Args:
        extremes: dict for `emb_number: [min_r, max_r, min_c, max_c]`.
        img_path: path to the raw image that will be cut.
        dest: directory where the movies will be saved.
        embryos: list of embryo numbers. Used to select a subgroup of embryos.
        active_ch: indicates the image active channel. Defaults to 1 and it
        is expected to be equal to 1 or 2.
        channels: (defaults to 2) number of channels imaged.
        pad: amount of padding to add to each movie, in pixels
        overwrite: boolean to determine if movies should be overwritten."""
    if embryos:
        extremes = filter_by_embryos(extremes, embryos)
    if active_ch not in [1, 2]:
        raise ValueError(f"Active channel should be 1 or 2, got {active_ch}.")
    if channels not in [1, 2]:
        raise ValueError(f"Can only parse 1 or 2 channels, but got {channels}")

    img = read_mmap(img_path)
    frame_shape = img.shape[2:]

    dest_path = Path(dest)
    tasks = create_tasks(
        extremes, frame_shape, pad, channels, active_ch, dest_path, overwrite
    )

    if len(tasks) == 0:
        return

    submit_tasks(img, tasks)


def output_file_name(id, ch, active_ch):
    if active_ch != 1 and active_ch != 2:
        raise ValueError(f"Active channel should be 1 or 2, got {active_ch}.")
    ch_number = ch + 1 if active_ch == 1 else active_ch - ch
    return f"emb{id}-ch{ch_number}.tif"


def save_movie(img, ch, x0, x1, y0, y1, output):
    movie = img[:, ch, x0:x1, y0:y1]
    imwrite(output, movie)


def add_padding(points, shape, pad=20):
    """Adds padding to the list of boundary points, pad//2 on each side."""
    p = pad // 2
    r, c = shape
    x0, x1, y0, y1 = points
    return [max(x0 - p, 0), min(x1 + p, r), max(y0 - p, 0), min(y1 + p, c)]


def boundary_to_rect_coords(boundary, shape):
    """Converts from `(x0, x1, y0, y1)` to `(x, y, w, h)`."""
    [x0, x1, y0, y1] = add_padding(boundary, shape)
    return [x0, y0, y1 - y0, x1 - x0]


def calculate_slice_coordinates(img_path, n_cols=3, thres_adjust=0):
    """Returns boundary points for all images in `img_path`.

    Args:
        img_path: absolute path to the raw data
        n_cols: number of columns in the FOV grid, used to enforce the naming
        convention of the extracted embryos.
    """
    binary_img = get_initial_binary_image(img_path, thres_adjust=thres_adjust)

    extremes = get_bbox_boundaries(binary_img, s=25, n_cols=n_cols)
    return extremes


def get_initial_frames_from_mmap(img_path, n=10):
    """Returns the first n frames from the file at `img_path`."""
    return read_mmap(img_path, num_frames=n)


def get_first_image_from_mmap(img_path):
    """Returns the first image from a mmap file, for plotting.

    The image is the average of the first 10 slices for channel 2.
    It is also equalized, since this method is supposed to be used for
    displaying the image."""
    img = get_initial_frames_from_mmap(img_path, n=10)
    first_frame = np.average(img[:, 1, :, :], axis=0)
    return equalize_hist(first_frame)


def get_initial_binary_image(img_path, n=10, thres_adjust=0):
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
