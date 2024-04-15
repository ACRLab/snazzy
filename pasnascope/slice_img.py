import os
import numpy as np
from tifffile import imwrite, TiffFile
from skimage.filters import threshold_triangle
from skimage.morphology import octagon, binary_closing
from skimage.exposure import equalize_hist

from pasnascope import utils


def get_metadata(img_path):
    '''Returns image metadata, used to open it as a memory map.'''
    with TiffFile(img_path) as tif:
        series = tif.series[0]
        shape = series.shape
        offset = series.dataoffset
        dtype = np.dtype(tif.byteorder + series.dtype.char)

    return offset, dtype, shape


def get_threshold(img):
    '''Returns image threshold using the triangle method.'''
    return threshold_triangle(img)


def within_boundaries(i, j, r, c):
    '''Checks if `i` and `j` are valid coords in a `r`x`c` matrix.'''
    return i >= 0 and i < r and j >= 0 and j < c


def mark_neighbors(img, x, y, s):
    '''Expands from the search box and marks points connected to any point
    within the search box.

    Returns:
        counter: amount of pixels marked as visited
        extremes: list with min and max pixel value for both dimensions
    '''
    neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    r, c = img.shape
    dq = utils.CounterDeque(img.shape)

    for di in range(s):
        for dj in range(s):
            if within_boundaries(x+di, y+dj, r, c) and img[x+di, y+dj] == 1:
                dq.append((x+di, y+dj))
    while len(dq) > 0:
        i, j = dq.pop()
        if img[i, j] == 1:
            dq.update_extremes(i, j)
            # 255 represents a visited pixel
            img[i, j] = 255
            for (ii, jj) in neighbors:
                if within_boundaries(i+ii, j+jj, r, c):
                    if img[i+ii, j+jj] == 1:
                        dq.append((i+ii, j+jj))
    return dq.counter, dq.extremes


def get_bbox_boundaries(img, s=25, n_cols=3):
    '''Gets bbox boundaries (max and min values for both coordinates).'''
    extremes = []
    r, c = img.shape
    # iterate over slices of size sxs
    for i in range(0, r, s):
        for j in range(0, c, s):
            slc = img[i:i+s, j:j+s]
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


def sort_by_grid_pos(extremes, n_cols):
    '''Sorts each boundary points list based on their position in the grid.

    Sorts by F-order (column-wise).'''
    centroids = [((x0+x1)//2, (y0+y1)//2, i)
                 for i, (x0, x1, y0, y1) in enumerate(extremes)]
    bin_size = (max((y for (x, y, i) in centroids))//n_cols) + 1

    bins = [[] for _ in range(n_cols)]

    for centroid in centroids:
        y = centroid[1]
        bin_idx = (y)//bin_size
        bins[bin_idx].append(centroid)

    # filter out possibly empty bins
    bins = [b for b in bins if len(b) > 0]

    for b in bins:
        b.sort(key=lambda b: b[0])

    indices = [b[-1] for bin in bins for b in bin if len(bin) > 0]

    return [extremes[i] for i in indices]


def cut_movies(extremes, img_path, dest, pad=20):
    '''Extracts movies from ch1 and ch2, based on the boundaries passed for
    each element of `extremes`.

    Args:
        extremes: list of `[min_r, max_r, min_c, max_c]` points.
        img_path: path to the raw image that will be cut.
        dest: directory where the movies will be saved.'''
    offset, dtype, shape = get_metadata(img_path)
    img = np.memmap(img_path, dtype=dtype, mode='r',
                    shape=shape, offset=offset)
    for i, extreme in enumerate(extremes):
        x0, x1, y0, y1 = add_padding(extreme, pad)

        cut_ch1 = img[:, 0, x0:x1, y0:y1]
        cut_ch2 = img[:, 1, x0:x1, y0:y1]

        print(f"Processing emb{i}-ch1...")
        imwrite(os.path.join(dest, f'emb{i}-ch1.tif'), cut_ch1)
        print(f"Processing emb{i}-ch2...")
        imwrite(os.path.join(dest, f'emb{i}-ch2.tif'), cut_ch2)


def add_padding(points, pad=20):
    '''Adds padding to the list of boundary points, pad//2 on each side.'''
    p = pad//2
    x0, x1, y0, y1 = points
    return [x0-p, x1+p, y0-p, y1+p]


def boundary_to_rect_coords(boundary):
    '''Converts from `(x0, x1, y0, y1)` to `(x, y, w, h)`.'''
    [x0, x1, y0, y1] = add_padding(boundary)
    return [x0, y0, y1-y0, x1-x0]


def calculate_slice_coordinates(img_path, n_cols=3):
    '''Returns boundary points for all images in `img_path`.

    Args:
        img_path: absolute path to the raw data
        n_cols: number of columns in the FOV grid, used to enforce the naming
        convention of the extracted embryos.
    '''
    binary_img = get_initial_binary_image(img_path)

    extremes = get_bbox_boundaries(binary_img, s=25, n_cols=n_cols)
    return extremes


def get_initial_frames_from_mmap(img_path, n=10):
    '''Returns the first n frames from the file at `img_path`.'''
    offset, dtype, shape = get_metadata(img_path)
    # Change first dimension to load just the 10 first images
    shape = (n, *shape[1:])
    img = np.memmap(img_path, dtype=dtype, mode='r',
                    shape=shape, offset=offset)
    return img


def get_first_image_from_mmap(img_path):
    '''Returns the first image from a mmap file, for plotting.

    The image is the average of the first 10 slices.
    It is also equalized, since this method is supposed to be used for 
    displaying the image.'''
    img = get_initial_frames_from_mmap(img_path, n=10)
    first_frame = np.average(img[:, 1, :, :], axis=0)
    return equalize_hist(first_frame)


def get_initial_binary_image(img_path, n=10):
    '''Binarizes the first `n` slices of the img, which is read as a mmap.'''
    img = get_initial_frames_from_mmap(img_path, n=n)

    frame = img[:, 1, :, :].copy()
    frame = np.max(frame, axis=0)

    thres = get_threshold(frame)
    frame[frame < thres] = 0
    frame[frame >= thres] = 1
    opened = np.zeros_like(frame, dtype=np.uint8)
    binary_closing(frame, footprint=octagon(5, 5), out=opened)
    return opened
