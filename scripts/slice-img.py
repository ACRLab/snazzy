import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import tifffile
from skimage.filters import threshold_multiotsu
from skimage.morphology import binary_opening, octagon


def get_metadata(img_path):
    '''Returns image metadata, used to open it as a memory map.'''
    with tifffile.TiffFile(img_path) as tif:
        series = tif.series[0]
        shape = series.shape
        offset = series.dataoffset
        dtype = np.dtype(tif.byteorder + series.dtype.char)

    return offset, dtype, shape


def get_threshold(img):
    return threshold_multiotsu(img)[0]


def update_extremes(extremes, i, j):
    min_i, max_i, min_j, max_j = extremes
    if i < min_i:
        extremes[0] = i
    if i > max_i:
        extremes[1] = i
    if j < min_j:
        extremes[2] = j
    if j > max_j:
        extremes[3] = j


def within_boundaries(i, j, r, c):
    return i >= 0 and i < r and j >= 0 and j < c


def mark_neighbors(img, x, y, s):
    '''Expands from the search box and marks all points connected to any point
    within the search box.

    Returns:
        counter: amount of pixels marked as visited
        extremes: list with min and max pixel value for both dimensions
    '''
    neighbors = [(-1, 0), (0, -1),
                 (0, 1), (1, 0)]
    r, c = img.shape
    counter = 0
    # TODO: create a custom class CounterDeque
    dq = deque()

    for di in range(s):
        for dj in range(s):
            if img[x+di, y+dj] == 1:
                dq.append((x+di, y+dj))
    extremes = [r, 0, c, 0]  # min_i, max_i, min_j, max_j
    while len(dq) > 0:
        i, j = dq.pop()
        if img[i, j] == 1:
            update_extremes(extremes, i, j)
            counter += 1
            # 255 represents a visited pixel
            img[i, j] = 255
            for (ii, jj) in neighbors:
                if within_boundaries(i+ii, j+jj):
                    if img[i+ii, j+jj] == 1:
                        dq.append((i+ii, j+jj))
    return counter, extremes


def mark_bbox(img, s=25):
    markers = []
    extremes = []
    r, c = img.shape
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
                    markers.append((i, j))
                    extremes.append(extreme)
    return markers, extremes


def get_bbox_pos(extremes):
    '''Approximates the centroid by the extreme points. Centers the bbox at 
    these coordinates. '''
    w = 350
    h = 200
    coords = []
    for e in extremes:
        min_i, max_i, min_j, max_j = e
        i_centroid = (max_i - min_i)//2 + min_i
        j_centroid = (max_j - min_j)//2 + min_j
        coords.append((i_centroid - h//2, j_centroid - w//2))
    return coords


if __name__ == '__main__':
    img_path = '/home/cdp58/Documents/raw_images/20240220_GlueWGlass1.tif'
    offset, dtype, shape = get_metadata(img_path)
    img1 = np.memmap(img_path, dtype=dtype, mode='r',
                     shape=shape, offset=offset)

    frame = img1[1000:1005, 1, :, :].copy()
    frame = np.average(frame, axis=0)

    thres = get_threshold(frame)

    frame[frame < thres] = 0
    frame[frame >= thres] = 1
    opened = np.zeros_like(frame, dtype=np.uint8)
    binary_opening(frame, footprint=octagon(6, 6), out=opened)

    points, extremes = mark_bbox(opened, s=75)
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax1.imshow(frame)
    ax2.imshow(opened)
    ax2.plot(y, x, 'g.')
    recs = [Rectangle((y, x), 350, 200)
            for (x, y) in get_bbox_pos(extremes)]
    pc = PatchCollection(recs, color='red', alpha=0.7,
                         linewidth=1, facecolor='none')
    ax1.add_collection(pc)

    plt.show()
