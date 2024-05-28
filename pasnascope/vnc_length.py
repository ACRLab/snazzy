import csv

import numpy as np

from skimage.measure import find_contours, regionprops
from scipy.spatial.distance import pdist

from pasnascope import centerline, roi, utils


def measure_VNC(masks):
    '''Returns the max feret diameter for a group of binary images.

    The VNC length is measured indirectly, based on the ROI mask length.

    Args:
        masks (nparray): a mask of a group of images, as a 3D nparray
    '''
    vnc_lengths = np.zeros(masks.shape[0])

    for i, mask in enumerate(masks):
        # Values that should be ignored in the mask are masked as True
        # So, we need to flip the values to use here:
        regions = regionprops(np.logical_not(mask).astype(np.uint8))
        # TODO: handle cases where no region is found
        # this can happen because sometimes mask is None
        if len(regions) == 0:
            continue
        props = regions[0]
        vnc_lengths[i] = props['feret_diameter_max']

    return vnc_lengths


def predict_next(previous):
    '''Estimates the next point by linear regression.'''
    y = np.array(previous)
    x = np.arange(y.shape[0])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m*len(x) + c


def measure_VNC_centerline(image, pixel_width=1.62, thres_rel=0.6, min_dist=5):
    '''Calculates the centerline distance for a 3D image.'''
    vnc_lengths = np.zeros(image.shape[0])
    for i, img in enumerate(image):
        bin_img = centerline.binarize(img)
        dist = centerline.centerline_dist(
            bin_img, pixel_width=pixel_width, thres_rel=thres_rel, min_dist=min_dist)
        if dist:
            vnc_lengths[i] = dist
        else:
            # Leave the dist as 0 and predict it once all values are calculated
            pass
    for i, curr in enumerate(vnc_lengths[10:], 10):
        prev = vnc_lengths[i-1]
        diff = abs((curr-prev)/prev)
        if diff > 0.09:
            vnc_lengths[i] = predict_next(vnc_lengths[i-10:i-1])

    return vnc_lengths


def get_length_from_csv(file_path, columns=(6,), end=None, in_pixels=False, pixel_width=1.62):
    '''Reads CSV data as a nparray.

    Expects the lengths to be in actual metric units, instead of pixels.'''
    data = np.genfromtxt(
        file_path, delimiter=',', skip_header=1, usecols=columns)
    lengths = data
    if in_pixels:
        lengths *= pixel_width
    if end is None:
        return lengths
    else:
        return lengths[:end]


def match_names(annotated, name_LUT):
    '''Gets the corresponding movie name for a list of annotated data, based 
    on the mapping passed in `name_LUT`.'''
    embs = []
    for a in annotated:
        a_idx = int(a.stem.split('-')[0][3:])
        e_idx = name_LUT.get(a_idx, None)
        if not e_idx:
            continue
        embs.append(f"emb{e_idx}-ch2.tif")
    return embs


def export_csv(embryos, vnc_lengths, output, downsampling, frame_interval=6):
    '''Generates a csv file with VNC Length data.

    Parameters:
        embryos: list of embryo names
        vnc_lengts: list of lists, where each nested list represents VNC lengths for a single embryo. Must have same length as the `embryos`.
        output: path to the output csv file.
        downsampling: interval used to calculate VNC lengths.
        frame_interval: time (seconds) between two image captures.
    '''
    header = ['Time[h:m:s]', 'Embryo_Name', 'ROI ID', 'VNC_Length']
    if output.exists():
        print(
            f"Warning: The file `{output.stem}` already exists. Select another file name or delete the original file.")
        return False
    with open(output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for id, (embryo, lengths) in enumerate(zip(embryos, vnc_lengths), 1):
            for t, length in enumerate(lengths):
                writer.writerow(format_csv_row(t, downsampling,
                                frame_interval, embryo.stem, id, length))
    return True


def format_csv_row(t, downsampling, frame_interval, embryo, id, length):
    '''Columns in the csv file: [Time(HH:mm:ss), emb_name, id, length].'''
    return [utils.format_seconds((t*downsampling)*frame_interval), embryo, id, f"{length:.2f}"]


def fit_regression(lengths):
    x = np.arange(lengths.size)
    coef = np.polyfit(x, lengths, deg=1)
    poly1d = np.poly1d(coef)

    return poly1d


def get_convex_hulls(rois, frame=None):
    '''Gets the convex hull for each ROI frame.

    Uses `skimage.measure.regionprops`. Centers the hull to a matrix that has
    the same shape as the original ROI.'''
    imgs, y, x = rois.shape
    hulls = np.zeros_like(rois)
    if frame is not None:
        rois = rois[frame:frame+1]
        imgs = 1
    for i in range(imgs):
        roi = rois[i]
        regions = regionprops(np.logical_not(roi).astype(np.uint8))
        if len(regions) == 0:
            continue
        props = regions[0]
        hull = props['image_convex']
        hull_y, hull_x = hull.shape
        y_pad = (y - hull_y)//2
        x_pad = (x - hull_x)//2
        hulls[i, y_pad:y_pad+hull_y, x_pad:x_pad+hull_x] = hull

    return np.array(hulls)


def feret_diameter_max(hulls):
    '''Gets the maximum Feret diameter for each convex hull binary image.'''
    points = []
    for hull in hulls:
        identity_hull = np.pad(hull, 2, mode='constant', constant_values=0)
        coordinates = np.vstack(find_contours(identity_hull, .5,
                                              fully_connected='high'))
        k, v = get_pair_of_maximum_distance(coordinates)
        points.append((coordinates[k], coordinates[v]))
    return points


def condensed_to_pair_indices(n, k):
    '''Converts from pdist indices back to coordinate indices.'''
    x = n-(4.*n**2-4*n-8*k+1)**.5/2-.5
    i = x.astype(int)
    j = k+i*(i+3-2*n)/2+1
    return i, j.astype(int)


def get_pair_of_maximum_distance(hulls):
    '''Get the two convex hull points that are farthest apart.'''
    distances = pdist(hulls, 'sqeuclidean')
    max_dist_index = np.argmax(distances)
    return condensed_to_pair_indices(hulls.shape[0], max_dist_index)


def get_feret_diams(image, mask=None):
    '''Calculate maximum feret diameter for each frame of the image.'''
    img_roi = roi.get_roi(image, mask=mask, window=1)
    hull = get_convex_hulls(img_roi)
    return feret_diameter_max(hull)
