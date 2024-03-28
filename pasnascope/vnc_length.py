import numpy as np

from skimage.measure import find_contours, regionprops
from scipy.spatial.distance import pdist

from pasnascope import roi


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
