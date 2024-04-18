import os
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes
from skimage.filters import threshold_multiotsu
import numpy as np


def get_largest_label(img):
    '''Returns the largest connected binary image.

    Uses a lower threshold, to enforce shape format.'''
    slc = img.copy()
    thres = threshold_multiotsu(slc)
    binary_mask = slc > thres[0]

    slc[...] = 0
    slc[binary_mask] = 1

    slc = slc.astype(np.bool_)
    remove_small_holes(slc, 200, out=slc)

    labels = label(slc, connectivity=2)
    largest_label = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    return largest_label.astype(np.uint16)*255


def extract_features(img):
    '''Binarizes image and extracts features for the largest label.

    Extracted features: r_centroid, c_centroid, hu_moment[0], hu_moment[1].'''
    label_img = get_largest_label(img)
    r = regionprops(label_img)[0]

    centroid_local = r['centroid_local']
    hu_moments = r['moments_hu'][:2]

    features = np.concatenate((centroid_local, hu_moments))
    return features


def extract(file_path, n_slices, save=False, output=None):
    '''Extracts features from the first `n_slices` of an image.

    Image is downsampled and represented as a numpy array.'''
    p = Path(file_path)
    img = np.load(file_path)[:n_slices]

    features = []
    for slc in img:
        features.append(extract_features(slc))

    new_file_name = f"feat-{p.stem.split('-')[1]}"
    if save and output:
        np.save(os.path.join(output, new_file_name), features)
        print(f"Saved features as {new_file_name}.")
    elif output is None:
        print("Files were not saved. An output path is required.")
    else:
        return features


def extract_all(path, n_slices=300, save=False, output=None):
    '''Extracts features of all downsampled files in a given directory.

    Assumes that downsampled files start with `ds`.'''
    filenames = [f for f in os.listdir(path) if f.startswith('ds')]

    for filename in filenames:
        filepath = os.path.join(path, filename)
        extract(filepath, n_slices, save=save, output=output)
