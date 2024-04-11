import os
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.filters import threshold_multiotsu
import numpy as np


def get_largest_label(img):
    slc = img.copy()
    thres = threshold_multiotsu(slc)
    binary_mask = slc > thres[0]

    slc[...] = 0
    slc[binary_mask] = 1

    labels = label(slc, connectivity=1)
    largest_label = labels == np.argmax(
        np.bincount(labels.flat)[1:])+1
    return largest_label*255


def extract_features(img):
    '''Binarizes image and extracts features for the largest label.

    Extracted features: r_centroid, c_centroid, hu_moment[0], hu_moment[1] and area.'''
    label_img = get_largest_label(img)
    r = regionprops(label_img)[0]

    centroid_local = r['centroid_local']
    hu_moments = r['moments_hu'][:2]
    area = r['area']

    features = np.concatenate((centroid_local, hu_moments, [area]))
    return features


def extract(file_path, n_slices, save=False):
    '''Extracts features from the first n_slices of an image.

    Image is downsampled and represented as a numpy array.'''
    p = Path(file_path)
    output_dir = os.path.join(p.parent, 'features')
    img = np.load(file_path)

    features = []
    for i, slc in enumerate(img[:n_slices]):
        features.append(extract_features(slc))

    new_file_name = f"feat-{p.stem.split('-')[1]}"
    if save:
        np.save(os.path.join(output_dir, new_file_name), features)
    print(f"Saved features as {new_file_name}.")


def extract_all(path, save=False, n_slices=300):
    '''Extracts features of all downsampled files in a given directory.

    Assumes that downsampled files start with `ds`.'''
    filenames = [f for f in os.listdir(path) if f.startswith('ds')]

    for filename in filenames:
        extract(os.path.join(path, filename), n_slices, save=save)
