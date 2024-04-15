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
    img = np.load(file_path)

    features = []
    for slc in img[:n_slices]:
        features.append(extract_features(slc))

    new_file_name = f"feat-{p.stem.split('-')[1]}"
    if save and output:
        np.save(os.path.join(output, new_file_name), features)
    if output is None:
        print("Files were not saved. An output path is required.")
    print(f"Saved features as {new_file_name}.")


def extract_all(path, n_slices=300, save=False, output=None):
    '''Extracts features of all downsampled files in a given directory.

    Assumes that downsampled files start with `ds`.'''
    filenames = [f for f in os.listdir(path) if f.startswith('ds')]

    for filename in filenames:
        extract(os.path.join(path, filename),
                n_slices, save=save, output=output)
