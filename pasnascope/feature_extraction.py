import os
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


def extract(file_name, n_slices=300):
    '''Extracts features from the first n_slices of an image.

    Image is downsampled and represented as a numpy array.'''
    img_dir = os.path.join(os.getcwd(), 'data', 'downsampled')
    output_dir = os.path.join(img_dir, 'features')
    img = np.load(os.path.join(img_dir, file_name))

    features = []
    for i, slc in enumerate(img[:n_slices]):
        if i % 100 == 0:
            print('.' * (i//100))
        features.append(extract_features(slc))

    new_file_name = f"feat-{file_name.split('-')[1]}"
    np.save(os.path.join(output_dir, new_file_name), features)
    print(f"Saved features for {new_file_name}")


def extract_all(path):
    '''Extracts features of all downsampled files in a given directory.

    Relies on the fact that downsampled files start with `ds`.'''
    filenames = [f for f in os.listdir(path) if f.startswith('ds')]

    for filename in filenames:
        extract(filename, n_slices=300)
