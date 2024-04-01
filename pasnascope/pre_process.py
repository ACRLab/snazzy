import os
from tifffile import imread
from skimage.transform import downscale_local_mean
import numpy as np


def pre_process(img, downscale_factors=(1, 8, 8)):
    # remove edges to avoid padding the image with zeros
    trim_rows = img.shape[1] % downscale_factors[1]
    trim_cols = img.shape[2] % downscale_factors[2]

    if trim_cols == 0:
        trim_cols = img.shape[2]
    else:
        trim_cols *= -1

    if trim_rows == 0:
        trim_rows = img.shape[1]
    else:
        trim_rows *= -1

    trim_img = img[:, :trim_rows, :trim_cols]

    return downscale_local_mean(trim_img, (1, 8, 8))


def downsample_all():
    '''Reduces individual movies to lower resolution, to be classified.'''
    img_dir = os.path.join(os.getcwd(), 'data', 'embryos')
    output_dir = os.path.join(os.getcwd(), 'results', 'cache', 'downsampled')
    file_names = [f for f in os.listdir(img_dir) if f.endswith('ch2.tif')]

    for file_name in file_names:
        file_path = os.path.join(img_dir, file_name)
        img = imread(file_path)

        downsampled = pre_process(img, (1, 8, 8))

        output_name = f"ds-{file_name[:-4]}.npy"
        np.save(os.path.join(output_dir, output_name), downsampled)
        print(f"Downsampled file {file_path}.")
