import os
from tifffile import imread
from skimage.transform import downscale_local_mean
import numpy as np

from pasnascope.animations import custom_animation


def pre_process(img, downscale_factors):
    '''Remove edges to avoid padding the image with zeros.'''
    if img.ndim != 3:
        raise ValueError('img must have 3 dimensions.')
    nr, nc = get_matching_size(img.shape[1:], downscale_factors)

    trim_img = img[:, :nr, :nc]

    return downscale_local_mean(trim_img, downscale_factors)


def get_matching_size(shape, factors):
    '''Return the maximum value for shape so that each dimension is divisible by the corresponding index of factors.

    Used so that `downscale_local_mean` will receive an array that perfectly matches the `downscale_factors`.'''
    r, c = shape
    trim_rows = r % factors[0]
    trim_cols = c % factors[1]

    return (r-trim_rows, c-trim_cols)


def downsample_all(data_path, save=True):
    '''Reduces movies to lower resolution, to be classified.'''
    img_dir = os.path.join(data_path, 'embryos')
    output_dir = os.path.join(data_path, 'downsampled')
    file_names = [f for f in os.listdir(img_dir) if f.endswith('ch2.tif')]

    for file_name in file_names:
        file_path = os.path.join(img_dir, file_name)
        img = imread(file_path)

        downsampled = pre_process(img, (1, 2, 2))

        if save:
            output_name = f"ds-{file_name[:-4]}.npy"
            np.save(os.path.join(output_dir, output_name), downsampled)
            print(f"Downsampled file {file_path}.")


def display(file_path):
    '''Helper to visualize downsampled movies.'''
    img = np.load(os.path.join(file_path))
    pa = custom_animation.PauseAnimation(img, interval=25)
    pa.display()


if __name__ == '__main__':
    output_dir = os.path.join(os.getcwd(), 'data', 'downsampled')
    file_names = [f for f in os.listdir(output_dir)]
    print(file_names)
    display(os.path.join(output_dir, file_names[1]))
