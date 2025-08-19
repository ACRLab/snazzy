from tifffile import imread
import numpy as np


def find_hatching_point(img_path):
    """First draft on how to calculate hatching points.

    Returns:
        hp (int): last slice before hatching.
    """
    img = imread(img_path)
    sums = np.sum(img, axis=(1, 2))
    ratios = np.divide(sums[1::2], sums[:-1:2])
    # Assume that hatching occurs when the total pixel sum decreases by 5%
    # between 2 frames.
    for i, r in enumerate(ratios):
        if r < 0.95:
            return i * 2 - 100
    return i * 2 - 100
