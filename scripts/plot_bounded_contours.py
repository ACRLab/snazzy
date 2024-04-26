import os
from pathlib import Path
from tifffile import imread

from pasnascope.animations import custom_animation
from pasnascope import roi, projs

data_dir = Path('./data')
experiments = [f.stem for f in data_dir.iterdir() if f.is_dir()]

print('Enter experiment name, based on index:')
for i, file in enumerate(experiments):
    print(f'[{i}] {file}')

e = int(input())
experiment = experiments[e]

img_dir = os.path.join(os.getcwd(), 'data', experiment, 'embs')

# All structural channel movies end with the suffix ch2
structs = [f for f in sorted(os.listdir(img_dir)) if f.endswith('ch2.tif')]

print('Select movie to display, based on index:')

for i, file in enumerate(structs):
    file_name = file.split('-')[0]
    print(f'[{i}] {file_name}')

idx = int(input())

print('Select downsample amount to calculate the contours:')

window = int(input())

img = imread(os.path.join(img_dir, structs[idx]))

m = projs.proj_mask(img[0])
bounding_contour = roi.get_contour(m)
contours = roi.get_contours(img, window=window, mask=m)

ca = custom_animation.BoundedContourAnimation(
    img, contours, bounding_contour, interval=20)
ca.display()
