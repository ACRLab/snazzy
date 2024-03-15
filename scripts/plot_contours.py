import os
from tifffile import imread

from pasnascope.animations import custom_animation
from pasnascope import roi


img_dir = os.path.join(os.getcwd(), 'data', 'embryos')

# All structural channel movies end with the suffix ch2
actives = [f for f in sorted(os.listdir(img_dir)) if f.endswith('ch1.tif')]
structs = [f for f in sorted(os.listdir(img_dir)) if f.endswith('ch2.tif')]

print('Select movie to display, based on index:')

for i, file in enumerate(actives):
    file_name = file.split('-')[0]
    print(f'[{i}] {file_name}')

idx = int(input())

print('Select channel: 1 for active channel, 2 for structural channel.')

ch = int(input())

print('Select downsample amount to calculate the contours:')

window = int(input())

struct = imread(os.path.join(img_dir, structs[idx]), key=range(0, 1000))

if ch == 1:
    img = imread(os.path.join(img_dir, actives[idx]), key=range(0, 1000))
else:
    img = struct

contours = roi.get_contours(struct, window=window)
ca = custom_animation.ContourAnimation(img, contours)
ca.display()
