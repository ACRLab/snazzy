import os
import numpy as np
from tifffile import imread

from pasnascope import roi, vnc_length
from pasnascope.animations import custom_animation


img_dir = os.path.join(os.getcwd(), 'data', 'embryos')
structs = [f for f in sorted(os.listdir(img_dir)) if f.endswith('ch2.tif')]

print('Select movie to display, based on index:')

for i, file in enumerate(structs):
    file_name = file.split('-')[0]
    print(f'[{i}] {file_name}')

idx = int(input())

img = imread(os.path.join(img_dir, structs[idx]))

print('Do you want to add an initial mask? [y/n]')

mask_opt = input()

if mask_opt.lower() == 'y':
    first_mask = roi.get_initial_mask(img, 200)
elif mask_opt.lower() == 'n':
    first_mask = None
else:
    print('Invalid option')
    exit(1)

print('Calculating ROIs...')
img_roi = roi.get_roi(img, mask=first_mask, window=1)
print('Calculating convex hull shapes...')
hulls = vnc_length.get_convex_hulls(img_roi, frame=None)
hulls = hulls.astype(np.uint16)*700

movie = np.hstack((img, hulls))

ani = custom_animation.PauseAnimation(movie, interval=50)

ani.display()
