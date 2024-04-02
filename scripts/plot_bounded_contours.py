import os
from tifffile import imread

from pasnascope.animations import custom_animation
from pasnascope import roi, iellipsis


img_dir = os.path.join(os.getcwd(), 'data', 'embryos')

# All structural channel movies end with the suffix ch2
actives = [f for f in sorted(os.listdir(img_dir)) if f.endswith('ch1.tif')]
structs = [f for f in sorted(os.listdir(img_dir)) if f.endswith('ch2.tif')]

print('Select movie to display, based on index:')

for i, file in enumerate(structs):
    file_name = file.split('-')[0]
    print(f'[{i}] {file_name}')

idx = int(input())

print('Select embryo orientation [v or l]:')

orientation = input()

print('Select channel: 1 for active channel, 2 for structural channel.')

ch = int(input())

print('Select downsample amount to calculate the contours:')

window = int(input())

struct = imread(os.path.join(img_dir, structs[idx]))

if ch == 1:
    img = imread(os.path.join(img_dir, actives[idx]))
else:
    img = struct

if orientation == 'l':
    ell = iellipsis.fit_ellipsis(img)
    initial_mask = iellipsis.create_mask_from_ellipse(ell, img[0].shape)
elif orientation == 'v':
    initial_mask = roi.get_initial_mask(img, 100)
bounding_contour = roi.get_contour(initial_mask)
contours = roi.get_contours(struct, window=window,
                            mask=initial_mask, orientation=orientation)

ca = custom_animation.BoundedContourAnimation(img, contours, bounding_contour)
ca.display()
