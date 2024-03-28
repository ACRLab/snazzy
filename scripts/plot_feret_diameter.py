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

embryo = file_name[:-4]
try:
    max_ferets = np.load(f'./results/cache/feret/{embryo}.npy')
except OSError:
    max_ferets = vnc_length.get_feret_diams(img, mask=None)
    np.save(f'./results/cache/feret/{embryo}.npy', max_ferets)

ani = custom_animation.FeretAnimation(img, max_ferets)
ani.display()
