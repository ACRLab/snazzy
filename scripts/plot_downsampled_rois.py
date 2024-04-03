import os
import numpy as np

from pasnascope.animations import custom_animation
from pasnascope import feature_extraction


img_dir = os.path.join(os.getcwd(), 'data', 'downsampled')

# All structural channel movies end with the suffix ch2
structs = [f for f in sorted(os.listdir(img_dir)) if f.startswith('ds')]

print('Select movie to display, based on index:')

for i, file in enumerate(structs):
    file_name = file.split('-')[1]
    print(f'[{i}] {file_name}')

idx = int(input())

struct = np.load(os.path.join(img_dir, structs[idx]))

img_roi = np.zeros_like(struct)
for i, img in enumerate(struct):
    img_roi[i] = feature_extraction.get_largest_label(img)

pa = custom_animation.PauseAnimation(img_roi)
pa.display()
