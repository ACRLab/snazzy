import os
from tifffile import imread

from pasnascope.animations import custom_animation


base_dir = "/home/cdp58/Documents/repos/pasnascope_analysis/data/embryos/"

# All structural channel movies end with the suffix ch2
active = [f for f in sorted(os.listdir(base_dir)) if f.endswith('ch1.tif')]
struct = [f for f in sorted(os.listdir(base_dir)) if f.endswith('ch2.tif')]

print('Select movie to display, based on index:')

for i, file in enumerate(active):
    file_name = file.split('-')[0]
    print(f'[{i}] {file_name}')

idx = int(input())

print('Select channel: 1 for active channel, 2 for structural channel.')

ch = int(input())

if ch == 1:
    file_name = active[idx]
elif ch == 2:
    file_name = struct[idx]
else:
    exit(1)

img = imread(base_dir + file_name, key=range(0, 1000))
pa = custom_animation.PauseAnimation(img, interval=50)
pa.display()
