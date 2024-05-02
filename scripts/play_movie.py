from pathlib import Path

from tifffile import imread

from pasnascope import utils
from pasnascope.animations import custom_animation


data_dir = Path('./data')
experiments = [f.stem for f in data_dir.iterdir() if f.is_dir()]

print('Enter experiment name, based on index:')
for i, file in enumerate(experiments):
    print(f'[{i}] {file}')

e = int(input())
experiment = experiments[e]

img_dir = data_dir.joinpath(experiment, 'embs')

# All structural channel movies end with the suffix ch2
structs = sorted(img_dir.glob('*ch2.tif'), key=utils.sort_by_emb_name)
active = sorted(img_dir.glob('*ch1.tif'), key=utils.sort_by_emb_name)

print('Select movie to display, based on index:')

for i, file in enumerate(active):
    print(f'[{i}] {file.stem}')

idx = int(input())

print('Select channel: 1 for active channel, 2 for structural channel.')

ch = int(input())

if ch != 1 and ch != 2:
    exit(1)

img = imread(structs[i]) if ch == 2 else imread(active[i])
pa = custom_animation.PauseAnimation(img, interval=25)
pa.display()
