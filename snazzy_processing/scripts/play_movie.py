from pathlib import Path

from tifffile import imread

from snazzy_processing import utils
from snazzy_processing.animations import custom_animation


data_dir = Path("./data")
datasets = [f.stem for f in data_dir.iterdir() if f.is_dir()]

print("Enter dataset name, based on index:")
for i, file in enumerate(datasets):
    print(f"[{i}] {file}")

e = int(input())
dataset = datasets[e]

img_dir = data_dir.joinpath(dataset, "embs")

# All structural channel movies end with the suffix ch2
structs = sorted(img_dir.glob("*ch2.tif"), key=utils.emb_number)
active = sorted(img_dir.glob("*ch1.tif"), key=utils.emb_number)

print("Select movie to display, based on index:")

for i, file in enumerate(active):
    print(f"[{i}] {file.stem.split('-')[0]}")

idx = int(input())

print("Select channel: 1 for active channel, 2 for structural channel.")

ch = int(input())

if ch != 1 and ch != 2:
    exit(1)

img = imread(structs[idx]) if ch == 2 else imread(active[idx])
pa = custom_animation.PauseAnimation(img, interval=25)
pa.display()
