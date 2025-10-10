from pathlib import Path

from tifffile import imread

from snazzy_processing.animations import custom_animation
from snazzy_processing import roi, utils

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

for i, file in enumerate(structs):
    print(f"[{i}] {file.stem.split('-')[0]}")

idx = int(input())

print("Select channel, based on index:")

print("[1]: Active channel")
print("[2]: Structural channel")

ch = int(input())

if ch != 1 and ch != 2:
    exit(1)

print("Select downsample amount to calculate the contours:")

window = int(input())

img = imread(structs[idx]) if ch == 2 else imread(active[idx])
struct_img = img if ch == 2 else imread(structs[idx])
contours = roi.get_contours(struct_img, window=window)

ca = custom_animation.ContourAnimation(img, contours, window, 1)
ca.display()
