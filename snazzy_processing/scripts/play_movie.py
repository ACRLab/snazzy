from pathlib import Path

from tifffile import imread

from snazzy_processing import utils
from snazzy_processing.animations import custom_animation
from snazzy_analysis import Dataset, Group, FrequencyAnalysis, myplots

# group = "vacht"
# dataset_name = "20250501vacht1-df"

# group = "elavgal4ctl"
# dataset_name = "20250724_attPCtl-Df-ctl elavG4 UAS G6stdTom on 3rd"

group = "vgat"
dataset_name = "20241011_vgatdf"

# group = "Hdc_JK"
# dataset_name = "20250519HdcDf"

# group = "vgatvglut"
# dataset_name = "20250508_VgatVglutMutant"

movie_idx = 1
emb_idx = 2
ch = 1

movie_dir = Path(f"/Volumes/Extreme Pro/NT_fig2_raw/_rep_movies")
dataset_path = dataset_path = Path(f"/Volumes/Extreme Pro/NT_fig2_raw/{group}").joinpath(dataset_name)

img_dir = movie_dir.joinpath(dataset_name, "embs")
active = sorted(img_dir.glob("*ch1.tif"), key=utils.emb_number)

dataset = Dataset(dataset_path)
embryos = list(dataset.embryos)
emb = embryos[emb_idx]

img = imread(active[movie_idx])

start = emb.trace.aligned_offset
stop = emb.trace.trim_idx

print("\n\nMOVIE PATH", active[movie_idx])
print("EMBRYO NAME:", emb.name, "\n\n")

pa = custom_animation.PauseAnimation(f"{dataset.name} {emb.name}", img, emb, start=start, stop=stop, interval=2)
# pa.display()
pa.save(f"ani_{dataset.name}_{emb.name}_{start}_{stop}")
