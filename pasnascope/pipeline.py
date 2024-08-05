from datetime import datetime
import shutil

from tifffile import imread

from pasnascope import activity, find_hatching, full_embryo_length, roi, utils, vnc_length


def measure_vnc_length(embs_src, res_dir, interval):
    '''Calculates VNC length for all embryos in a directory.'''
    embs = sorted(embs_src.glob('*ch2.tif'), key=utils.emb_number)
    output = res_dir.joinpath('lengths')
    output.mkdir(parents=True, exist_ok=True)
    lengths = []
    ids = []
    for emb in embs:
        id = utils.emb_number(emb.stem)
        if output.joinpath(f'emb{id}.csv').exists():
            print(f'File emb{id}.csv already exists. Skipping..')
            continue
        hp = find_hatching.find_hatching_point(emb)
        hp -= hp % interval

        img = imread(emb, key=range(0, hp, interval))
        vnc_len = vnc_length.measure_VNC_centerline(img)
        lengths.append(vnc_len)
        ids.append(id)

    vnc_length.export_csv(ids, lengths, output, interval)
    return len(ids)


def measure_embryo_full_length(embs_src, res_dir):
    embs = sorted(embs_src.glob('*ch2.tif'), key=utils.emb_number)
    output = res_dir.joinpath('full-length.csv')
    full_lengths = []
    embryo_names = []
    if output.exists():
        print(
            f"The file {output.stem} already exists, and won't be overwritten.")
        return 0

    for emb in embs:
        embryo_names.append(emb.stem)
        full_lengths.append(full_embryo_length.measure(emb))

    full_embryo_length.export_csv(full_lengths, embryo_names, output)
    return len(full_lengths)


def calc_activity(embs_src, res_dir, window):
    '''Calculate activity for active and structural channels'''
    active = sorted(embs_src.glob('*ch1.tif'), key=utils.emb_number)
    struct = sorted(embs_src.glob('*ch2.tif'), key=utils.emb_number)

    output = res_dir.joinpath('results', 'activity')
    output.mkdir(parents=True, exist_ok=True)

    embryos = []
    ids = []
    for act, stct in zip(active, struct):
        id = utils.emb_number(act)
        file_path = output.joinpath(f'emb{id}.csv')
        if file_path.exists():
            print(f'File {file_path.stem} already exists. Skipping..')
            continue
        active_img = imread(act)
        struct_img = imread(stct)
        mask = roi.get_roi(struct_img, window=window)

        masked_active = activity.apply_mask(active_img, mask)
        masked_struct = activity.apply_mask(struct_img, mask)

        signal_active = activity.get_activity(masked_active)
        signal_struct = activity.get_activity(masked_struct)

        emb = [signal_active, signal_struct]

        embryos.append(emb)
        ids.append(utils.emb_number(act))

    activity.export_csv(ids, embryos, output)
    return len(ids)


def clean_up_files(embs_src, first_frames_path):
    shutil.rmtree(embs_src)
    first_frames_path.unlink(missing_ok=True)


def log_params(path, **kwargs):
    with open(path, '+a') as f:
        f.write("Starting a new analysis...\n")
        f.write(f"{datetime.now()}\n")
        for name, value in kwargs.items():
            f.write(f"{name}: {value}\n")
        f.write("="*79 + "\n")
