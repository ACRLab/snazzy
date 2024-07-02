import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread

from pasnascope import centerline_errors, vnc_length


def heatmap(data, row_labels, col_labels, ax, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Adapted from: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data):
    '''
    Annotates the heatmap with the provided data.

    Adapted from: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    '''
    threshold = im.norm(data.max())/2.
    textcolors = ("black", "white")

    kw = dict(horizontalalignment="center", verticalalignment="center")
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def write_header(x_list, x_name, y_list, y_name, file_path):
    with open(file_path, 'w+') as f:
        f.write(f"Parameters used in the grid search: {x_name}, {y_name}.\n")
        f.write(f"Param: {x_name}\n{x_list}\n")
        f.write(f"Param: {y_name}\n{y_list}\n")
        f.write('\n')


def search(x_list, y_list, embryos, annotated, hatching_points=None, interval=20, num_samples=5):
    '''Applies a grid search to the estimator function.

    Expects `estimator` to return a dict that can be used to assess the
    estimator performance.'''
    errors = {(x, y,): [] for x in x_list for y in y_list}
    for emb in embryos[:num_samples]:
        print(emb.stem)
        hp = hatching_points[emb.stem]
        img = imread(emb, key=range(0, hp, interval))
        for x in x_list:
            for y in y_list:
                lengths = vnc_length.measure_VNC_centerline(
                    img, thres_rel=x, min_dist=y)
                error = centerline_errors.compare(
                    lengths, annotated[emb.stem])
                errors[(x, y,)].append([emb.stem, error])
    return errors


def write_grid(grid, file_path, x_list, x_name, y_list, y_name):
    write_header(x_list, x_name, y_list, y_name, file_path)
    with open(file_path, 'a+') as f:
        for (thres_rel, min_dist), embs in grid.items():
            f.write(f'{thres_rel}, {min_dist}\n')
            for e in embs:
                f.write(f'{e[0]}: {e[1]}\n')
            f.write('\n')


def parse_grid_search_output(file_path):
    with open(file_path, 'r') as f:
        lines = [l.rstrip() for l in f.readlines()]
        avg_errors = []
        errors = []
        # parse parameters
        x, x_values, y, y_values = parse_header(lines[1:5])
        for l in lines[6:]:
            # empty lines are the separators between parameter combinations
            if l == '' and len(errors) > 0:
                avg_errors.append(sum(errors)/len(errors))
                errors = []
            else:
                if l.startswith('emb'):
                    errors.append(float(l.split(' ')[1][1:-1])*100)
        if len(errors) > 0:
            avg_errors.append(sum(errors)/len(errors))

    grid = np.array(avg_errors).reshape((len(x_values), len(y_values)))

    fig, ax = plt.subplots()
    im, _ = heatmap(np.array(grid), x_values, y_values, ax=ax,
                    cmap='viridis_r', cbarlabel=f'%Err - {x} x {y}')
    annotate_heatmap(im, grid)
    fig.suptitle('Grid search')
    fig.tight_layout()
    plt.show()


def parse_header(lines):
    x = parse_header_variable(lines[0])
    x_values = parse_header_values(lines[1])
    y = parse_header_variable(lines[2])
    y_values = parse_header_values(lines[3])
    return x, x_values, y, y_values


def parse_header_variable(line):
    if not line.startswith('Param: '):
        raise ValueError(
            f'Grid search output is formatted incorrectly. Expected "Param: <param_name>", but got "{line}".')
    return line.split('Param: ')[1]


def parse_header_values(line):
    if not (line.startswith('[') and line.endswith(']')):
        raise ValueError(
            f'Grid search output is formatted incorrectly. Expected string to be read as an array, but got "{line}"')
    return line.strip('[]').split(',')


def get_emb_files(exp_dir, num_samples):
    annotated_dir = exp_dir.joinpath('annotated')
    annotated_files = centerline_errors.get_random_files(
        annotated_dir, n=num_samples)

    emb_names = [exp_dir/'embs'/f"{f.stem}.tif" for f in annotated_files]
    return emb_names
