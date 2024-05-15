import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pasnascope import centerline_errors


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
    with open(file_path, 'a+') as f:
        f.write(f"Parameters used in the grid search: {x_name}, {y_name}.\n")
        f.write(f"Values for {x_name}:\n{x_list}\n")
        f.write(f"Values for {y_name}:\n{y_list}\n")
        f.write('\n')


def search(x_list, x_name, y_list, y_name, estimator, should_save=True, file_path=None, **kwargs):
    '''Applies a grid search to the estimator function.

    Expects `estimator` to return a dict that can be used to assess the 
    estimator performance.'''
    if should_save and not file_path:
        print('File path required to save data.')
        return
    if should_save:
        write_header(x_list, x_name, y_list, y_name, file_path)
    for x in x_list:
        for y in y_list:
            print(f"Rel thres: {x}, min dist: {y}")
            extra_params = {x_name: x, y_name: y}
            kwargs.update(extra_params)
            estimation = estimator(**kwargs)
            if should_save:
                with open(file_path, 'a+') as f:
                    f.write(f"{x},{y}\n")
                    for k, v in estimation.items():
                        f.write(f"{k}: {v}\n")
                    f.write('\n')


def parse_grid_search_output(x_list, y_list, file_path):
    with open(file_path, 'r') as f:
        lines = [l.rstrip() for l in f.readlines()]
        avg_errors = []
        errors = []
        for l in lines[2:]:
            if l == '' and len(errors) > 0:
                avg_errors.append(sum(errors)/len(errors))
                errors = []
            else:
                if l.startswith('emb'):
                    errors.append(float(l.split(' ')[1][1:-1])*100)
        if len(errors) > 0:
            avg_errors.append(sum(errors)/len(errors))

    grid = np.array(avg_errors).reshape((len(x_list), len(y_list)))

    fig, ax = plt.subplots()
    im, cbar = heatmap(np.array(grid), x_list, y_list, ax=ax,
                       cmap='viridis_r', cbarlabel='%Err - thres_rel x min_distance')
    texts = annotate_heatmap(im, grid)

    fig.tight_layout()
    plt.show()


def get_emb_files(exp_dir, num_samples):
    annotated_dir = exp_dir.joinpath('annotated')
    annotated_files = centerline_errors.get_random_files(
        annotated_dir, n=num_samples)

    emb_names = [exp_dir/'embs'/f"{f.stem}.tif" for f in annotated_files]
    return emb_names
