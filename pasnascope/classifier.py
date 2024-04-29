import numpy as np
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tifffile import imread
import pickle

from pasnascope import pre_process, feature_extraction


def get_training_samples(orientation, samples_dir, exp_data, n=500):
    '''Returns annotated data, based on a given orientation.

    Picks a proporcional amount of images from each sample.
    Args:
        orientation: `v` (ventral) or `l` (lateral). Embryo orientation.
        samples_dir: path where the feature files are saved.
        exp_data: annotated data. See `fit_SVC` for details.
        n: int. Amount of samples.
    '''
    lengths = {}
    for name, intervals in exp_data[orientation].items():
        lengths[name] = sum(end-start for start, end in intervals)

    total_length = sum(lengths.values())

    sample_sizes = {name: int((l/total_length)*n)
                    for name, l in lengths.items()}

    X = np.zeros((n, exp_data['number_of_features']))
    i = 0
    for sample, intervals in exp_data[orientation].items():
        file_name = f"feat-{sample}.npy"
        curr = np.load(os.path.join(samples_dir, file_name))
        size = sample_sizes[sample]
        for start, end in intervals:
            if size == 0:
                break
            l = end - start
            if l < size:
                X[i:i+l] = curr[start:end]
                size -= l
            else:
                end = start + size
                X[i:i+size] = curr[start:end]
                size = 0
            i += end - start
    return X


def pre_process_tiff(file_path):
    img = imread(file_path, key=range(10))
    # The downscale_factors should match the ones used to fit the model
    downsampled = pre_process.pre_process(img, (1, 2, 2))
    downsampled = np.average(downsampled, axis=0)
    return downsampled


def get_features_from_tiff(file_path):
    '''Extracts features from a tiff file.

    Gets first 10 slices and uses them to calculate features.

    Args:
        file_path: absolute path to the tif file. 
    '''
    downsampled = pre_process_tiff(file_path)
    feats = feature_extraction.extract_features(downsampled)
    return feats


def classify_image(file_path, model_path, features=None):
    '''Classify image based on a previously fitted model.'''
    p = Path(file_path)
    img_features = get_features_from_tiff(file_path)
    if features is not None:
        img_features = img_features[features]
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"{model_path} not found.")
        return
    orientation = model.predict([img_features])[0]
    return 'l' if orientation == 1 else 'v'


def fit_SVC(n, samples_dir, exp_data, save=False, output_dir=None, name=None, features=None):
    '''Calculates the SVC model.

    Args:
        n: number of training samples.
        samples_dir: path to the samples.
        exp_data: annotated data, as a dict. Must have the following structure:
        `{number_of_features: n, 
        l: {sample:[start,end],...},
        v: {sample:[start,end],...}}`
        save: boolean to determine if the model should be saved or not.
        output_dir: path where the model will be saved
        name: name for the model file
        features: list with the indices of selected features. All features
    are used by default, but `features` allows to fit the model with only part
    of the features.
    '''
    # Gets half of the training samples from each orientation
    # `v` is marked as class 0 and `l` is marked as class 1
    v_samples = get_training_samples('v', samples_dir, exp_data, n//2)
    l_samples = get_training_samples('l', samples_dir, exp_data, n//2)
    X = np.concatenate((v_samples, l_samples))
    Y = np.zeros(X.shape[0])
    # masks second half of features as `l` orientation
    v = v_samples.shape[0]
    Y[v:] = 1

    # TODO: error check this
    if features is not None:
        X = X[:, features]

    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

    if save and output_dir and name:
        pipe.fit(X, Y)
        with open(os.path.join(output_dir, name), 'wb+') as f:
            pickle.dump(pipe, f)
    else:
        scores = cross_val_score(pipe, X, Y, cv=5)
        print(f"{scores.mean()} accuracy.")
        print(f"Standard deviation of {scores.std()}")


def plot_svc(samples_dir, exp_data, n=600, features=[0, 1]):
    '''Creates a Decision Boundary plot, with an SVC that only takes two
    features

    Args:
        n: number of samples
        features: list with the indices of the features that will be used.
    Needs to have len of 2 and values must be valid indices in the features
    list.
    '''
    if len(features) != 2:
        raise ValueError("Can only display visualization for 2 features.")
    if max(features) > exp_data['number_of_features']:
        raise ValueError(
            f"Provide valid indices for the features array. Should be within range 0 <= x < {exp_data['number_of_features']}")
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    v_samples = get_training_samples('v', samples_dir, n//2)
    l_samples = get_training_samples('l', samples_dir, n//2)
    X = np.concatenate((v_samples, l_samples))
    X = X[:, features]
    Y = np.zeros(X.shape[0])
    # masks second half of features as `l` orientation
    v = v_samples.shape[0]
    Y[v:] = 1

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.05, random_state=17)

    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    pipe.fit(X_train, Y_train)

    fig, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        pipe, X, cmap='RdBu', alpha=0.8, ax=ax, eps=0.5
    )
    ax.scatter(
        X[:, 0], X[:, 1], c=Y, cmap=cm_bright,  alpha=0.6
    )
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=Y_test,
        cmap=cm_bright,
        edgecolors="k",
    )

    plt.show()
