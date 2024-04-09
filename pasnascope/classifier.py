import numpy as np
from time import time
import os
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

# Metadata about the experiment
# Will be moved to another place soon
exp_data = {
    'number_of_features': 5,
    'l': ['emb31', 'emb33', 'emb43', 'emb51'],
    'v': ['emb11', 'emb12', 'emb13', 'emb52', 'emb63']
}


def get_training_samples(orientation, n=500):
    '''Returns annotated data, based on a given orientation.

    Picks an equal amount of images from each sample.
    Args:
        orientation: `v` (ventral) or `l` (lateral). Embryo orientation.
        n: int. Amount of samples.
    '''
    img_dir = os.path.join(os.getcwd(), 'data', 'downsampled', 'features')
    num_samples = len(exp_data[orientation])
    # number of images per sample
    f = n//num_samples
    # preallocate X and add slices of size f from each sample:
    X = np.ones((n, exp_data['number_of_features']))
    i = 0
    for sample in exp_data[orientation]:
        file_name = f"feat-{sample}.npy"
        curr = np.load(os.path.join(img_dir, file_name))
        X[i:i+f] = curr[:f]
        i += f
    return X


def get_features_from_tiff(file_name):
    '''Extracts features from a tiff file.

    Gets first 10 slices and uses them to calculate features.

    Args:
        file_name: file_name, expected to be in the directory
        `pasnascope/data/embryos`.
    '''
    img_dir = os.path.join(os.getcwd(), 'data', 'embryos')
    img = imread(os.path.join(img_dir, file_name), key=range(10))
    # The downscale_factors should match the ones used to fit the model
    downsampled = pre_process.pre_process(img, (1, 2, 2))
    downsampled = np.average(downsampled, axis=0)
    feats = feature_extraction.extract_features(downsampled)
    return feats


def classify_image(file_name):
    img_features = get_features_from_tiff(file_name)
    model_path = os.path.join(os.getcwd(), 'results', 'models', 'SVC')
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"{model_path} not found.")
        return
    orientation = model.predict([img_features])[0]
    return 'l' if orientation == 1 else 'v'


def fit_SVC(n=600, save=False, features=None):
    '''Calculates the SVC model.

    Args:
        n: number of training samples.
        save: boolean to determine if the model should be saved or not.
        features: list with the indices of selected features. All features
    are used by default, but `features` allows to fit the model with only part
    of the features.
    '''
    # Gets half of the training samples from each orientation
    # `v` is marked as class 0 and `l` is marked as class 1
    X = np.concatenate(
        (get_training_samples('v', n//2), get_training_samples('l', n//2)))
    Y = np.zeros(n)
    Y[n//2:] = 1

    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

    if save:
        pipe.fit(X, Y)
        model_path = os.path.join(os.getcwd(), 'results', 'models')
        with open(os.path.join(model_path, "SVC"), 'wb+') as f:
            pickle.dump(pipe, f)
    else:
        scores = cross_val_score(pipe, X, Y, cv=5)
        print(f"{scores.mean()} accuracy.")
        print(f"Standard deviation of {scores.std()}")


def plot_svc(n=600, features=[0, 1]):
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

    X = np.concatenate(
        (get_training_samples('v', n//2), get_training_samples('l', n//2)))
    Y = np.zeros(n)
    Y[n//2:] = 1
    # Extracts only the two selected features
    X = X[:, features]

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
