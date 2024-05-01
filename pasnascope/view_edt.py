import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.draw import disk, ellipse
from pasnascope.centerline import binarize, get_DT_maxima


def view_DT_3D_plot(image, metric=None):
    image = binarize(image)
    if metric:
        distance = ndi.distance_transform_cdt(image, metric=metric)
    else:
        distance = ndi.distance_transform_edt(image)
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(X, Y, distance, cmap=plt.cm.plasma)
    plt.show()


def view_DT(image, metric=None):
    image = binarize(image)
    if metric:
        distance = ndi.distance_transform_cdt(image, metric=metric)
    else:
        distance = ndi.distance_transform_edt(image)
    fig, ax = plt.subplots()
    ax.imshow(distance)
    plt.show()


def count_markers(image, metric):
    markers = []
    for img in image:
        bin_img = binarize(img)
        markers.append(get_DT_maxima(bin_img, metric=metric))
    return markers


def view_count_markers(img):
    edts = count_markers(img, None)
    edt_chess = count_markers(img, 'chessboard')
    edt_cab = count_markers(img, 'taxicab')

    fig, ax = plt.subplots()
    ax.set_title('Frequency of local maxima')
    ax.plot(edts, label='EDT')
    ax.plot(edt_chess, label='chessboard')
    ax.plot(edt_cab, label='taxicab')
    ax.legend()
    plt.tight_layout()
    plt.show()


def view_DT_disk_and_ellipse():
    img = np.zeros((50, 50))
    disk_mask = disk((18, 35), 6)
    img[disk_mask] = 1
    ell_mask = ellipse(25, 25, 6, 20)
    img[ell_mask] = 1

    distance = ndi.distance_transform_edt(img)
    distance_chess = ndi.distance_transform_cdt(img, metric='chessboard')
    distance_taxi = ndi.distance_transform_cdt(img, metric='taxicab')

    img[disk_mask] = 3

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()

    ax[0].imshow(img)
    ax[0].set_title('Binary image')
    ax[1].imshow(distance)
    ax[1].set_title('Euclidian transform')
    ax[2].imshow(distance_chess)
    ax[2].set_title('Chessboard transform')
    ax[3].imshow(distance_taxi)
    ax[3].set_title('Taxicab transform')

    for aa in ax:
        aa.set_axis_off()
    plt.show()
