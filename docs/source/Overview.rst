Overview
========

Pasnascope is a Python package to automate the extraction of primary data (VNC length, embryo size, and activity) from fluorescence imaging.
The package is an image processing pipeline, that can be divided intro three main stages:

* Process the raw images into several movies for individual embryos
* Calculate signal intensity inside ROIs
* Measure the VNC length

Process raw images
------------------

Since the imaging is done with a large Field of View microscope, usually during 6 hours or more, the raw images tend to be in the range of 50 ~ 100 GiB.
The simplest way to handle the raw data is to crop it in individual movies.
There is a considerable amount of background pixels that can be ignored in the raw data, so after cropping the embryos, all individual movies combined take about 30% of the original memory space.
This already saves considerable ROM memory but most importantly, it means we can easily load individual movies in the RAM of a common computer (8~16 GB RAM), without needing to use memory mapped files.

The algorithm to process the raw image can be resumed as:
1. Get the maximum projection of each pixel for the first 10 frames
2. Automatic threshold (Triangle method)
3. Binarize the image 
4. Mark connected regions
5. Select regions based on pixel count
6. Determine the bounding boxes for each movie based on each connected region
7. Open the image (mmap) and save the individual movies as tif

To calculate the bounding boxes of each embryo, we first take the maximum projection of each pixel for the first 10 frames, and then use the Triangle threshold method to binarize the image.
The Triangle threshold is a good choice here because the image has a lot of background pixels, resulting in an unimodal histogram that is centered around the background pixels average value.

Once we have the binary image, we traverse it to identify each embryo.
Whenever a foreground pixel is found, we mark all connecting foreground pixels, and also keep track of the amount of pixels marked and the extreme points (minimum and maximum coordinates in both dimensions).
The pixel count is used to determine if the marked area really corresponds to an embryo, or just a smaller artifact that was erroneously considered a foreground.
Regions corresponding to the embryo's eyes or gut are examples of smaller artifacts that sometimes are included in the binary image, but can easily be removed due to its size.
The extreme points are then used to generate the bounding boxes, which will determine the positions where the image will be cropped.

The raw image is opened as a memory map using `numpy`, and the individual embryos are cropped and saved as tif files.
For each embryo, we generate two files: one for the active channel and another for the structural channel.

ROIs and signal intensity
-------------------------

The ROIs are calculated for a given interval of frames.
By default, a single ROI is calculated for groups of 10 frames to speed up the process based on the fact that the VNC signal won't change considerably within this interval.
This is a good approximation and the speed up justifies the eventual errors in readings caused by movement (see `activity.ipynb <LINK>` for details about the error in activity caused by downsampling).

The ROI algorithm can be resumed as:
1. Average the group of frames into a single 2D matrix
2. Automatic threshold (Otsu's method)
3. Binarize the image 
4. Remove small holes inside the VNC
5. Select the largest group of connected foreground pixels
6. Return a mask that matches the largest label

To calculate the signal intensity, we apply the mask to the embryo and calculate the mean pixel value.
The active and structural channel measurements can be exported as a `.csv` file and further processed using the code from `pasna_fly <LINK>`.

TODO: Describe ratiometric activity calculation?

VNC length
----------

The VNC length is used as a way to characterize the different developmental stages of the embryo.
The algorithm to calculate the VNC length is named Center Line Estimation.
The idea is to find the line that will pass through the center of the VNC.

To determine this line, we go over the following steps:
1. Binarize the image
2. Apply a 'chessboard' distance transform 
3. Determine the maxima points from the distance transform
4. RANSAC the points to eliminate outliers (usually resulting from areas that correspond to brain lobes)
5. Estimate VNC length with the line fitted with RANSAC

The `vnc-length <LINK>` notebook has a more complete description and illustrates how the algorithm works.

Embryo size
-----------

The full embryo size is calculated by approximating the entire embryo shape as an ellipse, and measuring this ellipse's diameter.

The steps to calculate the embryo size are:
1. Equalize the image histogram
2. Automatic threshold (Triangle method)
3. Binarize the image
4. Calculate the corresponding ellipse's major axis