ROI length
==========

The ROI length is used as a proxy to measure the embryonic neurodevelopmental progression.
It is calculated by center line estimation.
The idea is to measure the line that will pass through the center of the ROI.
This will correspond to the ventral nerve cord length.

To determine this line, we go over the following steps:

1. Binarize the image
2. Apply a 'chessboard' distance transform 
3. Determine the maxima points from the distance transform
4. RANSAC the points to eliminate outliers (usually resulting from areas that correspond to brain lobes)
5. Estimate ROI length with the line fitted with RANSAC

The ``vnc-length.ipynb`` notebook has a more complete description and illustrates how the algorithm works.

Embryo Full size
----------------

The full specimen size is calculated by approximating the entire sample shape as an ellipse, and measuring this ellipse's diameter.

The steps to calculate the sample size are:
1. Equalize the image histogram
2. Automatic threshold (Triangle method)
3. Binarize the image
4. Calculate the corresponding ellipse's major axis