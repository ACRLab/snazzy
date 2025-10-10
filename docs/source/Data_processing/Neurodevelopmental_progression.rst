Neurodevelopmental Progression
==============================

The ROI length is calculated by center line estimation.
The general approach is to measure the line that will pass through the center of the ROI; this will correspond to the ventral nerve cord length.

Together the ROI length and the full embryo size are used as a proxy to measure the embryonic neurodevelopmental progression:

.. math::
    neurodevelopmental\ progression = \frac{embryo\ length}{ROI\ length}

To determine the ROI length, the following steps are used:

1. Binarize the image
2. Apply a 'chessboard' distance transform 
3. Determine the maxima points from the distance transform
4. RANSAC the points to eliminate outliers (usually resulting from areas that correspond to brain lobes)
5. Estimate ROI length with the line fitted with RANSAC

The ``vnc-length.ipynb`` notebook has a more complete description and illustrates how the algorithm works.

Embryo Full size
----------------

The full specimen size is calculated by approximating the entire sample shape as an ellipse, and measuring this ellipse's diameter.

The steps to calculate the embryo's size are:

1. Equalize the image histogram
2. Automatic threshold (Triangle method)
3. Binarize the image
4. Calculate the corresponding ellipse's major axis