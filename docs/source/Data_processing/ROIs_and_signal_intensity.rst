ROIs and signal intensity
=========================

The ROIs are calculated for a given interval of frames.
By default, a single ROI is calculated for groups of 10 frames to speed up the process based on the fact that the sample signal won't change considerably within this interval.
This is a good approximation and the speed up justifies the eventual errors in readings caused by movement (see `activity.ipynb` for details about the error in activity caused by downsampling).

The ROI algorithm can be resumed as:
1. Average the group of frames into a single 2D matrix
2. Automatic threshold (Otsu's method)
3. Binarize the image 
4. Remove small holes inside the VNC
5. Select the largest group of connected foreground pixels
6. Return a mask that matches the largest label

To calculate the signal intensity, we apply the mask to the embryo and calculate the mean pixel value.
The active and structural channel measurements are exported as a `.csv` file and further processed using the code from ``snazzy_analysis``.