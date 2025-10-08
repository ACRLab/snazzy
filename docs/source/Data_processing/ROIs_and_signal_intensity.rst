ROIs and signal intensity
=========================

When running the pipeline from the jupyter notebook, a single ROI is calculated for each frame.
The processing can be sped up by calculating a single ROI for groups of 5 or 10 frames, instead of a single ROI per frame. 
This is possible if the sample signal doesn't change considerably within this interval.
Therefore, calculating one ROI per group of frames is a good approximation, and it can be useful for quick analyses, at the cost of the eventual errors in readings caused by movement (see ``activity.ipynb`` for details about the error in activity caused by downsampling).

The ROI algorithm can be summarized as:

1. Average the group of frames into a single 2D matrix
2. Automatic threshold (Otsu's method)
3. Binarize the image 
4. Seal small holes inside the VNC
5. Select the largest group of connected foreground pixels
6. Return a mask that matches the largest label

For some datasets with lower VNC signal, a higher threshold value tends to provide better results.
To change the threshold method, change the ``threshold_method`` parameter in the ``pipeline.measure_vnc_length`` function to ``otsu``.

Even though the signal from pixels inside the VNC is stable, the selected threshold value is not always perfect.
After binarizing the image, the VNC might contain small regions that were lower than the calculated threshold (i.e., holes).
These regions are merged back into the VNC if they are completely contained inside the VNC binary component.

To calculate the signal intensity, a mask is applied to the embryo and the mean pixel value is calculated.
The dynamic and structural channel measurements are exported as a ``.csv`` file and further processed using ``snazzy_analysis``.

Visualizing calculated ROIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ROIs can be inspected visually by running the ``plot_countours.py`` script.
The script displays a matplotlib animation with an overlayed ROI contour.
To display it, ``cd`` into the ``snazzy_processing`` directory, and run the file:

.. code:: bash

    python3 scripts/plot_contours.py

It will look for any dataset directories you have inside the ``./data`` directory and present the available options in the terminal.
Animations can be paused by pressing any key.