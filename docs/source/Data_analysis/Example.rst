Example Analysis
================

An example of how to use the GUI to analyze the data from the ``snazzy_processing`` pipeline.

Open the GUI
------------

Open a terminal window and activate the conda environment.

.. code::

    conda activate snazzy-env

Then ``cd`` into the snazzy_analysis folder, and run the following command to open the GUI:

.. code::

    python3 snazzy_analysis/gui/gui.py

Refer to the `Getting Started <../Getting_Started.html>`__ documentation if you haven't installed conda or haven't created an environment yet.

Load data
---------

To load data in the GUI, select an entire folder that has ``snazzy_processing`` output.

The data from a folder is inspected and loaded as a Dataset object.
There are several configurable parameters that change how data is processed.
The parameters that change more often are presented as a dialog window as soon as we select a directory. 
For the example dataset, we are not going to change any of these parameters.
For more details about these parameters, refer to the GUI guide, section `Config Parameters <Graphical_User_Interface.html#config-parameters>`__.

More than one dataset can be loaded.
The GUI uses a 'Group' to manage datasets.
We can combine datasets by picking the ``Add dataset`` menu item, and picking an existing Group in the Load Dataset Modal.

Choose the `Compare with Dataset` option in the File Menu and load the Example 2 dataset.
Now, when you go to the Compare plots window, you should be able to see comparisons between the two selected Groups.
It is possible to load more groups, the plots are generated splitting the loaded data by Group.

Visualizing data
----------------

Once the data is loaded, the GUI presents a sidebar with accepted and removed embryos, and the currently selected embryo.
The signal from each channel can be inspected by clicking the button to the right of the trace plot.
The sidebar can be used to select other embryos.
Only the selected embryos are considered in any of the plots generated in the GUI. 

Adjusting peaks
---------------

The first option to change peaks is to change the frequency filter value.
Lower frequency values will result in more denoising, which will help if the signal has many fast oscillations that should be ignored.
A recommended workflow is to change the frequency slider and see how the selected trace looks.
Then click 'Apply Changes' and change the presentation mode to see All Embryos.
Inspect the new peaks for every embryo and stop once peaks are precise enough.
Because of the inherent biological variability of the data and inherent noisy acquisition, there will be cases where a given frequency value is good for most of traces, but not all of them.
To solve this problem, it is also possible to manually add new peaks or remove existing peaks.

The peak width can be controlled with the peak width slider.
The value of 0.98 works well for the majority of samples.
To better understand this parameter, refer to the `scipy.signal.find_peaks documentation <https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.signal.peak_widths.html#scipy.signal.peak_widths>`__. 
To inspect the peak widths, click the 'View Widths' button.
Increasing the value in the slider will increase the peak width, while decreasing the slider makes the peaks narrower.

.. NOTE:: Changes in the slider values must be applied to all samples by clicking "Apply Changes", otherwise they will be discarded.

Once all peak data looks good, we can open other directories as another Group, to compare trace properties between them.

Comparing with another Dataset
---------------------------------

Choose the `Compare with Dataset` option in the File Menu and load the Example 2 dataset.
Now, when you go to the Compare plots window, you should be able to see comparisons between the two selected Groups.
It is possible to load more groups, the plots are generated splitting the loaded data by Group.