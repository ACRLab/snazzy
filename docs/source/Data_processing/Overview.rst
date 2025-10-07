Overview
========

``snazzy_processing`` is a Python package to automate the extraction of primary data: Ventral Nerve Cord (VNC) length and signal intensity from fluorescence imaging of Drosophila embryos.
The package is an image processing pipeline, that can be divided intro three main stages:

* Crop movies for individual specimens from raw data
* Calculate signal intensity inside ROIs
* Measure the VNC length

Use the jupyter notebook ``snazzy-processing-pipeline.ipynb`` to run the pipeline.

The data processing requires a ``.tif`` file.
There is built in support for converting ``.nd2`` files to ``.tif``.
This means that you can feed either ``.tif`` or ``.nd2`` files into the pipeline.
If your raw data is in another format, you must first convert it to ``.tif``.
ImageJ for example provides several plugins to convert files to tif, including the excellent `BioFormats extension <https://imagej.net/formats/bio-formats>`__.

Before running the pipeline, which is the last cell of the jupyter notebook, we must determine from where to crop each movie in the raw data.

The bounding boxes for each individual sample are determined via thresholding.
Because the image histograms might vary, some manual adjustment might be necessary.
Inspect where the bounding boxes will be created in the jupyter notebook.
They should cover the entire sample.
If there is a bounding box that covers more than one specimen, because maybe they were touching each other, those specimens must be ignored or further manually processed.
We can't use it directly because the resulting ROI will not match a single sample and the length and activity data will be wrong.

After the bounding boxes are determined, you can run the pipeline.
By default, the methods in the pipeline will not overwrite any data.
If data for a given sample is found in the output directory, it will simply skip that sample.
If you want to recalculate any data, first remove or rename the current files.

Refer to the other documentation pages for a description of the pipeline steps:

* `Process raw data <Process_raw_data.html>`__
* `ROI and signal intensity <ROIs_and_signal_intensity.html>`__
* `ROI length <ROI_length.html>`__