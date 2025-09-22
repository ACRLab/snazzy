Getting Started
===============

Installation
------------
 
The project uses `conda <https://docs.conda.io>`__ to manage dependencies.
If you don’t already have conda, you can download and install it from the official website.

Make a copy of this repo (e.g. with ``git clone``), then ``cd`` into the root folder of the repo.

Recreate the conda environment:
 
.. code::

    conda env create -f=environment.yml
 
Activate the environment:
 
.. code:: 

    conda activate snazzy-env

Organization
------------
 
The code is split in two modules.

Parsing raw data into csv files with the relevant ROI metrics is done using ``snazzy_processing``.
The data analysis and GUI access is done using ``snazzy_analysis``.

Each one of the modules has the following structure:
 
* ``snazzy_[pkg]``: core code
* ``tests``: contains tests for the code
* ``data``: contains the data for the analysis. This folder is kept out of github, and should be populated in your local copy
* ``results``: contains the results of the analyses. It is also kept out of github and will be populated by performing the analyses
* ``notebooks``: contains examples of some steps of the pipeline. Also used for more specfic visualizations. 
* ``docs``: project documentation
 
Running the code
----------------

Refer to the Getting Started session of each module for how to run the code.

To process raw data, start with `Getting Started <Data_processing/Overview.html>`__.
To analyze the output of the processing step, go to `Getting Started <Data_analysis/Overview.html>`__.
 
The analyses can be executed using the provided jupyter notebooks, or running the files in the ``scripts`` directory.
The recommended way to analyze your data is to go through the notebooks in the following order:

1. ``process-raw-data.ipynb``
2. ``vnc-lengh.ipynb``
3. ``activity.ipynb``

There are details on how to use the code in each one of the notebooks.
 