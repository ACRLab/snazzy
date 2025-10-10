Getting Started
===============

Installation
------------
 
The project uses `conda <https://docs.conda.io>`__ to manage dependencies.
If you don’t already have conda, you can download and install it from the official website.
We suggest reading `Getting Started <https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html>`__ from the conda website for an introduction on how to use Conda.

Make a copy of the repo (e.g. with ``git clone``), then ``cd`` into the root folder of the repo.

Recreate the conda environment with the dependencies listed in ``environment.yml`` in the repo's root:
 
.. code::

    conda env create -f environment.yml
 
Activate the environment:
 
.. code:: 

    conda activate snazzy-env

Organization
------------
 
The code is split in two packages.
Parsing raw data into csv files with the relevant ROI metrics is done using ``snazzy_processing``.
The data analysis and GUI access is done using ``snazzy_analysis``.

Each one of the packages has the following structure:
 
* ``snazzy_[pkg]``: core code.
* ``tests``: contains tests for the code.
* ``data``: contains the data for the analysis. This folder is kept out of github, and should be populated in your local copy.
* ``results``: contains the results of the analyses. It is also kept out of github and will be populated by performing the analyses.
* ``notebooks``: illustrates how individual steps of the pipeline work. Also used for more specific visualizations.
 
Running the code
----------------

Refer to the Getting Started session of each package for how to run the code.

Two sample datasets with a reduced number of samples (to reduce dataset size) were uploaded to zenodo.
Please find the datasets here: https://doi.org/10.5281/zenodo.17295552.

To process raw data, start with `Getting Started <Data_processing/Overview.html>`__.
To analyze the output of the processing step, go to `Getting Started <Data_analysis/Overview.html>`__.
 
The analyses can be executed using the provided jupyter notebooks, or using the GUI.

Community Guidelines
--------------------

Thank you for being interested in ``snazzy``!
Check more information about how to get involved in the Contributing section of the `Github repository's Readme <https://github.com/ACRLab/snazzy/blob/main/README.md#Contributing>`__.