Getting Started
===============

Installation
------------
 
The project uses `conda <https://docs.conda.io>`__ to manage dependencies.
If you don’t already have conda, you can download and install it from the official website.

Make a copy of the repo (e.g. with ``git clone``), then ``cd`` into the root folder of the repo.

Recreate the conda environment with the dependencies listed in ``environment.yml`` in the repo's root:
 
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
* ``notebooks``: contains examples of some steps of the pipeline. Also used for more specific visualizations. 
 
Running the code
----------------

Refer to the Getting Started session of each module for how to run the code.

To process raw data, start with `Getting Started <Data_processing/Overview.html>`__.
To analyze the output of the processing step, go to `Getting Started <Data_analysis/Overview.html>`__.
 
The analyses can be executed using the provided jupyter notebooks, or using the GUI.

Community Guidelines
--------------------

Thank you for being interested in ``snazzy``!
Check more information about how to get involved in the Contributing section of the `Github repository's Readme <https://github.com/ACRLab/snazzy/blob/main/README.md#Contributing>`__.