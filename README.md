# Pasna Analysis
 
Data analysis for `pasnascope`'s pipeline output.
 
### Installation
 
Make a copy of this repo (e.g. with `git clone`), then `cd` into the root folder of the repo.
Recreate the conda environment:
 
`conda env create --name pscope_analysis --file=environment.yml`
 
Activate the environment:
 
`conda activate pscope_analysis`

Install the pasnascope package with `pip`:

`pip install -e .`
 
### Organization
 
* `pasnascope`: contains the core code used in all analysis
* `scripts`: Python code that combine the different modules in `pasnascope`, primarily for visualizing movies
* `tests`: contains tests for the code
* `data`: contains the data for the analysis. This folder is kept out of github, and should be populated in your local copy
* `results`: contains the results of the analyses. It is also kept out of github and will be populated by performing the analyses
* `notebooks`: contains examples and the front-end for using the `pasnascope` main module
* `docs`: project documentation
 
### Analyses
 
The analyses can be executed using the provided jupyter notebooks, or running the files in the `scripts` directory.
The recommended way to analyze your data is to go through the notebooks in the following order:

1. `process-raw-data.ipynb`
2. `vnc-lengh.ipynb`
3. `activity.ipynb`

There are details on how to use the code in each one of the notebooks.
 
### Adding data
 
Each experiment should have one corresponding folder inside `./data/`.
The file structure inside the `data` folder should look like:
 
```
|-- project_folder
|   -- data
|       -- 20240501
|           -- activity
|               emb1.csv
|               ..
|           -- lengths
|               emb1.csv
|               ..
|           full-length.csv
```