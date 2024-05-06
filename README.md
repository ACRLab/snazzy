# Pasnascope
 
Image processing pipeline for pasnascope imaging.
 
### Installation
 
Make a copy of this repo (e.g. with `git clone`), then `cd` into the root folder of the repo.
Recreate the conda environment:
 
`conda env create --name pasnascope --file=environment.yml`
 
Activate the environment:
 
`conda activate pasnascope`
 
### Organization
 
* `pasnascope`: contains the core code used in all analysis
* `scripts`: contains Python code that combine the different modules in `pasnascope` to perform an analysis
* `tests`: contains tests for the code code
* `data`: contains the data for the analysis. This folder is kept out of github, and should be populated in your local copy
* `results`: contains the results of the analyses. It is also kept out of github and will be populated by performing the analyses
* `notebooks`: contains examples and the front-end for using the `pasnascope` main module
* `docs`: project documentation
 
### Analyses
 
The analyses can be executed using the provided jupyter notebooks, or running the files in the `scripts` directory.
The recommended order to analyze your data is to go through the notebooks in the following order: `process-raw-data.ipynb`, `vnc-lengh.ipynb`, `activity.ipynb`.
There are details on how to use the code in each one of the notebooks.
 
### Adding data sources
 
Sample data can be downloaded from (...).
By running the code in `process-raw-data.ipynb`, the raw data will be parsed and saved inside `./data/experiment_name/embs`.
To add annotated data, create a new directory within the current experiment directory and save each file as `f"{file_name}.csv"`, where `file_name` matches the file name in `./data/experiment_name/embs`.
As a suggestion, each experiment should be named after the date it was performed.
Within each experiment folder, there is an expected folder structure, which goes as follows:
 
```
|-- project_folder
|   -- data
|       -- 20240305
|           -- embs
|           -- annotated
|       -- 20240501
|           -- embs
|           -- annotated
```

If you are adding your own parsed files, be aware that in some places the code relies on a naming convention: movies should be saved in splitted channels and named as `embXX-ch1` and `embXX-ch2`, where `XX` is a number identifier.