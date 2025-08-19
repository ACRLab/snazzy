# Pasna Analysis
 
Data analysis for `pasnascope`'s pipeline output.
 
### Installation
 
Make a copy of this repo (e.g. with `git clone`), then `cd` into the root folder of the repo.
Recreate the conda environment:
 
`conda env create --name pscope_analysis --file=environment.yml`
 
Activate the environment:
 
`conda activate pscope_analysis`

Install the pasna_analysis package with `pip`:

`pip install -e .`
 
### Organization
 
* `pasna_analysis`: contains the core code used in all analyses
* `pasna_analysis/gui`: GUI code
* `tests`: contains tests for the code
* `data`: contains the data for the analysis. This folder is not tracked, and should be populated in your local copy
* `results`: contains the results of the analyses. It is not tracked either and will be populated by performing the analyses
* `notebooks`: contains examples and the front-end for using the `pasna_analysis` main module
 
### Analyses
 
The analyses are primarily executed using the GUI.
After activating the environment, run `pasna_analysis/gui/gui.py` to start the GUI.
There are also jupyter notebooks available, which can be used alternatively and allows for image customization.
 
### Adding data
 
Each experiment should have one corresponding folder inside `./data/`.
The expected data is generated using the `pasnascope` package.
The file structure inside the `data` folder should look like:
The `embs` directory is used if you want to inspect movies inside the GUI.
The files are generated with `pasnascope`, as long as the flag `clean_up_data` in there is set to `False`. 
 
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
|           -- embs
|               emb1-ch1.tif
|               emb1-ch2.tif
|           full-length.csv
|           emb_numbers.png
```