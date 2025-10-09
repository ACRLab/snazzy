# SNAzzy Processing
 
Raw data processing for the SNAzzy pipeline.

### Running the code

To run the pipeline, use the jupyter notebook `snazzy_processing_pipeline.ipynb`.
Sample data is available in zenodo: https://doi.org/10.5281/zenodo.17295552.
To try the code, please download and extract the datasets first.
 
### Organization
 
* `snazzy_processing`: contains the core code used in all analysis
* `scripts`: Python code that combine the different modules in `snazzy_processing`, primarily for visualizing movies
* `tests`: contains tests for the code
* `data`: contains the data for the analysis. This folder is kept out of github, and should be populated in your local copy
* `results`: contains the results of the analyses. It is also kept out of github and will be populated by performing the analyses
* `notebooks`: contains examples and the front-end for using the `snazzy_processing` main module
* `docs`: project documentation
 
### Analyses
 
The analyses can be executed using the provided jupyter notebooks.
Use the notebook `snazzy_processing_pipeline.ipynb` to run the pipeline.
The other notebooks are used to understand in details the pipeline stages.

### Adding data
 
After processed, each dataset will have one corresponding folder inside `./data/`.
By running the code in `snazzy_processing_pipeline.ipynb`, the raw data will be parsed and saved inside `./data/{dataset_name}/embs`.
To compare the calculated VNC length against manual measurements, add an `annotated` folder inside the dataset directory.
The measurements should be saved as a csv file.

Given the description above, the file structure inside the `data` folder should look like:
 
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