# GUI

The graphical user interface is written using `PyQt6` and `pyqtgraph`.

The GUI's main functionalities are:

1. Visualize and adjust peak data.
2. Combine multiple experiments as a group.
3. Compare multipe groups.
4. Inspect TIF movies in sync with the DFF signal.
5. Inspect all parameters used in the analysis and share them with other users.

## Loading the GUI

First step to use the GUI is to activate the conda environment.
Refer to the Getting Started session if you haven't created an environment yet.

```sh
conda activate pscope_analysis
```

Then, from the pasna_analysis directory, run the GUI code:

```sh
python3 pasna_analysis/gui/gui.py
```

## Using the GUI

There are two primary modes to use the GUI:

1. Open a single experiment:

Allows you to visualize peak detection results and adjust parameters if needed.

2. Compare experiments:

You can load more than one experiment to a Group, or have multiple Groups to visualize comparisons.
When more than one experiment is loaded, you cannot change the analysis parameters anymore.
In this mode, the analysis results are read-only.
Therefore the general workflow is to first open each experiment separately and make sure all parameters are correct for peak detection.
The comparison plots in the Plot menu will show results by Group.
In the upper left corner of the GUI, a dropdown menu can be used to change the Group that is currently being visualized.

## Loading an Experiment

To load an Experiment select a directory that has `pasnascope` output.
The directory structure should be look like:

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

The `activity` and `lengths` directories, and the `full-length.csv` file are required.
If any of these is not found, the GUI will abort loading with an error message.

The `embs` directory will hold individual embryo movies in `.tif` format, and can be used to visualize embryo movies in sync with a DFF trace. 
The `emb_numbers.png` file represents a snapshot of the microscope's field of view at the start of the imaging session, and also shows the embryo id of each embryo.

## Config parameters

When loading an experiment the code will look for a config file named `peak_detection_params.json` inside the experiment directory and will use its data for the analysis.
If not found, a file with default parameters is created.
The default parameters can be found inside `config.py`. 
If you change any of the parameters, they will be recorded in this file.
To restore the original settings, simply delete `peak_detection_params.json` from the corresponding directory.
Sharing the `json` file allows someone else to reproduce your exact results in another machine.
Keep in mind that each directory should have its own `peak_detection_params.json` file.

The parameters that are most frequently changed are presented in the GUI when an Experiment is loaded.
From this window it's possible to set:

* Group name: name of the group that contains this experiment dataset
* First peak threshold: minimum time in minutes that has to pass before any peak happens.
Used to make sure that the first peak caught at the imaging session is really the activiy onset.
* To_exclude: embryo numbers that will be excluded from the analysis. These embryos will be excluded from the analysis.
* To_remove: embryo numbers that will be analyzed, but will show up in the 'Removed' group.
* Embryos that have it's first peak before the first peak threshold or that were marked by the user as removed will also be at the to_remove category.
* Has_transients: if selected the code will try to identify and skip the first peak if it's likely just a transient.
* Has_dsna: if selected the code will try to determine dSNA and ignore all peaks that happen after dSNA start.
* Dff_strategy: Combo box with the baseline strategy methods.
`local_minima` will pick the bottom 11 points out of the `baseline_window_size` and use that average as the baseline.
`baseline` will split the DFF values into bins and use the average of the most frequent bin as the baseline.
This method assumes that the bursts of activity are sparse, so that for all windows the most frequent bin falls into the baseline values.

Inside the File menu there is an option to open the `json` file and change any of its parameters.
Updating the file causes the entire Experiment to be recreated with the new configuration data.

## Visualizing traces

Once the data is loaded, you should see something similar to this:

The top app bar has buttons to change the data presentation.
Below the top app bar there are two sliders.
The first is for the frequency cutoff, which controls how much the signal is smoothed for the finding peaks algorithm.
The second is the peak width parameter, used to determine the start and end times of each peak.
The sidebar presents which embryos are currently considered for plots and analysis, and which ones should be removed.
You can toggle the embryo status between these two categories.
In the main view you will see the DFF trace of the currently selected embryo.
The pink dots represent the peak indices.
By pressing `shift` + `left mouse click` you can add a new peak to the plot.
Because we usually have many points over the X axis, it can be hard to click exactly where we want the peak index to land.
To help with this, the actual peak index after clicking in the local maximum value for a small window around the point that was clicked. 
By pressing `CTRL` + `left mouse click` you can remove a peak.
It also works on a small X axis range just like when adding new peaks.

You can also visualize the signal from each channel, by clicking on the button in the right of the screen.
This window will present the signal from each channel and also the hatching point, which can be changed manually by dragging the line. 

## Plots menu

### Field of View
### View embryo movies
### View phase boundaries
### View plots
### View comparison plots
