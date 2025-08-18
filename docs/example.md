# Example Analysis

An example of how to use the GUI to analyze the data output from the raw image processing pipeline.

## Open the GUI

Open a terminal window and activate the conda environment:

```
conda activate pscope_analysis
```

Refer to the Getting Started documentation if you haven't installed conda or haven't created an environment yet.

## Load data

To load data in the GUI, select an entire folder that has pasnascope output.

The data from a folder is inspected and loaded as an Experiment object.
There are several configurable parameters that change how data is processed.
The parameters that change more often are presented as a dialog window as soon as we select a directory. 
For the example dataset, we are not going to change any of these parameters.
For more details about these parameters, refer to the GUI guide item 'Config Parameters'.

## Visualizing data

When the data is loaded the GUI presents a sidebar with accepted and removed embryos, and the currenlty selected embryo.
The sidebar can be used to select other embryos.
For the selected trace, we can see the identified peaks.
The signal from each channel can be inspected by clicking the button to the right of the trace plot.

### Adjusting peaks

The first option to change peaks is to change the frequency filter value.
Higher frequency values will result in more denoising, which will help if the signal has many fast oscillations that should be ignored.
A recommended workflow is to change the frequency slider and see how the selected trace looks.
Then click 'Apply Changes' and change the presentation mode to see All Embryos.
Inspect the new peaks for every embryo and stop once peaks are precise enough.
Because of the inherent biological variability of the data, there will be cases where a given frequency value is good for most of traces, but not all of them.
To solve this problem, it is also possible to manually add new peaks or remove exisiting peaks.

The peak width can be controlled with the peak width slider.
The value of 0.98 works well for the majority of samples.
To evaluate the peak width values, click the 'View Widths' button.
Increasing the value in the slider will increase the peak width, while decreasing the slider makes the peaks more narrow.

Once all peak data looks good, we can open other directories as another Group, to compare trace properties between them.

## Comparing with another Experiment

We can combine data from multiple experiments in two ways: either by adding more data from another experiment to the same Group, or by adding another Group and compare the different loaded Groups.
In both modes, it's not possible to change the peak detection parameters, that's possbile only when a single experiment is loaded.

Choose the `Compare with Experiment` option in the File Menu and load the Example 2 dataset.
Now, when you go to the Compare plots window, you should be able to see comparisons between the two selected Groups.
It is possible to load more groups, the plots are generated splitting the loaded data by Group.