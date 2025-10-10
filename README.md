# SNAzzy: an image processing pipeline for investigating global Synchronous Network Activity

SNAzzy is a Python package for studying synchronous network activity (SNA) in Drosophia embryos via high-thoughput microscopy.
The software includes processing raw data into individual `.tif` files, quantification of fluorescence and changes in morphology, a custom peak detection algorithm, and a GUI for data visualization and curation.

## Getting Started

Refer to the README files inside the `snazzy_processing` or `snazzy_analysis` packages for details on running the code.

### Installation
 
The project uses [conda](https://docs.conda.io) to manage dependencies.
If you don’t already have conda, you can download and install it from the official website.

Make a copy of the repo (e.g. with `git clone`), then `cd` into the root folder of the repo.

Recreate the conda environment with the dependencies listed in `environment.yml` in the repo's root:
 
```
    conda env create -f environment.yml
``` 
Activate the environment:
 
```
    conda activate snazzy-env
```

## Testing

Tests can be run with pytest.

You can run the test suite from the project’s root directory to test everything at once.
Make sure the environment is active, and then run:

```
    pytest
```

## Contributing

Thank you for being interested in `snazzy`!

We accept contributions of all sorts: improving documentation, submitting bug reports, adding feature requests or writing code.
Feel free to create an issue or a pull request!

If you are new to open source and need help creating a pull request, we recommend taking a look at the [first-contributions repository](https://github.com/firstcontributions/first-contributions#first-contributions).

### How to report a bug

Please open an issue for any bugs or requests for help analyzing your data.

When filing an issue, please add the following information:

1. What operating system are you using?
2. What did you expect to see?
3. What did you see instead?

> [!IMPORTANT] 
> For issues related to data analysis, please provide an example dataset.

### How to suggest a feature or enhancement

Please file an issue explaining the desired feature or enhancement.
