# Cognitive maps for generalizing rewards
This repo contains code and some figures used for running the experiment as well as the behavioural modelling in our paper "Hippocampal spatio-predictive cognitive maps adaptively guide reward generalization".


## System requirements

The code was developed using Python 3.6, R version 3.6.1 and Matlab2020b. Certain files are run in Jupyter Notebooks.
Code was tested on Windows 10.

Non-native Python libraries used:
`Numpy`
`Matplotlib`
`Scipy`
`Pandas`
`Networkx`
`Seaborn`

R libraries used:
`lme4`
`lmrTest`
`ggplot2`
`car`
`caret`
`latex2exp`
`varhandle`
`reshape2`

## Installation
Code can be installed by downloading repository and downloading required libraries.

## Demo and instructions for use

Files to produce quantitative results of interest can be run after download and installation of above dependencies.
Examples:
To run the behavioural analyses, run the R file "parameter_fits.R" in the src folder.
To produce the model data used in these analyses, run the "parameter_search_notebook.ipynb" script in Jupyter Notebook.
To produce the predictors used in the fMRI analyses, run the "extract_fmri_notebook.ipynb" script in Jypter Notebook.
