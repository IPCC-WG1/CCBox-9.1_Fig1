# CCBox-9.1_Fig1

[![DOI](https://zenodo.org/badge/599016775.svg)](https://zenodo.org/badge/latestdoi/599016775)

The Python script used is called:
compute_OHC_ThSL_ensemble_FGD_python3.py 

It uses only one input data file:
AR6_GOHC_GThSL_timeseries_MDP_2021-01-20.mat

This file contains the pre-processed OHC timeseries from a large number of contributors, created by Catia Domingues.

To run the code, you will need to edit the paths for plotdir, savedir and datadir based on your local directory structure. 

On running the code, the script creates two *.pickle files and corresponding *.csv files that contain the ensemble estimates of OHC and ThSL. It also generates four figure files that show the original input timeseries and the ensemble estimate, following the approach described by Palmer et al [2021]. 

The full list of data outputs are:

AR6_FGD_OHC_ThSL_ensemble_structural_uncertainty_0-700m.png
AR6_FGD_OHC_ThSL_ensemble_internal_uncertainty_0-700m.png
AR6_FGD_OHC_ThSL_ensemble_structural_uncertainty_700-2000m.png
AR6_FGD_OHC_ThSL_ensemble_internal_uncertainty_700-2000m.png

AR6_OHC_ensemble_FGD.csv
AR6_OHC_ensemble_FGD.pickle
AR6_ThSL_ensemble_FGD.csv
AR6_ThSL_ensemble_FGD.pickle 


For any questions, please contact Matt Palmer: matthew.palmer@metoffice.gov.uk 

References: 

Palmer et al [2021] “An ensemble approach to quantify global mean sea-level rise over the 20th century from tide gauge reconstructions” Environ. Res. Lett. 16 044043
https://iopscience.iop.org/article/10.1088/1748-9326/abdaec 
