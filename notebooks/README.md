Supporting code for Trok et al. (2023) "Using machine learning and partial dependence analysis to investigate coupling between soil moisture and near-surface temperature"

To Cite:

Trok, J. T., Davenport, F. V., Barnes, E. A., & Diffenbaugh, N. S. (2023). Using Machine Learning with Partial Dependence Analysis to Investigate Coupling Between Soil Moisture and Near-surface Temperature. Journal of Geophysical Research: Atmospheres, 128, e2022JD038365. https://doi.org/10.1029/2022JD038365

Please contact Jared Trok at trok@stanford.edu with any questions about the code.

# Notebooks:

- 0a_process_CNN_inputs.ipynb:
    - process the input data to create CNN input files
    - removes the long-term temporal trend in SM, GPH, and TMAX at each grid cell
    - calculates standardized calendar-day anomalies for SM and GPH
    - loops over all regions, then repeats for both ERA5 and NCEP-R2 datasets

- 0b_calc_regional_averages.ipynb:
    - calculates regional mean SM, GPH, and TMAX over the prediction region
    - area-weighted mean over all non-ocean grid cells within the region bounds
    - loops over all regions, then repeats for both ERA5 and NCEP-R2 datasets

- 1_train_CNN_and_calc_PDPs.ipynb:
    - builds and trains a convolutional neural network to predict TMAX over a given region
    - then computes the partial dependence plot between SM and TMAX averaged over all summer days
    - loops over all regions, then repeats for both ERA5 and NCEP-R2 datasets

- 2_train_CNN_without_SM.ipynb:
    - build and train CNN models without a SM input layer
    - this model will serve as a baseline model skill comparison
    - loops over all regions, then repeats for both ERA5 and NCEP-R2 datasets

- 3_train_CNN_and_calc_100_shuffled_PDPs.ipynb:
    - trains 100 CNNs with randomly shuffled SM inputs (each with a different random seed)
    - then computes the partial dependence plots for these 100 CNNs
    - loops over all regions, then repeats for both ERA5 and NCEP-R2 datasets

- 4_calc_individual_day_PDPs.ipynb:
    - calculate the partial dependence plot between SM and TMAX for 5 individual days
    - 5 days selected to be highest GPH, median GPH, lowest GPH, model best-hit, and model worst-miss
    - used for panel plots in Figure 2

- 5_figures_ERA5.ipynb:
    - creates manuscript Figures 1-9 and S1-S4

- 5_figures_NCEP.ipynb:
    - creates manuscript S5-S11

- temporary_model:
    - the tensorflow CNN model will be temporarily saved to this directory during the analysis
    
- analysis_fxns.py:
    - contains several python functions called within the main notebooks

####################################################################

Note: All of the following notebooks contain the line: sys.path.append(r'/path/to/main_project_folder/') 

Edit this above line with the system path before running each notebook.

