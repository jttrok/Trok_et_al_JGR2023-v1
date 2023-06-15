# Using machine learning and partial dependence analysis to investigate coupling between soil moisture and near-surface temperature

####################################################################
Supporting code for Trok et al. (2023) "Using machine learning and partial dependence analysis to investigate coupling between soil moisture and near-surface temperature"

To Cite:

Trok, J. T., Davenport, F. V., Barnes, E. A., & Diffenbaugh, N. S. (2023). Using Machine Learning with Partial Dependence Analysis to Investigate Coupling Between Soil Moisture and Near-surface Temperature. Journal of Geophysical Research: Atmospheres, 128, e2022JD038365. https://doi.org/10.1029/2022JD038365

Please contact Jared Trok at trok@stanford.edu with any questions about the code.
####################################################################

### subdirectories
The project directory is organized into the following subdirectories:

- input_data_ERA5: 
    - contains all scripts necessary to download and pre-process ERA5 data over each analysis region
    - see README inside this subdirectory for details
    - See https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels for details about the ERA5 dataset
    - See https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land for details about the ERA5-Land dataset  
- input_data_NCEP: 
    - each sub-directory (e.g., hgt/) contains scripts necessary to download NCEP/DOE Reanalysis II data.
    - See https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html for more details about the NCEP/DOE Reanalysis II dataset
- notebooks: 
    - contains the jupyter notebooks necessary to perform the analysis
    - see README inside this subdirectory for details
- figures_ERA5:
    - placeholder used for .jpg files created in "notebooks/5_figures_ERA5.ipynb"
- figures_NCEP:
    - placeholder used for .jpg files created in "notebooks/5_figures_NCEP.ipynb"
- processed_data_ERA5:
    - placeholder used for ERA5 processed data files
    - run "processed_data_ERA5/0_make_region_dir.sh" to create nested directories for each region
- processed_data_NCEP:
    - placeholder used for NCEP processed data files
    - run "processed_data_NCEP/0_make_region_dir.sh" to create nested directories for each region
- project_utils:
    - functions called within jupyter notebooks
    - these are used to build the CNN, load CNN inputs, load CNN hyperparameters, etc.
    
####################################################################
### Python Environments: 

Prior to running these scripts, it is necessary to create a python environment with all the dependencies listed in the "pyproject.toml" file.

We recommend using Poetry (https://python-poetry.org/docs/) to manage this python environment. 

(1) Download and install Poetry (instructions at https://python-poetry.org/docs/)
(2) Create a new Poetry environment
(3) Replace the newly created pyproject.toml and poetry.lock file those provided in this directory
(4) Run "poetry update" from the command line

Note: This Poetry environment may need to be installed within a conda environment with python/3.9.x and proj/8.2.0.

Note: A separate environment with cartopy is needed to run "5_figures_ERA5.ipynb" and "5_figures_NCEP.ipynb".

####################################################################
### Steps to perform this analysis:

1. Install all necessary dependencies in a python environment (see above)
2. Install project_utils/ in python environment using the command: "pip install -e . --user"
3. Download ERA5/ERA5-Land reanalysis data by running the scripts in input_data_ERA5/
4. Download NCEP-NCAR-R2 reanalysis data by running the scripts in input_data_NCEP/
5. Run "0a_make_region_dir.sh" in "processed_data_ERA5/" and "processed_data_NCEP/"
6. Run the notebooks contained in notebooks/
