Supporting code for Trok et al. (2023) "Using machine learning and partial dependence analysis to investigate coupling between soil moisture and near-surface temperature"

To Cite:

Trok, J. T., Davenport, F. V., Barnes, E. A., & Diffenbaugh, N. S. (2023). Using Machine Learning with Partial Dependence Analysis to Investigate Coupling Between Soil Moisture and Near-surface Temperature. Journal of Geophysical Research: Atmospheres, 128, e2022JD038365. https://doi.org/10.1029/2022JD038365

Please contact Jared Trok at trok@stanford.edu with any questions about the code.

####################################################################
# Download and Process ERA5/ERA5-Land Data

Scripts: 

0_make_region_dir.sh: 
    - run this script with "bash 0_make_region_dir.sh" to create nested directories for input data
    
1_check_for_missing_files.ipynb:
    - prints all missing input files
    
2_queue_missing_requests.ipynb:
    - for each missing file, sends a download request to the Climate Data Store
    - check the status of requests at https://cds.climate.copernicus.eu/#!/home

3_download_missing_requests.ipynb:
    - downloads completed requests from https://cds.climate.copernicus.eu/#!/home
    - run this notebook after requests have finished processing

4_convert_hourly_2_daily.ipynb:
    - aggregates all hourly files to daily files
    - for geopotential height and soil moisture we calculate daily means
    - for 2-meter temperature we calculate daily maximums

5_regrid_to_ncep_t62.ipynb:
    - converts all daily data files to the t62 gaussian grid used by the NCEP-R2 reanalysis
    - this script uses gauss.grid.nc as the target grid
    
####################################################################

Note: All of the above scripts contain the line: proj_dir="/path/to/main_project_folder/"
Edit this line with the system path before running each script.

See https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels for more details about the ERA5 dataset

See https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land for more details about the ERA5-Land dataset 

