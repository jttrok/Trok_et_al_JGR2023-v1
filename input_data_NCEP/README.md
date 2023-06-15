Supporting code for Trok et al. (2023) "Using machine learning and partial dependence analysis to investigate coupling between soil moisture and near-surface temperature"

To Cite:

Trok, J. T., Davenport, F. V., Barnes, E. A., & Diffenbaugh, N. S. (2023). Using Machine Learning with Partial Dependence Analysis to Investigate Coupling Between Soil Moisture and Near-surface Temperature. Journal of Geophysical Research: Atmospheres, 128, e2022JD038365. https://doi.org/10.1029/2022JD038365

Please contact Jared Trok at trok@stanford.edu with any questions about the code.

# Download and process NCEP/DOE Reanalysis II data

- 0_down_soilw.sh
    - download global maps of daily soil moisture data (0-10cm layer)
    - run with from command line with " bash 0_down_soilw.sh "
    
- 1_down_tmax.sh
    - download global maps of daily maximum 2-meter temperature data
    - run with from command line with " bash 1_down_tmax.sh "
    
- 2_down_hgt.sh
    - download global maps of 500 millibar geopotential height data
    - run with from command line with " bash 2_down_hgt.sh "
    
- 3_regrid_hgt_to_t62.sh
    - converts all hgt data to a T62 gaussian grid to match the temperature and soil moisture fields
    - run with from command line with " bash 3_regrid_hgt_to_t62.sh "

####################################################################

Note: All above scripts contain the line " proj_dir="/path/to/main_project_folder/" "
Edit this line using the system path before running each script.

See https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html for more details about the NCEP/DOE Reanalysis II dataset
