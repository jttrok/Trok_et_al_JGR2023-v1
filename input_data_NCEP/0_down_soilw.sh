#!/bin/bash

# download NCEP-R2 soil moisture data from the 0-10cm layer
proj_dir="/path/to/main_project_folder/" # edit this line
target_dir=$proj_dir+"input_data_NCEP/NCEP-R2/soilw/"
cd $target_dir

for yr in {1979..2021}
do
    wget https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/Dailies/gaussian_grid/soilw.0-10cm.gauss.${yr}.nc
done
