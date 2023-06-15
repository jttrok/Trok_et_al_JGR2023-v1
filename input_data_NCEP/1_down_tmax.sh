#!/bin/bash

# download NCEP-R2 daily maximum 2-meter temperature data
proj_dir="/path/to/main_project_folder/" # edit this line
target_dir=$proj_dir+"input_data_NCEP/NCEP-R2/tmax/"
cd $target_dir

for yr in {1979..2021}
do
    wget https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/Dailies/gaussian_grid/tmax.2m.gauss.${yr}.nc
done