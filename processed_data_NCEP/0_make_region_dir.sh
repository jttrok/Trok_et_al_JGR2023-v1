#!/bin/bash

# create directories for processed data files

proj_dir="/path/to/main_project_folder/" # edit this line

regions=('northcentral_north_america' 'southcentral_north_america' 'southeastern_north_america' 'southwestern_europe' 'western_europe' 'central_europe' 'eastern_europe' 'northeastern_europe' 'northeastern_asia' 'southeastern_asia' 'northsouthern_south_america' 'southsouthern_south_america' 'southwestern_africa' 'southeastern_africa' 'southwestern_australia' 'southeastern_australia')

for reg in ${regions[*]}; do
    mkdir ${proj_dir}"processed_data_NCEP/"${reg}
    mkdir ${proj_dir}"processed_data_NCEP/"${reg}"/JJA_pdp/"
    mkdir ${proj_dir}"processed_data_NCEP/"${reg}"/SM_shuff_JJA_pdp/"
done
