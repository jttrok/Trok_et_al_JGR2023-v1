#!/bin/bash

# create nested directories for input data files

proj_dir="/path/to/main_project_folder/" # edit this line

regions=('northcentral_north_america' 'southcentral_north_america' 'southeastern_north_america' 'southwestern_europe' 'western_europe' 'central_europe' 'eastern_europe' 'northeastern_europe' 'northeastern_asia' 'southeastern_asia' 'northsouthern_south_america' 'southsouthern_south_america' 'southwestern_africa' 'southeastern_africa' 'southwestern_australia' 'southeastern_australia' 'west_texas' 'east_texas')
timescale=("daily" "hourly")
variables=("tmax" "swvl1" "z")

for reg in ${regions[*]}; do
    mkdir ${proj_dir}"input_data_ERA5/"${reg}
    for time in ${timescale[*]}; do
        mkdir ${proj_dir}"input_data_ERA5/"${reg}"/"${time}
        for var in ${variables[*]}; do
            if [[ $time == "hourly" && $var == "tmax" ]]; then
                mkdir ${proj_dir}"input_data_ERA5/"${reg}"/"${time}"/t2m"
            else
                mkdir ${proj_dir}"input_data_ERA5/"${reg}"/"${time}"/"${var}
            fi
        done
    done
done
