#!/bin/bash

# converts all daily ERA5 daily files into the t62 grid used in the NCEP-R2 reanalysis

proj_dir="/path/to/main_project_folder/" # edit this line

target_grid=${proj_dir}+"input_data_ERA5/gauss.grid.nc" 

regions=('northcentral_north_america' 'southcentral_north_america' 'southeastern_north_america' 'southwestern_europe' 'western_europe' 'central_europe' 'eastern_europe' 'northeastern_europe' 'northeastern_asia' 'southeastern_asia' 'northsouthern_south_america' 'southsouthern_south_america' 'southwestern_africa' 'southeastern_africa' 'southwestern_australia' 'southeastern_australia' 'west_texas' 'east_texas')
variables=("tmax" "swvl1" "z")
years="$(seq 1979 2021)"

for reg in ${regions[*]}; do
	echo ${reg}
	for var in ${variables[*]}; do
		echo ${var}
		for yr in ${years[*]}; do
			path=${proj_dir}"input_data_ERA5/"
			fbase="${path}${reg}/daily/${var}/${reg}_${var}_daily_${yr}"
			infile="${fbase}.nc"
			outfile="${fbase}_t62.nc"
			echo $infile
			echo $outfile
			
			ncremap -a bilinear -d ${target_grid} -i ${infile} -o ${outfile}
		done
	done
done
