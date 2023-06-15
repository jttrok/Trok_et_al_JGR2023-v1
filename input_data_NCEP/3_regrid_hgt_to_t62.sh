#!/bin/bash

# regrid NCEP-R2 geopotential height data to the T62 grid used by NCEP-R2 soil moisture and 2-meter temperature data
# (need to run this script after downloading NCEP-R2 tmax files)

proj_dir="/path/to/main_project_folder/" # edit this line
target_grid=$proj_dir+"input_data_NCEP/NCEP-R2/tmax/tmax.2m.gauss.2010.nc"

for yr in {1979..2021}; do
    infile=$proj_dir+"input_data_NCEP/NCEP-R2/hgt/hgt.${yr}.nc"
    outfile=$proj_dir+"input_data_NCEP/NCEP-R2/hgt/hgt.gauss.${yr}.nc"
    ncremap -a bilinear -d ${target_grid} -i ${infile} -o ${outfile}
done
