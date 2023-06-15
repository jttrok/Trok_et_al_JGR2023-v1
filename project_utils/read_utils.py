import numpy as np
import pandas as pd
import xarray as xr
import random
import sys
sys.path.append(r'/path/to/main_project_folder/')
from project_utils import parameters as param
from project_utils import load_region
np.random.seed(101)
random.seed(201)

def get_hgt_soilw_daily_input(region_str, dset):
    
    hem, region_input_lat_bbox, region_input_lon_bbox, region_box_x, region_box_y, region_lat, region_lon, region_lon_EW, region_t62_lats, region_t62_lons = load_region.load_region_constants(region_str)
    
    hgt_xr = xr.open_dataset("../processed_data_"+dset+"/"+region_str+"/hgt_calday_anomalies.nc")
    soilw_xr = xr.open_dataset("../processed_data_"+dset+"/"+region_str+"/soilw_calday_anomalies.nc")
    
    dat_np = np.concatenate([np.array(hgt_xr['hgt_anom_no_trend']).reshape(param.n, len(region_t62_lats), len(region_t62_lons), 1), 
                             np.array(soilw_xr['soilw_daily_anom']).reshape(param.n, len(region_t62_lats), len(region_t62_lons), 1)], 
                            axis = 3)
    
    dat_np = np.nan_to_num(dat_np) # sets all input anomaly nan's to 0.0 to avoid issue with nan inputs into model
    
    return(dat_np)
