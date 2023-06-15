import numpy as np
import pandas as pd
import xarray as xr
import time
import sys
sys.path.append(r'/path/to/main_project_folder')
from project_utils import parameters as param
from project_utils import utils as util
from project_utils import load_region
from project_utils import read_utils as read
import importlib
importlib.reload(param)
importlib.reload(util)
importlib.reload(load_region)
importlib.reload(read)


def get_model_inputs(region_str, nlag, SNOWFREE, hemisphere, dset):      
    hem, region_input_lat_bbox, region_input_lon_bbox, region_box_x, region_box_y, region_lat, region_lon, region_lon_EW, region_t62_lats, region_t62_lons = load_region.load_region_constants(region_str)
    
    # LOAD inputs
    x_dat_daily = read.get_hgt_soilw_daily_input(region_str, dset) 
    y_dat = pd.read_csv("../processed_data_"+dset+"/"+region_str+"/region_avg_tmax.csv")["tmax"]
    ind = np.arange(len(y_dat))
    caldays = pd.to_datetime(pd.read_csv("../processed_data_"+dset+"/"+region_str+"/region_avg_tmax.csv")['time']).dt.dayofyear
    time_vec = pd.read_csv("../processed_data_"+dset+"/"+region_str+"/region_avg_tmax.csv")['time']
    
    # LAG inputs
    if nlag:
        print('lagging now by',nlag,'day(s)')
        orig_x_dat = x_dat_daily
        orig_y_dat = y_dat
        orig_ind = ind
        orig_caldays = caldays
        lagged_already = True

        x_dat_daily = orig_x_dat
        y_dat = orig_y_dat
        ind = orig_ind
        caldays = orig_caldays

        lag_hgt_input = x_dat_daily[nlag:,:,:,0].reshape(param.n-nlag, len(region_t62_lats), len(region_t62_lons), 1)
        lag_soilw_input = x_dat_daily[:-nlag,:,:,1].reshape(param.n-nlag, len(region_t62_lats), len(region_t62_lons), 1)
        x_dat_daily = np.concatenate((lag_hgt_input, lag_soilw_input), axis=3)
        y_dat = y_dat[nlag:].reset_index(drop=True)
        ind = ind[:-nlag]
        caldays = caldays[nlag:].reset_index(drop=True)
        time_vec = time_vec[nlag:].reset_index(drop=True)

    # CLIP inputs to SNOWFREE months (remove DJF)
    if SNOWFREE:
        if hemisphere == 'south':
            print('removing JJA')
            SNOWFREE_time_vec = time_vec[(pd.to_datetime(time_vec).dt.month < 6) | (pd.to_datetime(time_vec).dt.month > 8)]
        elif hemisphere == 'north':
            print('removing DJF')
            SNOWFREE_time_vec = time_vec[(pd.to_datetime(time_vec).dt.month >= 3) & (pd.to_datetime(time_vec).dt.month <= 11)]
        SNOWFREE_idx = SNOWFREE_time_vec.index.values
        SNOWFREE_dates = SNOWFREE_time_vec.reset_index(drop=True)

        x_dat_daily = x_dat_daily[SNOWFREE_idx, :, :, :]
        y_dat = y_dat[SNOWFREE_idx].reset_index(drop=True)
        ind = np.arange(len(y_dat))
        caldays = caldays[SNOWFREE_idx].reset_index(drop=True)
        
        if hemisphere == 'south':
            caldays[caldays>152] = -1*(caldays[caldays>152]-244)
            caldays[caldays>0] = caldays[caldays>0]+122
            caldays[caldays<0] = caldays[caldays<0]*-1

    time_vec = SNOWFREE_time_vec
    caldays = caldays/caldays.max()   # normalize calday input
    
    return x_dat_daily, y_dat, ind, caldays, time_vec
