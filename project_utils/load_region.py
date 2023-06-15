import numpy as np
import pandas as pd
import xarray as xr
import time
import sys
sys.path.append(r'/path/to/main_project_folder/')
from project_utils import parameters as param
from project_utils import utils as util
import importlib
importlib.reload(param)
importlib.reload(util)


#############################################################


def load_ERA5_region_cnn_hyperparams(region_str, shuffle_sm, no_sm, nlag=None):
    activation_list = ['sigmoid']
    optimizer_list = ['RMSprop']
    decay_rate = 0.5
    curr_batch = 512
    decay_list = [True]
    reg_list = [0.00095]
    nepochs = 10000
    
    if not shuffle_sm:
        if region_str == "eastern_europe":
            tts = [12] 
            lr_list = [0.35]
            dw_list = [0.65] 
            decay_steps = [35] 
            reg_list = [0.00095]
            if nlag == 0:
                dw_list = [0.70] 
            if nlag in [7, 14]:
                dw_list = [0.65] 
        elif region_str == "southeast":
            tts = [32]
            lr_list = [0.40] 
            dw_list = [0.52] 
            decay_steps = [30] 
            reg_list = [0.00095]
            if nlag == 0:
                dw_list = [0.55] 
        elif region_str == "southcentral_north_america":
            tts = [4]       
            lr_list = [0.35] 
            dw_list = [0.60] 
            decay_steps = [40]
            reg_list = [0.00095]
        elif region_str == "southeastern_asia":
            tts = [24]           
            lr_list = [0.23]
            dw_list = [0.42]
            decay_steps = [40] 
            reg_list = [0.00100]
            if nlag == 14:
                dw_list = [0.48] 
            if nlag == 30:
                dw_list = [0.45] 
        elif region_str == "northeastern_asia":
            tts = [10]      
            lr_list = [0.35]
            dw_list = [0.65] 
            decay_steps = [40]
            reg_list = [0.001]
            if nlag == 1:
                tts = [2]
            if nlag == 0:
                dw_list = [0.67]
            if nlag == 3:
                dw_list = [0.45]
        elif region_str == "southwestern_europe":
            tts = [50] 
            lr_list = [0.30]
            dw_list = [0.16] 
            decay_steps = [45] 
            reg_list = [0.00081] 
            if nlag in [0, 2]:
                reg_list = [0.0035]
            if nlag == 1:
                tts = [8]
            if nlag == 3:
                reg_list = [0.001]
            if nlag == 7:
                reg_list = [0.0035] 
                dw_list = [0.14] 
            if nlag in [14, 30]:
                reg_list = [0.005]
            if no_sm:
                reg_list = [0.005] 
                decay_steps = [50]
        elif region_str == "western_europe":
            tts = [34]    
            lr_list = [0.25]
            dw_list = [0.70]
            decay_steps = [40]
            reg_list = [0.00087]
            if no_sm:
                dw_list = [0.72]
            if nlag == 7:
                dw_list = [0.65] 
        elif region_str == "central_europe":
            tts = [9]     
            lr_list = [0.27]
            dw_list = [0.65]
            decay_steps = [40]
            reg_list = [0.00086]
            if nlag == 0:
                dw_list = [0.71] 
                decay_steps = [40]
                reg_list = [0.00088] 
            if nlag == 2:
                dw_list = [0.67] 
            if nlag == 3:
                dw_list = [0.68] 
            if no_sm:
                dw_list = [0.64]
        elif region_str == "northeastern_europe":
            tts = [22]   
            lr_list = [0.20]
            dw_list = [0.52]
            decay_steps = [40]
            reg_list = [0.006] 
            if nlag == 0:
                dw_list = [0.59] 
                reg_list = [0.002] 
                decay_steps = [45]
                lr_list = [0.25]
            if nlag == 3:
                dw_list = [0.45] 
            if nlag == 7:
                dw_list = [0.38] 
        elif region_str == "northcentral_north_america":
            tts = [5]
            lr_list = [0.25]
            dw_list = [0.60]
            decay_steps = [40]
            reg_list = [0.0080]
            if nlag == 0:
                dw_list = [0.40] 
            if nlag == 2:
                dw_list = [0.48]
            if nlag == 3:
                dw_list = [0.43] 
            if nlag in [7, 14]:
                dw_list = [0.45] 
        elif region_str == "southwestern_australia":
            tts = [6]       
            lr_list = [0.2]
            dw_list = [0.29] 
            decay_steps = [25]
            reg_list = [0.00096]
            if nlag == 0:
                dw_list = [0.30]
        elif region_str == "southeastern_australia":
            tts = [20]    
            lr_list = [0.25]
            dw_list = [0.60] 
            decay_steps = [30]
            reg_list = [0.00091]
        elif region_str == "southeastern_africa":
            tts = [0]      
            lr_list = [0.2]
            dw_list = [0.20] 
            reg_list = [0.00100]
            decay_steps = [25]
        elif region_str == "southwestern_africa":
            tts = [1]
            lr_list = [0.2]
            dw_list = [0.28] 
            decay_steps = [25]
            reg_list = [0.00091] 
        elif region_str == "southsouthern_south_america":
            tts = [27]     
            lr_list = [0.20]
            dw_list = [0.58]
            decay_steps = [25]
            reg_list = [0.00085]
        elif region_str == "northsouthern_south_america":
            tts = [25]           
            lr_list = [0.25]
            dw_list = [0.75] 
            decay_steps = [25]
            reg_list = [0.00093] 
        elif region_str == "west_texas":
            tts = [4]       
            lr_list = [0.30] 
            dw_list = [0.44] 
            decay_steps = [40] 
            reg_list = [0.00095] 
        elif region_str == "east_texas":
            tts = [4]       
            lr_list = [0.25] 
            dw_list = [0.47] 
            decay_steps = [40] 
            reg_list = [0.00095] 
          
        
    else: 
        
        
        if region_str == "eastern_europe":
            tts = [25]
            lr_list = [0.25]
            dw_list = [0.75]
            decay_steps = [20]
        elif region_str == "southeastern_north_america":
            tts = [32]
            lr_list = [0.25]
            dw_list = [0.6]
            decay_steps = [20]
        elif region_str == "southcentral_north_america":
            tts = [4]
            lr_list = [0.25]
            dw_list = [0.50]
            decay_steps = [30]
        elif region_str == "southeastern_asia":
            tts = [24]
            lr_list = [0.25]
            dw_list = [0.8]
            decay_steps = [25]
        elif region_str == "northeastern_asia":
            tts = [10]
            lr_list = [0.5]
            dw_list = [0.80]
            decay_steps = [15]
        elif region_str == "southwestern_europe":
            tts = [15]
            lr_list = [0.2]
            dw_list = [0.45] 
            decay_steps = [25]
        elif region_str == "western_europe":
            tts = [34]
            lr_list = [0.25]
            dw_list = [0.7] 
            decay_steps = [25]
        elif region_str == "central_europe":
            tts = [9]
            lr_list = [0.3]
            dw_list = [0.7] 
            decay_steps = [25]
        elif region_str == "northeastern_europe":
            tts = [22]
            lr_list = [0.3]
            dw_list = [0.75] 
            decay_steps = [15]
        elif region_str == "northcentral_north_america":
            tts = [5]
            lr_list = [0.4]
            dw_list = [0.3]
            decay_steps = [35]
        elif region_str == "southwestern_australia":
            tts = [6]
            lr_list = [0.2]
            dw_list = [0.35] 
            decay_steps = [30]
        elif region_str == "southeastern_australia":
            tts = [20]
            lr_list = [0.2]
            dw_list = [0.5] 
            decay_steps = [25]
        elif region_str == "south_africa":
            tts = [0]
            lr_list = [0.25]
            dw_list = [0.65]
            decay_steps = [30]
        elif region_str == "southeastern_africa":
            tts = [0]      
            lr_list = [0.25]
            dw_list = [0.6] 
            decay_steps = [30]
        elif region_str == "southwestern_africa":
            tts = [1]
            lr_list = [0.25]
            dw_list = [0.6] 
            decay_steps = [40]
        elif region_str == "southsouthern_south_america":
            tts = [27]
            lr_list = [0.25]
            dw_list = [0.65] 
            decay_steps = [25]
        elif region_str == "northsouthern_south_america":
            tts = [25]
            lr_list = [0.2]
            dw_list = [0.65]
            decay_steps = [40]
        
    return tts, activation_list, optimizer_list, lr_list, dw_list, decay_list, decay_steps, decay_rate, curr_batch, nepochs, reg_list




###############################################################3



def load_NCEP_region_cnn_hyperparams(region_str, shuffle_sm, no_sm, nlag=None):
    activation_list = ['sigmoid']
    optimizer_list = ['RMSprop']
    reg_list = [0.00095]
    nepochs = 10000
    decay_rate = 0.5
    decay_list = [True]
    curr_batch = 512
    
    if not shuffle_sm:
        if region_str == "eastern_europe":
            tts = [12] 
            lr_list = [0.35]
            dw_list = [0.65] 
            decay_steps = [35]
            reg_list = [0.00095]
            if nlag == 1:     
                lr_list = [0.30] 
                decay_steps = [35]
                reg_list = [0.00099]
                dw_list = [0.71] 
            if nlag == 0:
                dw_list = [0.55] 
        elif region_str == "southeastern_north_america":
            tts = [32]
            lr_list = [0.40] 
            dw_list = [0.52] 
            decay_steps = [30]
            reg_list = [0.00095]
            if nlag == 0:
                dw_list = [0.60] 
            if nlag == 1:
                tts = [21]
                dw_list = [0.60] 
        elif region_str == "southcentral_north_america":
            tts = [4]       
            lr_list = [0.35] 
            dw_list = [0.60] 
            decay_steps = [40]
            reg_list = [0.00095]
            if nlag in [1, 14]:
                dw_list = [0.55] 
            if nlag == 7:
                dw_list = [0.45]
            if nlag == 30:
                dw_list = [0.50]
        elif region_str == "southeastern_asia":
            tts = [24]           
            lr_list = [0.23] 
            dw_list = [0.42]
            decay_steps = [40]
            reg_list = [0.00100]
            if nlag == 1:
                tts = [11]
                reg_list = [0.001] 
                dw_list = [0.46]
        elif region_str == "northeastern_asia":
            tts = [10]      
            lr_list = [0.35]
            dw_list = [0.65]
            decay_steps = [40]
            reg_list = [0.001]
            if nlag == 1:
                tts = [33]
                lr_list = [0.35]
                reg_list = [0.0035] 
                dw_list = [0.70] 
            if nlag == 0:
                dw_list = [0.56] 
            if nlag == 3:
                dw_list = [0.45]
            if nlag == 14:
                dw_list = [0.70] 
            if nlag == 30:
                dw_list = [0.55] 
        elif region_str == "southwestern_europe":
            tts = [8] 
            lr_list = [0.30]
            dw_list = [0.16] 
            decay_steps = [45]
            reg_list = [0.00081] 
            if nlag == 0:
                tts = [9]
                lr_list = [0.35]
                decay_steps = [45]
                reg_list = [0.004]
                dw_list = [0.24] 
            if nlag == 1:
                tts = [9] 
                lr_list = [0.35] 
                decay_steps = [45] 
                reg_list = [0.004] 
                dw_list = [0.28] 
            if nlag in [2, 7]:
                tts = [50]
                reg_list = [0.0035] 
            if nlag == 3:
                tts = [50]
                reg_list = [0.001] 
            if nlag in [14, 30]:
                tts = [50]
                reg_list = [0.005]
            if no_sm:
                tts = [50]
                reg_list = [0.0050] 
                decay_steps = [50]
        elif region_str == "western_europe":
            tts = [34]    
            lr_list = [0.25] 
            dw_list = [0.70] 
            decay_steps = [40]
            reg_list = [0.00087]
            if no_sm:
                dw_list = [0.72]
            if nlag == 7:
                dw_list = [0.65] 
            if nlag == 1:
                tts = [5] 
                reg_list = [0.00088] 
                dw_list = [0.71] 
        elif region_str == "central_europe":
            tts = [9]     
            lr_list = [0.27] 
            dw_list = [0.65]
            decay_steps = [40]
            reg_list = [0.00086]
            if nlag == 0:
                dw_list = [0.71] 
                decay_steps = [40]
                reg_list = [0.00088] 
            if nlag == 1:
                tts = [11] 
                lr_list = [0.26] 
                reg_list = [0.00084] 
                dw_list = [0.64] 
            if nlag == 2:
                dw_list = [0.67] 
            if nlag == 3:
                dw_list = [0.68] 
            if no_sm:
                dw_list = [0.64]
        elif region_str == "northeastern_europe":
            tts = [4]
            lr_list = [0.23] 
            decay_steps = [40] 
            reg_list = [0.0020]
            dw_list = [0.50] 
            if nlag == 0:
                tts = [22]
                lr_list = [0.25] 
                decay_steps = [45] 
                reg_list = [0.002] 
                dw_list = [0.59]
            if nlag == 3:
                tts = [22]   
                lr_list = [0.20] 
                decay_steps = [40]
                reg_list = [0.006]
                dw_list = [0.50] 
            if nlag in [14, 30]:
                dw_list = [0.40] 
            if no_sm:
                tts = [22]   
                lr_list = [0.20] 
                decay_steps = [40]
                reg_list = [0.006]
                dw_list = [0.52] 
        elif region_str == "northcentral_north_america":
            tts = [5]
            lr_list = [0.25] 
            dw_list = [0.60] 
            decay_steps = [40]
            reg_list = [0.0080]
            if nlag in [3, 7, 14]:
                dw_list = [0.45] 
            if nlag == 1:
                reg_list = [0.01] 
                dw_list = [0.54] 
        elif region_str == "southwestern_australia":
            tts = [3]        
            lr_list = [0.2]
            dw_list = [0.27]
            decay_steps = [25]
            reg_list = [0.00090] 
            if nlag == 0:
                dw_list = [0.29] 
            if no_sm:
                tts = [6]       
                lr_list = [0.2]
                dw_list = [0.29] 
                decay_steps = [25]
                reg_list = [0.00096]
        elif region_str == "southeastern_australia":
            tts = [20]    
            lr_list = [0.25]
            dw_list = [0.60]  
            decay_steps = [30]
            reg_list = [0.00091]
            if nlag == 0:
                tts = [3]
                reg_list = [0.00500]
                dw_list = [0.64]
            if nlag == 1:
                tts = [3]
                reg_list = [0.00500] 
                dw_list = [0.59] 
        elif region_str == "southeastern_africa":
            tts = [0]      
            lr_list = [0.2]
            dw_list = [0.20] 
            reg_list = [0.00100]
            decay_steps = [25]
            if nlag == 0:
                dw_list = [0.18] 
            if nlag == 1:
                tts = [11]
                reg_list = [0.00300] 
                dw_list = [0.19] 
        elif region_str == "southwestern_africa":
            tts = [1]
            lr_list = [0.2]
            dw_list = [0.28] 
            decay_steps = [25]
            reg_list = [0.00091]
            if nlag == 0:
                dw_list = [0.23] 
            if nlag == 1:
                tts = [10] 
                lr_list = [0.25]
                dw_list = [0.27]
                decay_steps = [30] 
                reg_list = [0.00057] 
        elif region_str == "southsouthern_south_america":
            tts = [27]     
            lr_list = [0.20]
            dw_list = [0.58] 
            decay_steps = [25]
            reg_list = [0.00085]
            if nlag == 0:
                tts = [17] 
                dw_list = [0.75] 
                reg_list = [0.0011]
                dw_list = [0.45] 
            if nlag == 1:
                tts = [17] 
                dw_list = [0.75] 
                reg_list = [0.0011] 
        elif region_str == "northsouthern_south_america":
            tts = [25]           
            lr_list = [0.25]
            dw_list = [0.75] 
            decay_steps = [25]
            reg_list = [0.00093] 
            if nlag == 1:
                tts = [42] 
                lr_list = [0.30] 
                dw_list = [0.67] 
                reg_list = [0.00016] 
            if nlag == 14:
                dw_list = [0.45]
            if nlag == 30:
                dw_list = [0.50] 
        elif region_str == "west_texas":
            tts = [4]       
            lr_list = [0.30] 
            dw_list = [0.44] 
            decay_steps = [40]
            reg_list = [0.00095] 
        elif region_str == "east_texas":
            tts = [4]       
            lr_list = [0.25] 
            dw_list = [0.47] 
            decay_steps = [40]
            reg_list = [0.00095] 
        
    else: 
        
        if region_str == "eastern_europe":
            tts = [25]
            lr_list = [0.25]
            dw_list = [0.75]
            decay_steps = [20]
        elif region_str == "southeastern_north_america":
            tts = [32]
            lr_list = [0.25]
            dw_list = [0.6]
            decay_steps = [20]
        elif region_str == "southcentral_north_america":
            tts = [4]
            lr_list = [0.25]
            dw_list = [0.50]
            decay_steps = [30]
        elif region_str == "southeastern_asia":
            tts = [24]
            lr_list = [0.25]
            dw_list = [0.8]
            decay_steps = [25]
        elif region_str == "northeastern_asia":
            tts = [10]
            lr_list = [0.5]
            dw_list = [0.80]
            decay_steps = [15]
        elif region_str == "southwestern_europe":
            tts = [15]
            lr_list = [0.2]
            dw_list = [0.45] 
            decay_steps = [25]
        elif region_str == "western_europe":
            tts = [34]
            lr_list = [0.25]
            dw_list = [0.7] 
            decay_steps = [25]
        elif region_str == "central_europe":
            tts = [9]
            lr_list = [0.3]
            dw_list = [0.7] 
            decay_steps = [25]
        elif region_str == "northeastern_europe":
            tts = [22]
            lr_list = [0.3]
            dw_list = [0.75] 
            decay_steps = [15]
        elif region_str == "northcentral_north_america":
            tts = [5]
            lr_list = [0.4]
            dw_list = [0.3]
            decay_steps = [35]
        elif region_str == "southwestern_australia":
            tts = [6]
            lr_list = [0.2]
            dw_list = [0.35] 
            decay_steps = [30]
        elif region_str == "southeastern_australia":
            tts = [20]
            lr_list = [0.2]
            dw_list = [0.5] 
            decay_steps = [25]
        elif region_str == "south_africa":
            tts = [0]
            lr_list = [0.25]
            dw_list = [0.65]
            decay_steps = [30]
        elif region_str == "southeastern_africa":
            tts = [0]      
            lr_list = [0.25]
            dw_list = [0.6] 
            decay_steps = [30]
        elif region_str == "southwestern_africa":
            tts = [1]
            lr_list = [0.25]
            dw_list = [0.6] 
            decay_steps = [40]
        elif region_str == "southsouthern_south_america":
            tts = [27]
            lr_list = [0.25]
            dw_list = [0.65] 
            decay_steps = [25]
        elif region_str == "northsouthern_south_america":
            tts = [25]
            lr_list = [0.2]
            dw_list = [0.65]
            decay_steps = [40]
                
    return tts, activation_list, optimizer_list, lr_list, dw_list, decay_list, decay_steps, decay_rate, curr_batch, nepochs, reg_list


##################################################################


def load_region_constants(region_str):
    if region_str == "eastern_europe":
        hem = "north"
        region_input_lat_bbox = param.eastern_europe_input_lat_bbox
        region_input_lon_bbox = param.eastern_europe_input_lon_bbox
        region_box_x = param.eastern_europe_box_x
        region_box_y = param.eastern_europe_box_y
        region_lat = param.eastern_europe_lat
        region_lon = param.eastern_europe_lon
        region_lon_EW = param.eastern_europe_lon_EW
        region_t62_lats = param.eastern_europe_input_t62_lats
        region_t62_lons = param.eastern_europe_input_t62_lons
    if region_str == "southeastern_north_america":
        hem = "north"
        region_input_lat_bbox = param.southeastern_north_america_input_lat_bbox
        region_input_lon_bbox = param.southeastern_north_america_input_lon_bbox
        region_box_x = param.southeastern_north_america_box_x
        region_box_y = param.southeastern_north_america_box_y
        region_lat = param.southeastern_north_america_lat
        region_lon = param.southeastern_north_america_lon
        region_lon_EW = param.southeastern_north_america_lon_EW
        region_t62_lats = param.southeastern_north_america_input_t62_lats
        region_t62_lons = param.southeastern_north_america_input_t62_lons
    elif region_str == "southcentral_north_america":
        hem = "north"
        region_input_lat_bbox = param.southcentral_north_america_input_lat_bbox
        region_input_lon_bbox = param.southcentral_north_america_input_lon_bbox
        region_box_x = param.southcentral_north_america_box_x
        region_box_y = param.southcentral_north_america_box_y
        region_lat = param.southcentral_north_america_lat
        region_lon = param.southcentral_north_america_lon
        region_lon_EW = param.southcentral_north_america_lon_EW
        region_t62_lats = param.southcentral_north_america_input_t62_lats
        region_t62_lons = param.southcentral_north_america_input_t62_lons
    elif region_str == "southeastern_asia":
        hem = "north"
        region_input_lat_bbox = param.southeastern_asia_input_lat_bbox
        region_input_lon_bbox = param.southeastern_asia_input_lon_bbox
        region_box_x = param.southeastern_asia_box_x
        region_box_y = param.southeastern_asia_box_y
        region_lat = param.southeastern_asia_lat
        region_lon = param.southeastern_asia_lon
        region_lon_EW = param.southeastern_asia_lon_EW
        region_t62_lats = param.southeastern_asia_input_t62_lats
        region_t62_lons = param.southeastern_asia_input_t62_lons
    elif region_str == "northeastern_asia":
        hem = "north"
        region_input_lat_bbox = param.northeastern_asia_input_lat_bbox
        region_input_lon_bbox = param.northeastern_asia_input_lon_bbox
        region_box_x = param.northeastern_asia_box_x
        region_box_y = param.northeastern_asia_box_y
        region_lat = param.northeastern_asia_lat
        region_lon = param.northeastern_asia_lon
        region_lon_EW = param.northeastern_asia_lon_EW
        region_t62_lats = param.northeastern_asia_input_t62_lats
        region_t62_lons = param.northeastern_asia_input_t62_lons
    elif region_str == "southwestern_europe":
        hem = "north"
        region_input_lat_bbox = param.southwestern_europe_input_lat_bbox
        region_input_lon_bbox = param.southwestern_europe_input_lon_bbox
        region_box_x = param.southwestern_europe_box_x
        region_box_y = param.southwestern_europe_box_y
        region_lat = param.southwestern_europe_lat
        region_lon = param.southwestern_europe_lon
        region_lon_EW = param.southwestern_europe_lon_EW
        region_t62_lats = param.southwestern_europe_input_t62_lats
        region_t62_lons = param.southwestern_europe_input_t62_lons
    elif region_str == "western_europe":
        hem = "north"
        region_input_lat_bbox = param.western_europe_input_lat_bbox
        region_input_lon_bbox = param.western_europe_input_lon_bbox
        region_box_x = param.western_europe_box_x
        region_box_y = param.western_europe_box_y
        region_lat = param.western_europe_lat
        region_lon = param.western_europe_lon
        region_lon_EW = param.western_europe_lon_EW
        region_t62_lats = param.western_europe_input_t62_lats
        region_t62_lons = param.western_europe_input_t62_lons
    elif region_str == "central_europe":
        hem = "north"
        region_input_lat_bbox = param.central_europe_input_lat_bbox
        region_input_lon_bbox = param.central_europe_input_lon_bbox
        region_box_x = param.central_europe_box_x
        region_box_y = param.central_europe_box_y
        region_lat = param.central_europe_lat
        region_lon = param.central_europe_lon
        region_lon_EW = param.central_europe_lon_EW
        region_t62_lats = param.central_europe_input_t62_lats
        region_t62_lons = param.central_europe_input_t62_lons
    elif region_str == "northeastern_europe":
        hem = "north"
        region_input_lat_bbox = param.northeastern_europe_input_lat_bbox
        region_input_lon_bbox = param.northeastern_europe_input_lon_bbox
        region_box_x = param.northeastern_europe_box_x
        region_box_y = param.northeastern_europe_box_y
        region_lat = param.northeastern_europe_lat
        region_lon = param.northeastern_europe_lon
        region_lon_EW = param.northeastern_europe_lon_EW
        region_t62_lats = param.northeastern_europe_input_t62_lats
        region_t62_lons = param.northeastern_europe_input_t62_lons
    elif region_str == "northcentral_north_america":
        hem = "north"
        region_input_lat_bbox = param.northcentral_north_america_input_lat_bbox
        region_input_lon_bbox = param.northcentral_north_america_input_lon_bbox
        region_box_x = param.northcentral_north_america_box_x
        region_box_y = param.northcentral_north_america_box_y
        region_lat = param.northcentral_north_america_lat
        region_lon = param.northcentral_north_america_lon
        region_lon_EW = param.northcentral_north_america_lon_EW
        region_t62_lats = param.northcentral_north_america_input_t62_lats
        region_t62_lons = param.northcentral_north_america_input_t62_lons
    elif region_str == "southwestern_australia":
        region_abbrev = "WAu"
        region_input_lat_bbox = param.southwestern_australia_input_lat_bbox
        region_input_lon_bbox = param.southwestern_australia_input_lon_bbox
        region_box_x = param.southwestern_australia_box_x
        region_box_y = param.southwestern_australia_box_y
        region_lat = param.southwestern_australia_lat
        region_lon = param.southwestern_australia_lon
        region_lon_EW = param.southwestern_australia_lon_EW
        region_t62_lats = param.southwestern_australia_input_t62_lats
        region_t62_lons = param.southwestern_australia_input_t62_lons
    elif region_str == "southeastern_australia":
        region_abbrev = "EAu"
        region_input_lat_bbox = param.southeastern_australia_input_lat_bbox
        region_input_lon_bbox = param.southeastern_australia_input_lon_bbox
        region_box_x = param.southeastern_australia_box_x
        region_box_y = param.southeastern_australia_box_y
        region_lat = param.southeastern_australia_lat
        region_lon = param.southeastern_australia_lon
        region_lon_EW = param.southeastern_australia_lon_EW
        region_t62_lats = param.southeastern_australia_input_t62_lats
        region_t62_lons = param.southeastern_australia_input_t62_lons
    elif region_str == "southeastern_africa":
        region_abbrev = "SEAf"
        region_input_lat_bbox = param.southeastern_africa_input_lat_bbox
        region_input_lon_bbox = param.southeastern_africa_input_lon_bbox
        region_box_x = param.southeastern_africa_box_x
        region_box_y = param.southeastern_africa_box_y
        region_lat = param.southeastern_africa_lat
        region_lon = param.southeastern_africa_lon
        region_lon_EW = param.southeastern_africa_lon_EW
        region_t62_lats = param.southeastern_africa_input_t62_lats
        region_t62_lons = param.southeastern_africa_input_t62_lons
    elif region_str == "southwestern_africa":
        region_abbrev = "SWAf"
        region_input_lat_bbox = param.southwestern_africa_input_lat_bbox
        region_input_lon_bbox = param.southwestern_africa_input_lon_bbox
        region_box_x = param.southwestern_africa_box_x
        region_box_y = param.southwestern_africa_box_y
        region_lat = param.southwestern_africa_lat
        region_lon = param.southwestern_africa_lon
        region_lon_EW = param.southwestern_africa_lon_EW
        region_t62_lats = param.southwestern_africa_input_t62_lats
        region_t62_lons = param.southwestern_africa_input_t62_lons
    elif region_str == "southsouthern_south_america":
        region_abbrev = "SArg"
        region_input_lat_bbox = param.southsouthern_south_america_input_lat_bbox
        region_input_lon_bbox = param.southsouthern_south_america_input_lon_bbox
        region_box_x = param.southsouthern_south_america_box_x
        region_box_y = param.southsouthern_south_america_box_y
        region_lat = param.southsouthern_south_america_lat
        region_lon = param.southsouthern_south_america_lon
        region_lon_EW = param.southsouthern_south_america_lon_EW
        region_t62_lats = param.southsouthern_south_america_input_t62_lats
        region_t62_lons = param.southsouthern_south_america_input_t62_lons
    elif region_str == "northsouthern_south_america":
        region_abbrev = "CArg"
        region_input_lat_bbox = param.northsouthern_south_america_input_lat_bbox
        region_input_lon_bbox = param.northsouthern_south_america_input_lon_bbox
        region_box_x = param.northsouthern_south_america_box_x
        region_box_y = param.northsouthern_south_america_box_y
        region_lat = param.northsouthern_south_america_lat
        region_lon = param.northsouthern_south_america_lon
        region_lon_EW = param.northsouthern_south_america_lon_EW
        region_t62_lats = param.northsouthern_south_america_input_t62_lats
        region_t62_lons = param.northsouthern_south_america_input_t62_lons
    elif region_str == "west_texas":
        hem = "north"
        region_input_lat_bbox = param.west_texas_input_lat_bbox
        region_input_lon_bbox = param.west_texas_input_lon_bbox
        region_box_x = param.west_texas_box_x
        region_box_y = param.west_texas_box_y
        region_lat = param.west_texas_lat
        region_lon = param.west_texas_lon
        region_lon_EW = param.west_texas_lon_EW
        region_t62_lats = param.west_texas_input_t62_lats
        region_t62_lons = param.west_texas_input_t62_lons
    elif region_str == "east_texas":
        hem = "north"
        region_input_lat_bbox = param.east_texas_input_lat_bbox
        region_input_lon_bbox = param.east_texas_input_lon_bbox
        region_box_x = param.east_texas_box_x
        region_box_y = param.east_texas_box_y
        region_lat = param.east_texas_lat
        region_lon = param.east_texas_lon
        region_lon_EW = param.east_texas_lon_EW
        region_t62_lats = param.east_texas_input_t62_lats
        region_t62_lons = param.east_texas_input_t62_lons


    return hem, region_input_lat_bbox, region_input_lon_bbox, region_box_x, region_box_y, region_lat, region_lon, region_lon_EW, region_t62_lats, region_t62_lons