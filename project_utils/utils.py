import numpy as np
import pandas as pd
import xarray as xr
from project_utils import parameters as param

## ----------------------------------------------------------------------------

def train_test_unseen_split_by_years(yrs_train, yrs_test, yrs_unseen, x_dat_daily, y_dat, ind, sample_weights, caldays, time_vec):
    
    first = True
    for y in yrs_train:
        strt = time_vec[time_vec.dt.year == y].index.values[0]
        stp = time_vec[time_vec.dt.year == y].index.values[-1]+1
        if first:
            x_train = x_dat_daily[strt:stp,:,:,:]
            y_train = y_dat[strt:stp]
            ind_train = ind[strt:stp]
            sweight_train = sample_weights[strt:stp]
            cday_train = caldays[strt:stp]
            time_train = time_vec[strt:stp].values
            first = False
        else:
            x_train = np.concatenate((x_train, x_dat_daily[strt:stp,:,:,:]), axis=0)
            y_train = np.concatenate((y_train, y_dat[strt:stp]), axis=0)
            ind_train = np.concatenate((ind_train, ind[strt:stp]), axis=0)
            sweight_train = np.concatenate((sweight_train, sample_weights[strt:stp]), axis=0)
            cday_train = np.concatenate((cday_train, caldays[strt:stp]), axis=0)
            time_train = np.concatenate((time_train, time_vec[strt:stp].values), axis=0)

    first = True
    for y in yrs_test:
        strt = time_vec[time_vec.dt.year == y].index.values[0]
        stp = time_vec[time_vec.dt.year == y].index.values[-1]+1
        if first:
            x_test = x_dat_daily[strt:stp,:,:,:]
            y_test = y_dat[strt:stp]
            ind_test = ind[strt:stp]
            sweight_test = sample_weights[strt:stp]
            cday_test = caldays[strt:stp]
            time_test = time_vec[strt:stp].values
            first = False
        else:
            x_test = np.concatenate((x_test, x_dat_daily[strt:stp,:,:,:]), axis=0)
            y_test = np.concatenate((y_test, y_dat[strt:stp]), axis=0)
            ind_test = np.concatenate((ind_test, ind[strt:stp]), axis=0)
            sweight_test = np.concatenate((sweight_test, sample_weights[strt:stp]), axis=0)
            cday_test = np.concatenate((cday_test, caldays[strt:stp]), axis=0)
            time_test = np.concatenate((time_test, time_vec[strt:stp].values), axis=0)
            
    
    first = True
    for y in yrs_unseen:
        strt = time_vec[time_vec.dt.year == y].index.values[0]
        stp = time_vec[time_vec.dt.year == y].index.values[-1]+1
        if first:
            x_unseen = x_dat_daily[strt:stp,:,:,:]
            y_unseen = y_dat[strt:stp]
            ind_unseen = ind[strt:stp]
            sweight_unseen = sample_weights[strt:stp]
            cday_unseen = caldays[strt:stp]
            time_unseen = time_vec[strt:stp].values
            first = False
        else:
            x_unseen = np.concatenate((x_unseen, x_dat_daily[strt:stp,:,:,:]), axis=0)
            y_unseen = np.concatenate((y_unseen, y_dat[strt:stp]), axis=0)
            ind_unseen = np.concatenate((ind_unseen, ind[strt:stp]), axis=0)
            sweight_unseen = np.concatenate((sweight_unseen, sample_weights[strt:stp]), axis=0)
            cday_unseen = np.concatenate((cday_unseen, caldays[strt:stp]), axis=0)
            time_unseen = np.concatenate((time_unseen, time_vec[strt:stp].values), axis=0)


    return x_train, x_test, x_unseen, y_train, y_test, y_unseen, ind_train, ind_test, ind_unseen, sweight_train, sweight_test, sweight_unseen, cday_train, cday_test, cday_unseen, time_train, time_test, time_unseen

## ----------------------------------------------------------------------------

import statsmodels.api as sm

def fit_ols(x, y, constant = True):
    
    if constant is not False:
        x = sm.add_constant(x)
        
    model = sm.OLS(y, x).fit()
    
    return [model.params[0], model.params[1], model.pvalues[1]]

