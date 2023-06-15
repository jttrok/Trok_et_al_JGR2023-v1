proj_dir='/path/to/main_project_folder/' # edit this line ((Also on line 257 !))

import numpy as np
import random
import xarray as xr
import time
import h5py
import pandas as pd
from denseweight import DenseWeight
from sklearn.model_selection import train_test_split
from tensorflow.random import set_seed as tf_set_seed
import matplotlib.pyplot as plt
from scipy.stats import loguniform
#from tensorflow.keras import optimizers
from tensorflow.compat.v1.keras import optimizers
import tensorflow.compat.v1 as tf_compat_v1
from tensorflow.keras.models import load_model as tf_load_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from matplotlib import colors
import multiprocessing as mp
import sys
sys.path.append(proj_dir)
from project_utils import parameters as param
from project_utils import load_region
from project_utils import prepare_inputs
from project_utils import utils as util
from project_utils import model_utils as mu
import importlib
importlib.reload(param)
importlib.reload(prepare_inputs)
importlib.reload(util)
importlib.reload(mu)
importlib.reload(load_region)

np.random.seed(101)
random.seed(201)
tf_set_seed(333)
session_conf = tf_compat_v1.ConfigProto(device_count={'CPU': 24})
sess = tf_compat_v1.Session(config=session_conf)


#####################################################################
def model_skill(region_str,
                nlag,
                tmax_predictions_unseen,
                tmax_predictions_train,
                tmax_predictions_test,
                y_unseen,
                y_train,
                y_test,
                sweight_unseen,
                sweight_train,
                sweight_test,
                dset):
    
    fig, axes = plt.subplots(3,3, figsize=(18,15))
    ax = axes.flatten()

    _, bins, _ = ax[0].hist(tmax_predictions_unseen, bins=30, 
                            range=[np.floor(np.min(y_unseen)-5), np.ceil(np.max(y_unseen))+5], 
                            label="pred tmax", alpha=0.6, density=True)

    ax[0].hist(y_unseen, bins=bins, label="true tmax", alpha=0.6, density=True)
    ax[0].legend()
    ax[0].set_xlabel("tmax (K)")
    ax[0].set_ylabel("count")
    ax[0].set_xlim([np.floor(np.min(y_unseen))-5, np.ceil(np.max(y_unseen))+5])
    ax[0].set_title("both")

    ax[1].hist(tmax_predictions_train, bins=bins, label="pred tmax", alpha=0.6, density=True)
    ax[1].hist(y_train, bins=bins, label="true tmax", alpha=0.6, density=True)
    ax[1].legend()
    ax[1].set_xlabel("tmax (K)")
    ax[1].set_ylabel("count")
    ax[1].set_xlim([np.floor(np.min(y_unseen))-5, np.ceil(np.max(y_unseen))+5])
    ax[1].set_title("train")

    ax[2].hist(tmax_predictions_test, bins=bins, label="pred tmax", alpha=0.6, density=True)
    ax[2].hist(y_test, bins=bins, label="true tmax", alpha=0.6, density=True)
    ax[2].legend()
    ax[2].set_xlabel("tmax (K)")
    ax[2].set_ylabel("count")
    ax[2].set_xlim([np.floor(np.min(y_unseen))-5, np.ceil(np.max(y_unseen))+5])
    ax[2].set_title("test")

    c_map = 'twilight_shifted'
    nbins = 50

    ax[3].scatter(y_unseen, tmax_predictions_unseen, s=0.5, alpha=0.5)
    p = ax[3].hist2d(y_unseen, tmax_predictions_unseen, bins = nbins, cmin = 10, 
                 zorder = 5, cmap = c_map) #norm = colors.LogNorm(), 
    x_l, x_r = ax[3].get_xlim()
    ax[3].set_ylim([x_l,x_r])
    y_b, y_t = ax[3].get_ylim()
    ax[3].plot(np.arange(x_l,x_r, 0.5), np.arange(x_l,x_r, 0.5), 'r', zorder = 10)
    ax[3].set_xlabel("true tmax (K)")
    ax[3].set_ylabel("predicted tmax (K)")
    ax[3].annotate("R-squared = {:.3f}".format(r2_score(y_unseen, tmax_predictions_unseen)), (x_l+1, y_t-2))
    ax[3].annotate("MAE = {:.3f}".format(mean_absolute_error(y_unseen, tmax_predictions_unseen)), (x_l+1, y_t-4))
    ax[3].annotate("MAEw = {:.3f}".format(mean_absolute_error(y_unseen, tmax_predictions_unseen, sample_weight=sweight_unseen)), (x_l+1, y_t-6))
    ax[3].annotate("MSE = {:.3f}".format(mean_squared_error(y_unseen, tmax_predictions_unseen)), (x_l+1, y_t-8))
    ax[3].annotate("MSEw = {:.3f}".format(mean_squared_error(y_unseen, tmax_predictions_unseen, sample_weight=sweight_unseen)), (x_l+1, y_t-10))
    ax[3].annotate("max_pred = {:.1f}".format(np.max(tmax_predictions_unseen)), (x_r-20, y_b+6))
    ax[3].annotate("min_pred = {:.1f}".format(np.min(tmax_predictions_unseen)), (x_r-20, y_b+4))
    cbar = plt.colorbar(p[3], label = "number of days in bin", ax=ax[3]);

    ax[4].scatter(y_train, tmax_predictions_train, s=0.5, alpha=0.5)
    p = ax[4].hist2d(y_train, tmax_predictions_train, bins = nbins, cmin = 10, 
                 zorder = 5, cmap = c_map)
    x_l, x_r = ax[4].get_xlim()
    ax[4].set_ylim([x_l,x_r])
    y_b, y_t = ax[4].get_ylim()
    ax[4].plot(np.arange(x_l,x_r, 0.5), np.arange(x_l,x_r, 0.5), 'r', zorder = 10)
    ax[4].set_xlabel("true tmax (K)")
    ax[4].set_ylabel("predicted tmax (K)")
    ax[4].annotate("R-squared = {:.3f}".format(r2_score(y_train, tmax_predictions_train)), (x_l+1, y_t-2))
    ax[4].annotate("MAE = {:.3f}".format(mean_absolute_error(y_train, tmax_predictions_train)), (x_l+1, y_t-4))
    ax[4].annotate("MAEw = {:.3f}".format(mean_absolute_error(y_train, tmax_predictions_train, sample_weight=sweight_train)), (x_l+1, y_t-6))
    ax[4].annotate("MSE = {:.3f}".format(mean_squared_error(y_train, tmax_predictions_train)), (x_l+1, y_t-8))
    ax[4].annotate("MSEw = {:.3f}".format(mean_squared_error(y_train, tmax_predictions_train, sample_weight=sweight_train)), (x_l+1, y_t-10))
    ax[4].annotate("max_pred = {:.1f}".format(np.max(tmax_predictions_train)), (x_r-20, y_b+6))
    ax[4].annotate("min_pred = {:.1f}".format(np.min(tmax_predictions_train)), (x_r-20, y_b+4))
    cbar = plt.colorbar(p[3], label = "number of days in bin", ax=ax[4]);

    ax[5].scatter(y_test, tmax_predictions_test, s=0.5, alpha=0.5)
    p = ax[5].hist2d(y_test, tmax_predictions_test, bins = nbins, cmin = 10, 
                 zorder = 5, cmap = c_map)
    x_l, x_r = ax[5].get_xlim()
    ax[5].set_ylim([x_l,x_r])
    y_b, y_t = ax[5].get_ylim()
    ax[5].plot(np.arange(x_l,x_r, 0.5), np.arange(x_l,x_r, 0.5), 'r', zorder = 10)
    ax[5].set_xlabel("true tmax (K)")
    ax[5].set_ylabel("predicted tmax (K)")
    ax[5].annotate("R-squared = {:.3f}".format(r2_score(y_test, tmax_predictions_test)), (x_l+1, y_t-2))
    ax[5].annotate("MAE = {:.3f}".format(mean_absolute_error(y_test, tmax_predictions_test)), (x_l+1, y_t-4))
    ax[5].annotate("MAEw = {:.3f}".format(mean_absolute_error(y_test, tmax_predictions_test, sample_weight=sweight_test)), (x_l+1, y_t-6))
    ax[5].annotate("MSE = {:.3f}".format(mean_squared_error(y_test, tmax_predictions_test)), (x_l+1, y_t-8))
    ax[5].annotate("MSEw = {:.3f}".format(mean_squared_error(y_test, tmax_predictions_test, sample_weight=sweight_test)), (x_l+1, y_t-10))
    ax[5].annotate("max_pred = {:.1f}".format(np.max(tmax_predictions_test)), (x_r-20, y_b+6))
    ax[5].annotate("min_pred = {:.1f}".format(np.min(tmax_predictions_test)), (x_r-20, y_b+4))
    cbar = plt.colorbar(p[3], label = "number of days in bin", ax=ax[5]);


    hist_, bin_edges = np.histogram(y_unseen, bins=15)
    bin_indices = np.digitize(y_unseen, bins = bin_edges)
    bin_indices_train = np.digitize(y_train, bins = bin_edges)
    bin_indices_test = np.digitize(y_test, bins = bin_edges)

    resids = abs(y_unseen - tmax_predictions_unseen)
    resids_train = abs(y_train - tmax_predictions_train)
    resids_test = abs(y_test - tmax_predictions_test)

    bin_avg_resid = []
    bin_avg_resid_train = []
    bin_avg_resid_test = []

    for jj in range(1, len(bin_edges)):
        bin_avg_resid.append(resids[bin_indices==jj].mean())
        bin_avg_resid_train.append(resids_train[bin_indices_train==jj].mean())
        bin_avg_resid_test.append(resids_test[bin_indices_test==jj].mean())

    bins = bin_edges
    ax[6].bar((bins[1:] + bins[:-1])*0.5, bin_avg_resid, width=np.diff(bins)[0])
    ax[6].set_xlabel("true tmax (K)")
    ax[6].set_ylabel("avg residuals")

    ax[7].bar((bins[1:] + bins[:-1])*0.5, bin_avg_resid_train, width=np.diff(bins)[0])
    ax[7].set_xlabel("true tmax (K)")
    ax[7].set_ylabel("avg residuals")

    ax[8].bar((bins[1:] + bins[:-1])*0.5, bin_avg_resid_test, width=np.diff(bins)[0])
    ax[8].set_xlabel("true tmax (K)")
    ax[8].set_ylabel("avg residuals")

    plt.suptitle(region_str+" (dset= "+dset+")")
    plt.tight_layout()        
    plt.show()
    

#####################################################################
def save_model(region_str,
               nlag,
               model,
               history,
               tmax_predictions,
               ind_test,
               ind_unseen,
               time_vec,
               y_dat,
               shuffle_sm, 
               no_sm,
               dset):
    
    if shuffle_sm:
        model.save_weights("../processed_data_"+dset+"/"+region_str+"/SM_shuff_"+str(shuffle_sm).zfill(2)+
                               "_trained_weights"+"_lag"+str(nlag)+".h5")
    elif no_sm:
        model.save_weights("../processed_data_"+dset+"/"+region_str+"/no_SM_rep_"+str(1).zfill(2)+
                               "_trained_weights"+"_lag"+str(nlag)+".h5")
    else:
        model.save_weights("../processed_data_"+dset+"/"+region_str+"/trained_weights"+"_lag"+str(nlag)+".h5")

    predict_df = pd.DataFrame(tmax_predictions)
    predict_df = predict_df.rename(columns = {0: 'predicted_tmax'})
    predict_df['set'] = 'train'
    predict_df.loc[ind_test, 'set'] = 'test'
    predict_df.loc[ind_unseen, 'set'] = 'unseen'
    predict_df['date'] = time_vec.values
    predict_df['true_y'] = y_dat
    predict_df

    if shuffle_sm:
        predict_df.to_csv("../processed_data_"+dset+"/"+region_str+"/SM_shuff_"+str(shuffle_sm).zfill(2)+
                          "_model_predictions"+"_lag"+str(nlag)+".csv", index = False)
    elif no_sm:
        predict_df.to_csv("../processed_data_"+dset+"/"+region_str+"/no_SM_"+str(1).zfill(2)+
                          "_model_predictions"+"_lag"+str(nlag)+".csv", index = False)
    else:
        predict_df.to_csv("../processed_data_"+dset+"/"+region_str+"/model_predictions"+"_lag"+str(nlag)+".csv", index = False)

    plt.plot(range(len(predict_df['true_y'])), predict_df['true_y'],'b', label='true')
    print('m_true:', np.polyfit(range(len(predict_df['true_y'])), predict_df['true_y'], deg=1)[0]*365, 'deg/yr')
    plt.plot(range(len(predict_df['predicted_tmax'])), predict_df['predicted_tmax'],'r', label='predicted')
    print('m_pred:', np.polyfit(range(len(predict_df['predicted_tmax'])), predict_df['predicted_tmax'], deg=1)[0]*365, 'deg/yr')
    plt.title(region_str)
    plt.legend()
    plt.show()

    
#####################################################################
def save_centered_predictions(region_str, 
                              hgt_layer, sorted_soilw_input_dat, x_vec, 
                              cday, hgt_day, ninputs, shuffle_sm, dset, w=500):
                
        hgt_input = np.tile(hgt_layer, [len(sorted_soilw_input_dat[:,0,0]), 1, 1]) 
        pdp_input = np.stack((hgt_input, sorted_soilw_input_dat[:,:,:]), axis=3)
        cday_input = np.tile(cday, [len(sorted_soilw_input_dat[:,0,0]), 1, 1])
        
        if shuffle_sm:
            model=tf_load_model('../processed_data'+dset+'/'+region_str+'/SM_shuff_temporary_model/parallel_model')
        else:
            model=tf_load_model('../processed_data_'+dset+'/'+region_str+'/temporary_model/parallel_model')
        
        tmax_predictions = pd.DataFrame(model.predict({"stacked_input" : pdp_input, "calday": cday_input}, verbose=0)[:,0])
        
        origin_idx = np.abs(x_vec).argmin()
        centered_predictions = np.array(tmax_predictions) - np.array(tmax_predictions.rolling(w, min_periods=10).mean())[origin_idx,0]
        
        return centered_predictions[:,0] 

        
#####################################################################        
def pdp_by_yr(region_str, 
              yr, nlag, summer_indices, input_dat, 
              hgt_input_dat, sorted_soilw_input_dat, 
              x_vec, caldays, ninputs, shuffle_sm, w, dset):
    
    proj_dir='/path/to/main_project_folder/' # edit this line
    import numpy as np
    import random
    import xarray as xr
    import time
    import h5py
    import pandas as pd
    from denseweight import DenseWeight
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from scipy.stats import loguniform
    #from tensorflow.keras import optimizers
    from tensorflow.compat.v1.keras import optimizers
    import tensorflow.compat.v1 as tf_compat_v1
    from tensorflow.random import set_seed as tf_set_seed
    from tensorflow.keras.models import load_model as tf_load_model
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from matplotlib import colors
    import sys
    sys.path.append(proj_dir)
    from project_utils import parameters as param
    from project_utils import load_region
    from project_utils import prepare_inputs
    from project_utils import utils as util
    from project_utils import model_utils as mu
    import analysis_fxns as fxns
    import importlib
    importlib.reload(fxns)
    importlib.reload(param)
    importlib.reload(prepare_inputs)
    importlib.reload(util)
    importlib.reload(mu)
    importlib.reload(load_region)
    import multiprocessing as mp
    np.random.seed(101)
    random.seed(201)
    tf_set_seed(333)
    session_conf = tf_compat_v1.ConfigProto(device_count={'CPU': 2})
    sess = tf_compat_v1.Session(config=session_conf)
    
    centered_predictions = np.zeros([len(input_dat[:len(summer_indices[str(yr)]),0,0]), len(input_dat[:,0,0])])
    print(np.shape(centered_predictions))
    arglist = []
    for ii, hgt_day in enumerate(summer_indices[str(yr)]):
        centered_predictions[ii,:] = save_centered_predictions(region_str, 
                                                               hgt_input_dat[hgt_day, :, :].reshape(1, len(input_dat[0,:,0,0]),len(input_dat[0,0,:,0])), 
                                                               sorted_soilw_input_dat, x_vec, caldays[hgt_day], 
                                                               hgt_day, ninputs, shuffle_sm, dset=dset, w=w)
    if shuffle_sm:
         np.save("../processed_data_"+dset+"/"+region_str+"/SM_shuff_JJA_pdp/"+region_str
                +"_"+str(shuffle_sm).zfill(2)+"_centered_predictions_pdp_data_JJA_"+str(yr)+"_w"+str(w)+"_nlag"+str(nlag)
                +".npy", centered_predictions)
    else:
        np.save("../processed_data_"+dset+"/"+region_str+"/JJA_pdp/"+region_str+
                "_centered_predictions_pdp_data_JJA_"+str(yr)+"_w"+str(w)+"_nlag"+str(nlag)+
                ".npy", centered_predictions)
    
#####################################################################
def calc_pdps(region_str, hem, nlag, SNOWFREE, years, shuffle_sm, ninputs, w, dset):
    
    ## Read soilw data and sort ##
    soilw_df = pd.read_csv("../processed_data_"+dset+"/"+region_str+"/region_avg_soilw_cday_anomaly.csv")
    soilw_df['time'] = pd.to_datetime(soilw_df['time'])
    time_vec = soilw_df['time']
    print(time_vec)

    if hem == 'south':
        if SNOWFREE:
            print('removing JJA soilw maps')
            soilw_df = soilw_df[(pd.to_datetime(time_vec).dt.month < 6) | (pd.to_datetime(time_vec).dt.month > 8)].iloc[nlag:].reset_index(drop=True)
    elif hem == 'north':
        if SNOWFREE:
            print('removing DJF soilw maps')
            soilw_df = soilw_df[(pd.to_datetime(time_vec).dt.month >= 3) & (pd.to_datetime(time_vec).dt.month <= 11)].reset_index(drop=True)
    time_vec = soilw_df['time']
    print(time_vec)
    sorted_soilw_df = soilw_df.sort_values(by='soilw_daily_anom')

    ## Sort soilw input maps in order of increasing avg soilw anomaly ##
    input_dat, y_dat, ind, caldays, time_vec = prepare_inputs.get_model_inputs(region_str, nlag, SNOWFREE, hemisphere=hem, dset=dset)
    print(time_vec)
    hgt_input_dat = input_dat[:, :, :, 0]
    print(np.shape(hgt_input_dat))
    print(soilw_df)
    sorted_soilw_input_dat = input_dat[:,:,:,1][sorted_soilw_df.index.values, :, :]
    print(np.shape(sorted_soilw_input_dat))

    ## get summer indices ##
    summer_indices = {}
    if hem == 'south':
        for kk, yr in enumerate(years):
            summer_indices[str(yr)] = soilw_df.time[(soilw_df.time.dt.year == yr) & ((soilw_df.time.dt.month <= 2) | (soilw_df.time.dt.month == 12))].index.values
    elif hem == 'north':
        for kk, yr in enumerate(years):
            summer_indices[str(yr)] = soilw_df.time[(soilw_df.time.dt.year == yr) & (soilw_df.time.dt.month >= 6) & (soilw_df.time.dt.month <= 8)].index.values

    np.random.seed(101)
    random.seed(201)
    tf_set_seed(333)

    ## calculate daily PDPs over all summer indices and all years ##
    x_vec = sorted_soilw_df.soilw_daily_anom.values
    
    arglist = []
    for yr in years:
        print(yr)
        arglist.append((region_str, yr, nlag, summer_indices, input_dat, hgt_input_dat, sorted_soilw_input_dat, x_vec, caldays, ninputs, shuffle_sm, w, dset))

    print('beginning pool')
    with mp.get_context("spawn").Pool() as pool:
        pool.starmap(pdp_by_yr, arglist)
    print('end pool')
        
    return x_vec


#####################################################################
def plot_pdp(region_str, hem, nlag, years, x_vec, shuffle_sm, w, dset):
    
    def calc_full_rolling_mean(vec):
        mid_idx = np.median(range(len(vec)))
        last_idx = len(vec)-1
        rolling_mean = np.zeros(np.shape(vec))
        for jj in range(len(vec)):
            if (jj == 0) | (jj == len(vec)-1):
                rolling_mean[jj] = vec[jj]
            elif jj <= mid_idx:
                win_size = jj
                win_size = min(win_size, 500)
                rolling_mean[jj] = np.mean(vec[jj-win_size:jj+win_size+1])
            elif jj > mid_idx:
                win_size = last_idx - jj
                win_size = min(win_size, 500)
                rolling_mean[jj] = np.mean(vec[jj-win_size:jj+win_size+1])

        return rolling_mean


    fig, ax = plt.subplots(1,1,figsize=(15,15))

    if hem == 'north':
        n_summer_days = 92
    elif hem == 'south':
        n_summer_days = 90
    
    for yr in years:
        if (hem=='south') & (yr==1979):
            print('ERROR: skipping 1979 test year due to lack of summer indices from lag issues')
            years.remove(1979)
    
    predictions = np.ones((len(years), n_summer_days, len(x_vec)))
    for kk_yr, yr in enumerate(years):
            print(yr)
            
            if shuffle_sm:
                print('shuffled sm')
                if hem == 'north':
                    predictions[kk_yr,:,:] = np.load("../processed_data_"+dset+"/"+region_str+"/SM_shuff_JJA_pdp/"+region_str+"_"+str(shuffle_sm).zfill(2)+"_centered_predictions_pdp_data_JJA_"+str(yr)+"_w"+str(w)+"_nlag"+str(nlag)+".npy")
                elif hem == 'south':
                    predictions[kk_yr,:,:] = np.load("../processed_data_"+dset+"/"+region_str+"/SM_shuff_JJA_pdp/"+region_str+"_"+str(shuffle_sm).zfill(2)+"_centered_predictions_pdp_data_JJA_"+str(yr)+"_w"+str(w)+"_nlag"+str(nlag)+".npy")[:90,:]
            else:
                print('no shuffling')
                if hem == 'north':
                    predictions[kk_yr,:,:] = np.load("../processed_data_"+dset+"/"+region_str+"/JJA_pdp/"+region_str+"_centered_predictions_pdp_data_JJA_"+str(yr)+"_w"+str(w)+"_nlag"+str(nlag)+".npy")
                elif hem == 'south':
                    predictions[kk_yr,:,:] = np.load("../processed_data_"+dset+"/"+region_str+"/JJA_pdp/"+region_str+"_centered_predictions_pdp_data_JJA_"+str(yr)+"_w"+str(w)+"_nlag"+str(nlag)+".npy")[:90,:]

                    
    predictions_grid = predictions.reshape(len(years)*n_summer_days, len(x_vec))
    long_predictions = predictions_grid.flatten()
    print(np.shape(predictions_grid), np.shape(long_predictions))

    x_vec_grid = np.tile(x_vec, [len(years)*n_summer_days, 1])
    long_x_vec = x_vec_grid.flatten()
    print(np.shape(x_vec_grid), np.shape(long_x_vec))

    ax.scatter(long_x_vec, long_predictions, s=0.2, color='k', linewidths=0, label='_nolegend_')
    p = ax.hist2d(long_x_vec, long_predictions, bins = 200, cmin = 50*len(years), 
                      norm = colors.LogNorm(), zorder = 5, cmap = "cividis")

    pdp_mean = calc_full_rolling_mean(np.mean(predictions_grid, axis=0))
    pdp_5th = calc_full_rolling_mean(np.percentile(predictions_grid, q=5, axis=0))
    pdp_95th = calc_full_rolling_mean(np.percentile(predictions_grid, q=95, axis=0))
    
    clip = 15
    ax.plot(x_vec[clip:-clip], pdp_mean[clip:-clip], color='r', zorder=500, linewidth=5, label='mean')
    ax.plot(x_vec[clip:-clip], pdp_5th[clip:-clip], color='r', zorder=500, linewidth=0.5, label='5th percentile')
    ax.plot(x_vec[clip:-clip], pdp_95th[clip:-clip], color='r', zorder=500, linewidth=0.5, label='95th percentile')
    
    ax.scatter(0,0, color='dodgerblue', label='TMAX at SM = 0', zorder=1000, s=100)
    ax.hlines(y=0, xmin=-5, xmax=5, color='k', linewidth=0.5, zorder=2000)
    ax.vlines(x=0, ymin=long_predictions.min(), ymax=long_predictions.max(), color='k', linewidth=0.5, zorder=2000)
    ax.legend()
    
    ax.set_title(region_str+' summertime PDP ('+dset+') (lag='+str(nlag)+' days)', fontsize=20)
    ax.set_xlabel('region average SM anomaly', fontsize=16)
    ax.set_ylabel('$\Delta$TMAX relative to TMAX at SM=0 (C)', fontsize=16)
    ax.set_xlim(x_vec.min(), x_vec.max())  
    
    fig.savefig("../figures_"+dset+"/"+region_str+"_jja_pdp_nlag"+str(nlag)+".png", transparent=False)
    