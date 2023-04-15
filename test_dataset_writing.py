#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:12:51 2022

@author: shayan
"""

import pandas as pd
import numpy as np
import argparse
import os

from datetime import datetime, date
import datetime as dt
from dateutil.relativedelta import relativedelta



import random
random.seed(200)
np.random.seed(200)

# dataset_name = "solar_weekly_dataset.txtt"
# data = np.loadtxt(dataset_name)

meta_data = pd.read_csv("/home/shayan/semifor/meta_data.csv")
# meta_data = meta_data.drop(meta_data.columns[0], axis=1)
meta_data.columns = ['dataset_id'] + meta_data.columns.tolist()[1:]#renaming the unnamed:0 column as the dataset id


dataset_names = os.listdir("/home/shayan/semifor/preproc1/")
dataset_names = [dataset_name for dataset_name in dataset_names if ".txt" in dataset_name]


#set the number of samples for the datasets
total_count = len(dataset_names)*100*50
counter=0
dataset_names=  ["australian_electricity_demand_dataset.txt",
                "bitcoin_dataset_without_missing_values.txt",
                "car_parts_dataset_without_missing_values.txt",
                "covid_deaths_dataset.txt",
                "electricity_hourly_dataset.txt",
                "electricity_weekly_dataset.txt",
                "fred_md_dataset.txt",
                "hospital_dataset.txt",
                "kaggle_web_traffic_weekly_dataset.txt",
                "kdd_cup_2018_dataset_without_missing_values.txt",
                "m1_monthly_dataset.txt",
                "m1_quarterly_dataset.txt",
                "m1_yearly_dataset.txt",
                "m3_monthly_dataset.txt",
                "m3_quarterly_dataset.txt",
                "m3_yearly_dataset.txt",
                "m4_daily_dataset.txt",
                "m4_hourly_dataset.txt",
                "m4_monthly_dataset.txt",
                "m4_quarterly_dataset.txt",
                "m4_weekly_dataset.txt",
                "nn5_daily_dataset_without_missing_values.txt",
                "nn5_weekly_dataset.txt",
                "pedestrian_counts_dataset.txt",
                "rideshare_dataset_without_missing_values.txt",
                "saugeenday_dataset.txt",
                "solar_weekly_dataset.txt",
                "sunspot_dataset_without_missing_values.txt",
                "temperature_rain_dataset_without_missing_values.txt",
                "tourism_monthly_dataset.txt",
                "tourism_quarterly_dataset.txt",
                "tourism_yearly_dataset.txt",
                "traffic_hourly_dataset.txt",
                "traffic_weekly_dataset.txt",
                "us_births_dataset.txt",
                "vehicle_trips_dataset_without_missing_values.txt"]

#code for plotting figures
# plt.figure()
# for i in range(test_split_x.shape[0]):
#     plt.plot(list(range(len(test_split_x[i]))), np.log(test_split_x[i]+1e-5), color='b'),
#     plt.plot(list(range(len(test_split_x[i])-1 , len(test_split_x[i])+len(test_split_y[i]) -1)) , np.log(test_split_y[i]+1e-5), color='b', linestyle='--' )
#     plt.axvline(x=len(test_split_x[i])-1, color='r', linestyle='-')
# dataset_names=["m1_monthly_dataset.txt"]

for d,dataset_name in enumerate(sorted(dataset_names)):#:
    data1 = np.loadtxt("/home/shayan/semifor/preproc1/"+dataset_name)
    if len(data1.shape)==1:
        data1 = data1[np.newaxis,:]
    
    #the 0.05 enumeration below can be ignored, it is a leftover 
    #implementation from initial experiments of seeing the first 
    #5% of the range of the time series in contrast to later
    #decided seeing the first forecast_horizon + lag of the dataset
    for ratio_index, ratio in enumerate([0.05]):
        dataset_id = 36#meta_data[meta_data.filename==dataset_name.replace('.txt','')].dataset_id.item()#giving it a new id after the highest observed 35#maximum time series id'd can be until 3865=3600+266
        frequency = meta_data[meta_data.filename==dataset_name.replace('.txt','')].frequency.item()
        lag = meta_data[meta_data.filename==dataset_name.replace('.txt','')].lag.item()
        forecast_horizon = meta_data[meta_data.filename==dataset_name.replace('.txt','')].forecast_horizon.item()
        
        input_windows=[]
        target_windows=[]
        normalization_parameters=[]
        
        test_input_windows=[]
        test_target_windows=[]
        test_normalization_parameters = []
        
        T = data1.shape[1]-6#in this T, the 6 time information features in the beginning are also included
        
        #the ratio hyperparameter is not used and can be ignored below
        if ratio==1.0:
            # data = data1[:,:data1.shape[1]-forecast_horizon]
            data = data1[:,:data1.shape[1]]
        else:
            #ignoring the ratio hyperparameter and seeing the data until the forecast_horizon+lag as training range
            #and the remaining forecast_horizon range as the testing range
            data = data1[:,:forecast_horizon+lag+6+forecast_horizon]
            
        # total_time_series_ids = 0#for every dataset 100 unique time series ids
        total_time_series_ids = {}
        for epoch in range(data1.shape[0]):#see each time series from data1 for test and not sample 100
            # if dataset_name=="m1_monthly_dataset.txt" and epoch==132:
            #     continue
        
            #for idx in range(data.shape[0]):#idx is time_series_id
            # if ratio==1.0 and epoch not in time_series_for_ratio_10:
            #     continue
                
            idx = epoch#np.random.randint(data1.shape[0])#
            
            #every time series idx randomly sampled is given a unique id from 0-99 in case we
            #sample 100 time series randomly from training but not applicable to test
            #since in test we forecast on all series for the forecast_horizon range observed after
            #lag+forecast_horizon
            
            if idx not in total_time_series_ids.keys():
                total_time_series_ids[idx] = epoch #epoch is a unique id from 0-99
        
            #for loop begin from before
            time_series1 = data[idx]
            
            _, train_start_time_year, train_start_time_month, train_start_time_day, train_start_time_hour, train_start_time_minute = time_series1[0:6]#representing frequency in numeric form as _
            start_datetime = pd.to_datetime('-'.join([str(int(x)) for x in [train_start_time_year, train_start_time_month, train_start_time_day]]) + ' '\
                                            + ':'.join([str(int(x)) for x in [train_start_time_hour, train_start_time_minute]]))
            
            if frequency == 'daily':
                datetime_array = [start_datetime + pd.DateOffset(days=i) for i in range(time_series1.shape[0]-6) ]#first 6 values are frequency, year, month,...
            elif frequency=='monthly':
                datetime_array = [start_datetime + pd.DateOffset(months=i) for i in range(time_series1.shape[0]-6) ]
            elif frequency=='half_hourly':
                datetime_array = [start_datetime + pd.DateOffset(minutes=i*30) for i in range(time_series1.shape[0]-6) ]
            elif frequency=='yearly':
                datetime_array = [start_datetime + pd.DateOffset(years=i) for i in range(time_series1.shape[0]-6) ]
            elif frequency=='quarterly':
                datetime_array = [start_datetime + pd.DateOffset(months=i*3) for i in range(time_series1.shape[0]-6) ]
            elif frequency=='4_seconds':
                datetime_array = [start_datetime + pd.DateOffset(seconds=i*4) for i in range(time_series1.shape[0]-6) ]
            elif frequency=='10_minutes':
                datetime_array = [start_datetime + pd.DateOffset(minutes=i*10) for i in range(time_series1.shape[0]-6) ]
            elif frequency=='hourly':
                datetime_array = [start_datetime + pd.DateOffset(hours=i) for i in range(time_series1.shape[0]-6) ]
            elif frequency=='weekly':
                datetime_array = [start_datetime + pd.DateOffset(weeks=i) for i in range(time_series1.shape[0]-6) ]
            elif frequency=='minutely':
                datetime_array = [start_datetime + pd.DateOffset(minutes=i) for i in range(time_series1.shape[0]-6) ]
                
            
            social_time_covariates = np.array([(x.year,x.month,x.day,x.hour,x.minute) for x in datetime_array])#(4587, 5)
            absolute_position_encoding = np.arange(len(social_time_covariates))
            dataset_id_repeated = np.repeat(dataset_id, len(social_time_covariates))
            
            time_series_id_to_be_repeated = total_time_series_ids[idx]
            time_series_id_repeated = np.repeat((dataset_id*100)+time_series_id_to_be_repeated, len(social_time_covariates))#dataset_id starts from 0, so offset by dataset_id+1 
            
            print("dataset_id: ", dataset_id, "(dataset_id*100)+total_time_series_ids: ", (dataset_id*100)+time_series_id_to_be_repeated, "/", data1.shape[0])
            
            time_series = np.copy(time_series1[6:])##first 6 values are frequency, year, month,...
            time_series_and_covariates = np.concatenate((time_series[:,np.newaxis],
                                                         social_time_covariates[:],
                                                         absolute_position_encoding[:,np.newaxis],
                                                         dataset_id_repeated[:,np.newaxis],
                                                         time_series_id_repeated[:,np.newaxis]), axis=-1)#time series is the first channel, channel=0
            if ratio==1.0:
                # test_split_y = np.copy(data1[idx, -forecast_horizon:])
                test_split_y = np.copy(time_series_and_covariates[-forecast_horizon:])
            else:
                # test_split_y = np.copy(data1[idx, int(T*ratio)+6:int(T*ratio)+6+forecast_horizon])
                test_split_y = np.copy(time_series_and_covariates[-forecast_horizon:,:])##removed 6 because time_series in line 149 already leaves the first 6
            
            # time_series_and_covariates = time_series_and_covariates[:-forecast_horizon]#have to do this because of addition of forecast_horizon in line 90 and 93
            test_split_x = np.copy(time_series_and_covariates[:-forecast_horizon,:][-lag: ,  :])
            time_series_channel_from_test_split_x = test_split_x[:,0] 
            time_series_channel_from_test_split_x = time_series_channel_from_test_split_x + 1e-5
            test_split_x_normalization_parameter = time_series_channel_from_test_split_x.mean()
            time_series_channel_from_test_split_x_normalized = time_series_channel_from_test_split_x / test_split_x_normalization_parameter
            test_split_x[:,0] = time_series_channel_from_test_split_x_normalized
            
            
            time_series_and_covariates = time_series_and_covariates[:-forecast_horizon,:]
            
            if np.isnan(test_split_x_normalization_parameter)==True:
                import pdb
                pdb.set_trace()
            if len(np.argwhere(test_split_x==np.inf))>0 or len(np.argwhere(test_split_y==np.inf))>0:
                print("continuing for epoch: ", epoch, "datset_name: ", dataset_name)
                continue
                # import pdb
                # pdb.set_trace()
                # time_series_with_infs.append(epoch)
                
            if True in np.unique(np.isnan(test_split_x[:,0])) or True in np.unique(np.isnan(test_split_y)):
                #there can be nans here if the time series has infs in the end, padding for a series that is
                # time_series_with_infs.append(epoch)
                # import pdb
                # pdb.set_trace()    
                print("continuing for epoch: ", epoch, "datset_name: ", dataset_name)
                continue
    
    
            test_input_windows.append(test_split_x)
            test_target_windows.append(test_split_y)
            test_normalization_parameters.append(test_split_x_normalization_parameter)
            
            sampled_window = np.copy(time_series_and_covariates)
            target_window = np.copy(time_series_and_covariates[:, 0])#0 indicates only target series
        
                # if len(np.argwhere(sampled_window==np.inf))>0 or len(np.argwhere(target_window==np.inf))>0:
                #     import pdb
                #     pdb.set_trace()
            time_series_channel_from_sampled_window_input = sampled_window[:,0]
            time_series_channel_from_sampled_window_input = time_series_channel_from_sampled_window_input+1e-5#in case there is a 0 in the input
            normalization_parameter = time_series_channel_from_sampled_window_input.mean()
            if np.isnan(normalization_parameter)==True:
                import pdb
                pdb.set_trace()
                
            time_series_channel_from_sampled_window_input_normalized = (time_series_channel_from_sampled_window_input / normalization_parameter ) 
            sampled_window[:,0] = time_series_channel_from_sampled_window_input_normalized
    
            if len(np.argwhere(sampled_window==np.inf))>0 or len(np.argwhere(target_window==np.inf))>0:
                import pdb
                pdb.set_trace()
    
            if True in np.unique(np.isnan(sampled_window[:,0])) or True in np.unique(np.isnan(target_window)):
                import pdb
                pdb.set_trace()
                
            input_windows.append(sampled_window)
            target_windows.append(target_window)
            normalization_parameters.append(normalization_parameter)
        
        test_input_windows = np.array(test_input_windows)
        test_target_windows = np.array(test_target_windows)
        test_normalization_parameters = np.array(test_normalization_parameters)[:,np.newaxis]
        # ratio = 0.05
        np.save("preproc_test_bugfree/onepass_seq_test_"+dataset_name.replace(".txt","")+"_input_windows", test_input_windows)
        np.save("preproc_test_bugfree/onepass_seq_test_"+dataset_name.replace(".txt","")+"_target_windows", test_target_windows)
        np.save("preproc_test_bugfree/onepass_seq_test_"+dataset_name.replace(".txt","")+"_normalization_parameters", test_normalization_parameters)
    
        input_windows = np.array(input_windows)
        target_windows = np.array(target_windows)
        normalization_parameters = np.array(normalization_parameters)[:,np.newaxis]
        
        np.save("preproc_test_bugfree/onepass_seq_better_5k_fewtrain_"+dataset_name.replace(".txt","")+"_input_windows", input_windows)
        np.save("preproc_test_bugfree/onepass_seq_better_5k_fewtrain_"+dataset_name.replace(".txt","")+"_target_windows", target_windows)
        np.save("preproc_test_bugfree/onepass_seq_better_5k_fewtrain_"+dataset_name.replace(".txt","")+"_normalization_parameters", normalization_parameters)
