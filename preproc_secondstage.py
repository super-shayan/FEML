#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:12:51 2022

@author: shayan
"""

import pandas as pd
import numpy as np
import os

from datetime import datetime, date
import time
import datetime as dt
from dateutil.relativedelta import relativedelta
import random


random.seed(200)
np.random.seed(200)

# dataset_name = "solar_weekly_dataset.txtt"
# data = np.loadtxt(dataset_name)

meta_data = pd.read_csv("meta_data.csv")
# meta_data = meta_data.drop(meta_data.columns[0], axis=1)
meta_data.columns = ['dataset_id'] + meta_data.columns.tolist()[1:]#renaming the unnamed:0 column as the dataset id


dataset_names = os.listdir("preproc1")
dataset_names = [dataset_name for dataset_name in dataset_names if ".txt" in dataset_name]

total_count = len(dataset_names)*100*50
counter=0

for d,dataset_name in enumerate(["electricity_weekly_dataset.txt"]):#sorted(dataset_names)):#["kaggle_web_traffic_weekly_dataset.txt"]
    data = np.loadtxt("preproc1/"+dataset_name)
    
    if len(data.shape)==1:
        data = data[np.newaxis,:]
        
    dataset_id = meta_data[meta_data.filename==dataset_name.replace('.txt','')].dataset_id.item()
    frequency = meta_data[meta_data.filename==dataset_name.replace('.txt','')].frequency.item()
    lag = meta_data[meta_data.filename==dataset_name.replace('.txt','')].lag.item()
    forecast_horizon = meta_data[meta_data.filename==dataset_name.replace('.txt','')].forecast_horizon.item()
    
    input_windows=[]
    target_windows=[]
    normalization_parameters=[]
    
    # total_time_series_ids = 0#for every dataset 100 unique time series ids
    total_time_series_ids = {}
    
    for epoch in range(100):
        #for idx in range(data.shape[0]):#idx is time_series_id
        idx = np.random.randint(data.shape[0])
        
        #every time series idx randomly sampled is given a unique id from 0-99
        if idx not in total_time_series_ids.keys():
            total_time_series_ids[idx] = epoch #epoch is a unique id from 0-99

        #for loop begin from before
        time_series = data[idx]
        
        _, train_start_time_year, train_start_time_month, train_start_time_day, train_start_time_hour, train_start_time_minute = time_series[0:6]#representing frequency in numeric form as _
        start_datetime = pd.to_datetime('-'.join([str(int(x)) for x in [train_start_time_year, train_start_time_month, train_start_time_day]]) + ' '\
                                        + ':'.join([str(int(x)) for x in [train_start_time_hour, train_start_time_minute]]))
        
        if frequency == 'daily':
            datetime_array = [start_datetime + pd.DateOffset(days=i) for i in range(time_series.shape[0]-6) ]#first 6 values are frequency, year, month,...
        elif frequency=='monthly':
            datetime_array = [start_datetime + pd.DateOffset(months=i) for i in range(time_series.shape[0]-6) ]
        elif frequency=='half_hourly':
            datetime_array = [start_datetime + pd.DateOffset(minutes=i*30) for i in range(time_series.shape[0]-6) ]
        elif frequency=='yearly':
            datetime_array = [start_datetime + pd.DateOffset(years=i) for i in range(time_series.shape[0]-6) ]
        elif frequency=='quarterly':
            datetime_array = [start_datetime + pd.DateOffset(months=i*3) for i in range(time_series.shape[0]-6) ]
        elif frequency=='4_seconds':
            datetime_array = [start_datetime + pd.DateOffset(seconds=i*4) for i in range(time_series.shape[0]-6) ]
        elif frequency=='10_minutes':
            datetime_array = [start_datetime + pd.DateOffset(minutes=i*10) for i in range(time_series.shape[0]-6) ]
        elif frequency=='hourly':
            datetime_array = [start_datetime + pd.DateOffset(hours=i) for i in range(time_series.shape[0]-6) ]
        elif frequency=='weekly':
            datetime_array = [start_datetime + pd.DateOffset(weeks=i) for i in range(time_series.shape[0]-6) ]
        elif frequency=='minutely':
            datetime_array = [start_datetime + pd.DateOffset(minutes=i) for i in range(time_series.shape[0]-6) ]
            
        
        social_time_covariates = np.array([(x.year,x.month,x.day,x.hour,x.minute) for x in datetime_array])#(4587, 5)
        absolute_position_encoding = np.arange(len(social_time_covariates))
        dataset_id_repeated = np.repeat(dataset_id, len(social_time_covariates))
        
        # time_series_id_repeated = np.repeat((dataset_id+1)*idx, len(social_time_covariates))#dataset_id starts from 0, so offset by dataset_id+1 
        # time_series_id_repeated = np.repeat((dataset_id*100)+total_time_series_ids, len(social_time_covariates))#dataset_id starts from 0, so offset by dataset_id+1 
        
        time_series_id_to_be_repeated = total_time_series_ids[idx]
        time_series_id_repeated = np.repeat((dataset_id*100)+time_series_id_to_be_repeated, len(social_time_covariates))#dataset_id starts from 0, so offset by dataset_id+1 
        # total_time_series_ids+=1
        
        print("dataset_id: ", dataset_id, "(dataset_id*100)+total_time_series_ids: ", (dataset_id*100)+time_series_id_to_be_repeated)
        
        time_series = time_series[6:]##first 6 values are frequency, year, month,...
        # time_series = (time_series - time_series.mean()) / time_series.std()#normalize the train time series 
        time_series_and_covariates = np.concatenate((time_series[:,np.newaxis],
                                                     social_time_covariates,
                                                     absolute_position_encoding[:,np.newaxis],
                                                     dataset_id_repeated[:,np.newaxis],
                                                     time_series_id_repeated[:,np.newaxis]), axis=-1)#time series is the first channel, channel=0
        

        for i in range(50):#100 epochs and 50 batches per epoch defaults corresponding to gluonts/monash repo
            try:
                sampled_index_from_valid_range = np.random.randint(lag, np.argwhere(time_series==np.inf).reshape(-1).min()-forecast_horizon)
            except:
                sampled_index_from_valid_range = np.random.randint(lag, len(time_series)-forecast_horizon)
            
            sampled_window = np.copy(time_series_and_covariates[sampled_index_from_valid_range-lag  :  sampled_index_from_valid_range+forecast_horizon, :])
        
            time_series_channel_from_sampled_window_input = sampled_window[:lag,0]
            normalization_parameter = time_series_channel_from_sampled_window_input.mean()
            
            # print("normalization_parameter: ",  normalization_parameter)
            if np.isnan(normalization_parameter)==True:
                import pdb
                pdb.set_trace()
            
            
            #the following line causes a bug, a very serious one, because it overwrites the above time_series_and_covariates array, solved by using np.copy() for sampled_window in line 142
            # time_series_channel_from_sampled_window_input_normalized = (time_series_channel_from_sampled_window_input / time_series_channel_from_sampled_window_input.mean() ) 
            time_series_channel_from_sampled_window_input = time_series_channel_from_sampled_window_input+1e-5#in case there is a 0 in the input
            time_series_channel_from_sampled_window_input_normalized = (time_series_channel_from_sampled_window_input / time_series_channel_from_sampled_window_input.mean() ) 
            

            input_window = sampled_window[:lag,:]
            input_window[:lag,0] = time_series_channel_from_sampled_window_input_normalized
            target_window = sampled_window[lag:,:]
            
            input_windows.append(input_window)
            target_windows.append(target_window)
            normalization_parameters.append(normalization_parameter)
            
            if True in np.unique(np.isnan(input_window)) or True in np.unique(np.isnan(target_windows)):
                import pdb
                pdb.set_trace()
            
            counter+=1
            print("done with: ", dataset_name,"/", d, "/", len(dataset_names), "counter: ", (counter/total_count) * 100, "%" )
        

        #for loop end from before
    
    input_windows = np.array(input_windows)
    target_windows = np.array(target_windows)
    normalization_parameters = np.array(normalization_parameters)[:,np.newaxis]
    
    np.save("preproc2/"+dataset_name.replace(".txt","")+"_input_windows", input_windows)
    np.save("preproc2/"+dataset_name.replace(".txt","")+"_target_windows", target_windows)
    np.save("preproc2/"+dataset_name.replace(".txt","")+"_normalization_parameters", normalization_parameters)
    print("done with: ",dataset_name,"   ", d, "/", len(dataset_names) )
