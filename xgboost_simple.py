#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:50:02 2022

@author: shayan
"""

import pandas as pd
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser(description='Semi supervised forecastting')
parser.add_argument('--seed', default=200, type=int)
parser.add_argument('--test_dataset_name', default='None', type=str)
parser.add_argument('--validation_dataset_name', default='rideshare_dataset_without_missing_values', type=str)
parser.add_argument('--v_partition', default=0.1, type=int)
parser.add_argument('--lr', default=0.05, type=float)

parser.add_argument('--method', default='nlin', type=str)
parser.add_argument('--n_estimators', default=100, type=int)
# parser.add_argument('--max_depth', default=3, type=int)

global args
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

meta_data_df = pd.read_csv("dataset_that_can_be_used_to_train_info.csv")
meta_data_df = meta_data_df.drop(meta_data_df.columns[0], axis=1)

train_input_data = np.load("preproc_test_bugfree/onepass_seq_better_5k_fewtrain_"+args.validation_dataset_name+"_input_windows.npy")#, allow_pickle=True)
train_target_data = np.load("preproc_test_bugfree/onepass_seq_better_5k_fewtrain_"+args.validation_dataset_name+"_target_windows.npy")#, allow_pickle=True)
train_scalor_data = np.load("preproc_test_bugfree/onepass_seq_better_5k_fewtrain_"+args.validation_dataset_name+"_normalization_parameters.npy")#, allow_pickle=True)

test_input_data = np.load("preproc_test_bugfree/onepass_seq_test_"+args.validation_dataset_name+"_input_windows.npy")
test_target_data = np.load("preproc_test_bugfree/onepass_seq_test_"+args.validation_dataset_name+"_target_windows.npy")
test_scalor_data = np.load("preproc_test_bugfree/onepass_seq_test_"+args.validation_dataset_name+"_normalization_parameters.npy") 
        
indexes = np.arange(len(train_input_data))
np.random.shuffle(indexes)
num_validation_indexes = int( float(args.v_partition) * len(indexes) )

if num_validation_indexes==0:
    num_validation_indexes=1
    
train = train_input_data[:-num_validation_indexes]
validation = train_input_data[num_validation_indexes:]

train_scalor = train_scalor_data[:-num_validation_indexes]
validation_scalor = train_scalor_data[num_validation_indexes:]

meta_data_basic = pd.read_csv("meta_data.csv")
len_forecast_horizons = meta_data_basic[meta_data_basic['filename']==args.validation_dataset_name].forecast_horizon.item()
input_range_len = meta_data_basic[meta_data_basic['filename']==args.validation_dataset_name].lag.item()
frequency = meta_data_basic[meta_data_basic['filename']==args.validation_dataset_name].frequency.item()

train_input_channel_unnormalized = train[:,:input_range_len,0] * train_scalor 
validation_input_channel_unnormalized = validation[:,:input_range_len,0] * validation_scalor 
test_input_channel_unnormalized = test_input_data[:,:input_range_len,0] * test_scalor_data 

x_train = train[:,:input_range_len,0]
x_train[:,:input_range_len] =  train_input_channel_unnormalized
y_train = train[:,input_range_len:,0] * train_scalor 

x_validation = validation[:,:input_range_len,0]
x_validation[:,:input_range_len] =  validation_input_channel_unnormalized
y_validation  = validation[:,input_range_len:,0] * validation_scalor 

x_test = test_input_data[:,:input_range_len,0]
x_test[:,:input_range_len] =  test_input_channel_unnormalized
y_test  = test_target_data[:,:,0] #no need to normalize


x_train = x_train.reshape(x_train.shape[0], -1)
x_validation = x_validation.reshape(x_validation.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
def mae(final_forecasts_array,outsample_array ):#seasonality or frequency
    mase_per_series = []
    for i in range(final_forecasts_array.shape[0]):
        mase = np.mean(np.abs(final_forecasts_array[i] - outsample_array[i]))
        mase_per_series.append(mase)
    return np.mean(mase_per_series)#, len(mase_per_series)

#the following code was opted out in favor of the later MultiOutputRegressor call 
#that is better parallelized with n_jobs=-1 using all available CPUs.
#XGBoost basically fits to one horizon at a time
'''
validation_predictions=[]
test_predictions = []

for i in range(y_validation.shape[1]):
    reg = GradientBoostingRegressor(random_state=0, verbose=1, n_estimators = args.n_estimators, learning_rate = args.lr, max_depth = args.max_depth)
    reg.fit(x_train, y_train[:,i])
    vpreds_i = reg.predict(x_validation)
    testpreds_i = reg.predict(x_test)
    
    validation_predictions.append( vpreds_i )
    test_predictions.append( testpreds_i )
    print("done with: ", i, "/", y_train.shape[1])
    
validation_predictions = np.array(validation_predictions).T
test_predictions = np.array(test_predictions).T
'''

reg=MultiOutputRegressor(GradientBoostingRegressor(random_state=args.seed, 
                                                   verbose=1, 
                                                   n_estimators = args.n_estimators) 
                                                   # learning_rate = args.lr, 
                                                   # max_depth = args.max_depth)
                         , n_jobs=-1)
reg.fit(x_train, y_train)
validation_predictions = reg.predict(x_validation)
test_predictions = reg.predict(x_test)

validation_mae = mae(validation_predictions, y_validation)
test_mae =  mae(test_predictions, y_test)

print("validation mae: ", validation_mae)
print("test mae: ", test_mae)