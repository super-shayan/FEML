#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:50:02 2022

@author: shayan
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4"
import pandas as pd
import numpy as np
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler,SequentialSampler

from datetime import datetime, date
import time
import datetime as dt
from dateutil.relativedelta import relativedelta
import random
from utils import *

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Semi supervised forecastting')
parser.add_argument('--seed', default=200, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--test_dataset_name', default='None', type=str)
parser.add_argument('--validation_dataset_name', default='fred_md_dataset', type=str)
parser.add_argument('--d_model', default=24, type=int)
parser.add_argument('--run_locally', default=1, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--v_partition', default=0.1, type=int)

parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--layer_size', default=100, type=int)
parser.add_argument('--stacks', default=3, type=int)

parser.add_argument('--method', default='nbeats', type=str)

parser.add_argument('--lr', default=1e-4, type=float)

global args
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


meta_data_df = pd.read_csv("dataset_that_can_be_used_to_train_info.csv")
meta_data_df = meta_data_df.drop(meta_data_df.columns[0], axis=1)


class dataset(Dataset):
    def __init__(self, dataset_name, test=False):
        self.dataset_name = dataset_name
        if test==False:
            self.input_data = np.load(dataset_name+"_input_windows.npy")#, allow_pickle=True)
            self.target_data = np.load(dataset_name+"_target_windows.npy")#, allow_pickle=True)
            self.scalor_data = np.load(dataset_name+"_normalization_parameters.npy")#, allow_pickle=True)
        elif test==True:
            self.input_data = np.load(dataset_name+"_input_windows.npy")
            self.target_data = np.load(dataset_name+"_target_windows.npy")
            self.scalor_data = np.load(dataset_name+"_normalization_parameters.npy") 
        
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx], self.scalor_data[idx]
# import pdb
# pdb.set_trace()
train_datasets_that_can_be_used = meta_data_df[meta_data_df['index']==args.validation_dataset_name].iloc[0,1:].tolist()#'covid_deaths_dataset'

try:
    train_datasets_that_can_be_used.pop(train_datasets_that_can_be_used.index("temperature_rain_dataset_without_missing_values"))
except:
    pass
try:
    train_datasets_that_can_be_used.pop(train_datasets_that_can_be_used.index("kaggle_web_traffic_weekly_dataset"))
except:
    pass

train_datasets_that_can_be_used.append(args.validation_dataset_name)
train_datasets_that_can_be_used = [x for x in train_datasets_that_can_be_used if str(x)!='nan']


train_datasets_that_can_be_used = ["saugeenday_dataset",#only 1 time series, nothing to validate on
                    "us_births_dataset",#only 1 time series, nothing to validate on
                    "m1_monthly_dataset",
                    "m3_quarterly_dataset",
                    "nn5_weekly_dataset",
                    "nn5_daily_dataset_without_missing_values",
                    "m3_yearly_dataset",
                    "car_parts_dataset_without_missing_values",
                    "m3_monthly_dataset",
                    "m1_quarterly_dataset",
                    "hospital_dataset",
                    "solar_weekly_dataset",
                    "tourism_yearly_dataset",
                    "tourism_quarterly_dataset",
                    "tourism_monthly_dataset",
                    "electricity_weekly_dataset",
                    "electricity_hourly_dataset",
                    "australian_electricity_demand_dataset",
                    "vehicle_trips_dataset_without_missing_values",
                    "traffic_weekly_dataset",
                    "traffic_hourly_dataset",
                    "temperature_rain_dataset_without_missing_values",
                    "kaggle_web_traffic_weekly_dataset",
                    "m4_hourly_dataset",
                    "kdd_cup_2018_dataset_without_missing_values",
                    "m4_daily_dataset",
                    "rideshare_dataset_without_missing_values",
                    "m4_weekly_dataset",
                    "fred_md_dataset",
                    "m1_yearly_dataset",
                    "pedestrian_counts_dataset",
                    "sunspot_dataset_without_missing_values",#only 1 time series, nothing to validate on
                    "covid_deaths_dataset",
                    "m4_quarterly_dataset",
                    "m4_monthly_dataset",
                    "bitcoin_dataset_without_missing_values"]



main_data_loaders = {}
for main_dataset_name in train_datasets_that_can_be_used:
    if args.validation_dataset_name==main_dataset_name:
        print("yes")
        continue#deal with test below, by defining a validation split for it as well
    
    main_dataset = dataset(main_dataset_name)#+'.arff')
    
    num_main_dataset = len(main_dataset)
    main_dataset_idx = list( range(num_main_dataset) )
    
    main_train_sampler = SubsetRandomSampler(main_dataset_idx)
        
    main_train_loader = DataLoader(main_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler =main_train_sampler)
    main_data_loaders[main_dataset_name] = main_train_loader


few_main_dataset = dataset(args.validation_dataset_name)
few_indices = list( range( len(few_main_dataset) ) )
split = int(args.v_partition* len(few_indices))

if split==0:
    split=1

np.random.shuffle(few_indices)
few_train_idx, few_valid_idx = few_indices[split:], few_indices[:split]
few_train_sampler = SubsetRandomSampler(few_train_idx)
few_valid_sampler = SequentialSampler(few_valid_idx)

few_train_loader = DataLoader(few_main_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler =few_train_sampler)
main_data_loaders[args.validation_dataset_name] = few_train_loader

few_validation_loader = DataLoader(few_main_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler =few_valid_sampler)

test_main_dataset = dataset(dataset_name=args.validation_dataset_name, test=True)
test_num_main_dataset = len(test_main_dataset)
test_main_dataset_idx = list( range(test_num_main_dataset) )
test_main_sampler = SequentialSampler(test_main_dataset_idx)#not really train, but just!
test_main_loader = DataLoader(test_main_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler =test_main_sampler)

#########################################################################################################################################
meta_data_basic = pd.read_csv("meta_data.csv")
len_forecast_horizons = meta_data_basic[meta_data_basic['filename']==args.validation_dataset_name].forecast_horizon.item()
input_range_len = meta_data_basic[meta_data_basic['filename']==args.validation_dataset_name].lag.item()

input_output_len_info_dict={}
for tuple1 in meta_data_basic[['filename', 'lag','forecast_horizon']].values:
    input_output_len_info_dict[tuple1[0]] = (tuple1[1], tuple1[2])
    
        
class NLinear(nn.Module):
    def __init__(self, input_range_len, len_forecast_horizons, layer_size, layers):
        super(NLinear, self).__init__()
        # self.layers = nn.ModuleList([nn.Linear(in_features=input_range_len, out_features=layer_size)] +
        #                               [nn.Linear(in_features=layer_size, out_features=layer_size)
        #                                for _ in range(layers - 1)])
        
        self.layers = torch.nn.ModuleList( [ torch.nn.Conv1d(1, layer_size, 3,padding='same') ] +
                                           [ torch.nn.Conv1d(layer_size, layer_size, 3,padding='same') 
                                             for _ in range(layers - 1) ]  )
        
        self.basis_parameters_dict = nn.ModuleDict( {
                                                     'australian_electricity_demand_dataset': nn.Linear(layer_size*420,336),
                                                     'bitcoin_dataset_without_missing_values': nn.Linear(layer_size*9,30),
                                                     'car_parts_dataset_without_missing_values': nn.Linear(layer_size*15,12),
                                                     'covid_deaths_dataset': nn.Linear(layer_size*9,30),
                                                     'electricity_hourly_dataset': nn.Linear(layer_size*30,168),
                                                     'electricity_weekly_dataset': nn.Linear(layer_size*65,8),
                                                     'fred_md_dataset': nn.Linear(layer_size*15,12),
                                                     'hospital_dataset': nn.Linear(layer_size*15,12),
                                                     'kaggle_web_traffic_weekly_dataset': nn.Linear(layer_size*10,8),
                                                     'kdd_cup_2018_dataset_without_missing_values': nn.Linear(layer_size*210,168),
                                                     'm1_monthly_dataset': nn.Linear(layer_size*15,18),
                                                     'm1_quarterly_dataset': nn.Linear(layer_size*5,8),
                                                     'm1_yearly_dataset': nn.Linear(layer_size*2,6),
                                                     'm3_monthly_dataset': nn.Linear(layer_size*15,18),
                                                     'm3_quarterly_dataset': nn.Linear(layer_size*5,8),
                                                     'm3_yearly_dataset': nn.Linear(layer_size*2,6),
                                                     'm4_daily_dataset': nn.Linear(layer_size*9,14),
                                                     'm4_hourly_dataset': nn.Linear(layer_size*210,48),
                                                     'm4_monthly_dataset': nn.Linear(layer_size*15,18),
                                                     'm4_quarterly_dataset': nn.Linear(layer_size*5,8),
                                                     'm4_weekly_dataset': nn.Linear(layer_size*65,13),
                                                     'nn5_daily_dataset_without_missing_values': nn.Linear(layer_size*9,56),
                                                     'nn5_weekly_dataset': nn.Linear(layer_size*65,8),
                                                     'pedestrian_counts_dataset': nn.Linear(layer_size*210,24),
                                                     'rideshare_dataset_without_missing_values': nn.Linear(layer_size*210,168),
                                                     'saugeenday_dataset': nn.Linear(layer_size*9,30),
                                                     'solar_weekly_dataset': nn.Linear(layer_size*6,5),
                                                     'sunspot_dataset_without_missing_values': nn.Linear(layer_size*9,30),
                                                     'temperature_rain_dataset_without_missing_values': nn.Linear(layer_size*9,30),
                                                     'tourism_monthly_dataset': nn.Linear(layer_size*15,24),
                                                     'tourism_quarterly_dataset': nn.Linear(layer_size*5,8),
                                                     'tourism_yearly_dataset': nn.Linear(layer_size*2,4),
                                                     'traffic_hourly_dataset': nn.Linear(layer_size*30,168),
                                                     'traffic_weekly_dataset': nn.Linear(layer_size*65,8),
                                                     'us_births_dataset': nn.Linear(layer_size*9,30),
                                                     'vehicle_trips_dataset_without_missing_values': nn.Linear(layer_size*9,30)
                                                 })
        
    def forward(self, simple_input, dataset_name): 
        simple_input = simple_input.unsqueeze(1)#torch.Size([64, 1, 210])
        last_input =simple_input[:,:,-1].unsqueeze(-1)#torch.Size([64, 1, 1])
        simple_input = simple_input - last_input
        # final_forecast = self.linear1(simple_input)
        # final_forecast = final_forecast + last_input
        block_input = simple_input
        for layer in self.layers:
            block_input = F.relu(layer(block_input))
        block_input = block_input.reshape(block_input.shape[0], -1)
        basis_parameters = self.basis_parameters_dict[dataset_name](block_input)    
        # basis_parameters = self.basis_parameters(block_input)
        final_forecast = (basis_parameters.unsqueeze(-1) + last_input).squeeze()
        return final_forecast
    
sfnet = NLinear(input_range_len, len_forecast_horizons, args.layer_size, args.num_layers ).float().to(device)

#########################################################################################################################################
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(sfnet.parameters(), lr=args.lr)

def validation_iters(validation_loader):
    v_loss_batch_wise=[]
    for idx, data in enumerate(validation_loader):
        input_window_main, target_window_main, normalization_parameter_main =data
        
        simple_input_main = target_window_main[  : ,  :input_output_len_info_dict[args.validation_dataset_name][0]   ].float().to(device)
        simple_target_main = target_window_main[  : ,  input_output_len_info_dict[args.validation_dataset_name][0]:   ].float().to(device)
        normalization_parameter_main = normalization_parameter_main.float().to(device)
        
        simple_input_main = simple_input_main / normalization_parameter_main
        simple_target_main = simple_target_main / normalization_parameter_main
        with torch.no_grad():
            forecast = sfnet(simple_input_main, args.validation_dataset_name)
        scaled_forecast = forecast.squeeze()
        main_loss = criterion(scaled_forecast.squeeze(), simple_target_main.squeeze())
        
        v_loss_batch_wise.append(main_loss.item())
        print("done with: ", idx, "/", len(validation_loader))
        # break
    return np.mean(v_loss_batch_wise)

def test_iters(test_loader):
    forecasts = []
    outsample_arrays = []
    # insample_arrays = []
    for idx, data in enumerate(test_loader):
        input_window_main, target_window_main, normalization_parameter_main = data#next ( iter( validation_loader ) )#since the dataloaders have random samplers, we just sample a random batch with next
        # print("target_window.shape, test, :", target_window.shape)
        """
        dataset_id = input_window[:,:,-2].long().to(device)
        time_series_id = input_window[:,:,-1].long().to(device)
        
        input_window = input_window.float().to(device)
        target_window = target_window.float().to(device)
        normalization_parameter = normalization_parameter.float().to(device)
        """
        normalization_parameter_main = normalization_parameter_main.float().to(device)
        
        simple_input_main = input_window_main[:,:,0].float().to(device)
        with torch.no_grad():
            batch_forecasts = sfnet(simple_input_main, args.validation_dataset_name)#torch.Size([64, 168])
        
        batch_only_targets = copy.deepcopy(target_window_main[:,:,0])

        rescaled_batch_forecasts = batch_forecasts.squeeze() *normalization_parameter_main #torch.Size([64, 30]) * torch.Size([64, 1])
        
        forecasts.append(rescaled_batch_forecasts.squeeze().cpu().numpy())#torch.Size([64, 30])
        outsample_arrays.append(batch_only_targets.squeeze().cpu().numpy())

        print("done with: ", idx, "/", len(test_loader))
    return forecasts,outsample_arrays#,insample_arrays

epoch_wise_loss={}
for dataset_name in train_datasets_that_can_be_used:
    epoch_wise_loss[dataset_name]=[]
    
epoch_wise_loss["validation_losses_mae"] = []
epoch_wise_loss["test_losses_mae"] = []
epoch_wise_loss["main_loss"]=[]

start = time.time()
n_iters = 1000*500

renewed_main_data_loaders = {}
        
early_stopping = EarlyStopping(patience=7, verbose=True)

for epoch in range(1000):
    batch_wise_loss={}
    for dataset_name in train_datasets_that_can_be_used:
        batch_wise_loss[dataset_name]=[]
    batch_wise_loss["main_loss"]=[]    
    
    for num_batches in range(500):
        losses=[]
        for i in range(len(train_datasets_that_can_be_used)): 
            random_dataset_chosen = train_datasets_that_can_be_used[i]
            try:
                input_window, target_window, normalization_parameter = next ( renewed_main_data_loaders[random_dataset_chosen]  )#since the dataloaders have random samplers, we just sample a random batch with next
            except:
                renewed_main_data_loaders[random_dataset_chosen] = iter(main_data_loaders[random_dataset_chosen])
                input_window, target_window, normalization_parameter = next ( renewed_main_data_loaders[random_dataset_chosen]  )
            
            simple_input = target_window[  : ,  :input_output_len_info_dict[random_dataset_chosen][0]   ].float().to(device)
            simple_target = target_window[  : ,  input_output_len_info_dict[random_dataset_chosen][0]:   ].float().to(device)
            normalization_parameter = normalization_parameter.float().to(device)
            
            simple_input = simple_input / normalization_parameter
            simple_target = simple_target / normalization_parameter
    
            """
            dataset_id = input_window[:,:,-2].long().to(device)
            time_series_id = input_window[:,:,-1].long().to(device)
            input_window = input_window.float().to(device)
            target_window = target_window.float().to(device)
            normalization_parameter = normalization_parameter.float().to(device)
            """
            
            forecast = sfnet(simple_input, random_dataset_chosen)
            scaled_forecast = forecast.squeeze() #* normalization_parameter#torch.Size([64, 73]) * torch.Size([64, 1])
            loss = criterion(scaled_forecast, simple_target)

            
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    validation_l1error = validation_iters(few_validation_loader)  
    print("validation mae: ", validation_l1error)
    
    test_forecasts, test_outsample_arrays = test_iters(test_main_loader) 
    test_forecasts = np.concatenate(( [x[np.newaxis,:] if len(x.shape)==1 else x for x in test_forecasts] ))#(270, 168)
    test_outsample_arrays = np.concatenate(( [x[np.newaxis,:] if len(x.shape)==1 else x for x in test_outsample_arrays] ))#(270, 168)
    test_l1error = mae(test_forecasts, test_outsample_arrays)
    print("test mae: ", test_l1error)

    
    early_stopping(validation_l1error)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    
print("total time taken: ", time.time() - start)
