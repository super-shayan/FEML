#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:50:02 2022

@author: shayan
"""
import pandas as pd
import numpy as np
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler,SequentialSampler

from utils import *
import time
import random

import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Semi supervised forecastting')
parser.add_argument('--seed', default=2, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--test_dataset_name', default='None', type=str)
parser.add_argument('--validation_dataset_name', default='m4_daily_dataset', type=str)
parser.add_argument('--d_model', default=24, type=int)
parser.add_argument('--run_locally', default=1, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--v_partition', default=0.1, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--adversarial_weight', default=1e-1, type=float)
parser.add_argument('--method', default='nlin', type=str)

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

main_data_loaders = {}
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
few_validation_loader = DataLoader(few_main_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler =few_valid_sampler)

test_main_dataset = dataset(dataset_name=args.validation_dataset_name, test=True)
test_num_main_dataset = len(test_main_dataset)
test_main_dataset_idx = list( range(test_num_main_dataset) )
test_main_sampler = SequentialSampler(test_main_dataset_idx)#not really train, but just!
test_main_loader = DataLoader(test_main_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler =test_main_sampler)


meta_data_basic = pd.read_csv("meta_data.csv")
len_forecast_horizons = meta_data_basic[meta_data_basic['filename']==args.validation_dataset_name].forecast_horizon.item()
input_range_len = meta_data_basic[meta_data_basic['filename']==args.validation_dataset_name].lag.item()
frequency = meta_data_basic[meta_data_basic['filename']==args.validation_dataset_name].frequency.item()


class NLinear(nn.Module):
    def __init__(self, input_range_len, len_forecast_horizons):
        super(NLinear, self).__init__()
        self.linear1 = nn.Linear(input_range_len, len_forecast_horizons)
        self.input_range_len = input_range_len
        self.len_forecast_horizons = len_forecast_horizons
        
    def forward(self, simple_input): 
        last_input =simple_input[:,-1].unsqueeze(-1)
        simple_input = simple_input - last_input
        final_forecast = self.linear1(simple_input)
        final_forecast = final_forecast + last_input
        return final_forecast
    
sfnet = NLinear(input_range_len, len_forecast_horizons ).float().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(sfnet.parameters(), lr=args.lr)


def validation_iters(validation_loader):
    v_loss_batch_wise=[]
    for idx, data in enumerate(validation_loader):
        input_window, target_window, normalization_parameter = data
        simple_input = target_window[:,:input_range_len].float().to(device)
        simple_target = target_window[:,input_range_len:].float().to(device)
        with torch.no_grad():
            forecast = sfnet(simple_input)
        scaled_forecast = forecast.squeeze() #* normalization_parameter#torch.Size([64, 73]) * torch.Size([64, 1])
        main_loss = criterion(scaled_forecast.squeeze(), simple_target.squeeze())
        v_loss_batch_wise.append(main_loss.item())
    return np.mean(v_loss_batch_wise)

def test_iters(test_loader):
    forecasts = []
    outsample_arrays = []
    for idx, data in enumerate(test_loader):
        
        input_window, target_window, normalization_parameter = data#next ( iter( validation_loader ) )#since the dataloaders have random samplers, we just sample a random batch with next
        simple_input = input_window[:,:input_range_len,0].float().to(device)
        normalization_parameter = normalization_parameter.float().to(device)
        simple_input = simple_input *normalization_parameter
        
        
        batch_only_targets = copy.deepcopy(target_window[:,:,0])
        with torch.no_grad():
            batch_forecasts = sfnet(simple_input)
            
        rescaled_batch_forecasts = batch_forecasts.squeeze()
        forecasts.append(rescaled_batch_forecasts.squeeze().cpu().numpy())#torch.Size([64, 30])
        outsample_arrays.append(batch_only_targets.squeeze().cpu().numpy())
    return forecasts,outsample_arrays


def mae(final_forecasts_array,outsample_array ):#seasonality or frequency
    mase_per_series = []
    for i in range(final_forecasts_array.shape[0]):
        mase = np.mean(np.abs(final_forecasts_array[i] - outsample_array[i]))
        # mase = np.sqrt(np.mean(np.square(final_forecasts_array[i] - outsample_array[i])))
        mase_per_series.append(mase)
    return np.mean(mase_per_series)#, len(mase_per_series)


start = time.time()
n_iters = 1000*500

early_stopping = EarlyStopping(patience=7, verbose=True)
main_data_loaders[args.validation_dataset_name] = few_train_loader
renewed_main_data_loaders={}


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


for epoch in range(1000):
    batch_wise_loss=[]
    for num_batches in range(500):
        
        try:
            input_window_main, target_window_main, normalization_parameter_main = next (main_data_loaders[args.validation_dataset_name] ) #since the dataloaders have random samplers, we just sample a random batch with next
        except:
            renewed_main_data_loaders[args.validation_dataset_name] = iter(main_data_loaders[args.validation_dataset_name])
            input_window_main, target_window_main, normalization_parameter_main = next (renewed_main_data_loaders[args.validation_dataset_name] )

        simple_input = target_window_main[:,:input_range_len].float().to(device)
        simple_target = target_window_main[:,input_range_len:].float().to(device)
        
        ########################Adversarial
        simple_input.requires_grad = True
        ########################Adversarial
        
        
        forecast = sfnet(simple_input)
        scaled_forecast = forecast.squeeze() 
        main_loss = criterion(scaled_forecast.squeeze(), simple_target)
        # main_loss.backward(retain_graph=True)
        # torch.autograd.grad(main_loss, simple_input)
        
        ########################Adversarial
        perturbed_data = simple_input + args.adversarial_weight * torch.sign(torch.autograd.grad(main_loss, simple_input,retain_graph=True )[0])
        # data_grad = simple_input.grad.data
        # perturbed_data = fgsm_attack(simple_input, 0.01, data_grad)
        
        forecast_perturbed = sfnet(perturbed_data)
        aux_loss = criterion(forecast_perturbed.squeeze(), simple_target)
        total_loss = aux_loss + 2*main_loss
        ########################Adversarial
        
        #print("main_loss: ", main_loss.item(), "aux_loss: ", aux_loss.item(), "total_loss: ", total_loss.item())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        batch_wise_loss.append( total_loss.item() )
        # print("main loss: ", main_loss.item())
        if torch.isnan(main_loss).item()==True:
            print("stop early")

    print("epoch: ", epoch, "| main_loss: ", np.mean(batch_wise_loss))
        
    validation_l1error = validation_iters(few_validation_loader)  
    print("validation mae: ", validation_l1error)
    
    test_forecasts, test_outsample_arrays = test_iters(test_main_loader) 
    test_forecasts = np.concatenate(([x[np.newaxis,:] if len(x.shape)==1 else x for x in test_forecasts]))# (266, 30, 1)
    test_outsample_arrays = np.concatenate(([x[np.newaxis,:] if len(x.shape)==1 else x for x in test_outsample_arrays]))#(266, 30)
    test_l1error = mae(test_forecasts, test_outsample_arrays)
    print("test mae: ", test_l1error)

    early_stopping(validation_l1error)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    
print("total time taken: ", time.time() - start)