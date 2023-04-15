#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 23:19:54 2021

@author: shayan
"""


import os
import numpy as np
import random
import sys
import re
import operator
import time
import copy
import pandas as pd

filenames=os.listdir()
filenames = [filename for filename in filenames if ".log" in filename and "nlinradvw" in filename]
print(filenames)
results = {}

for i,filename in enumerate(filenames):
    with open(filename, "r") as fp:
        content  = fp.read()
        print("filename: ", filename)
        if "m1_monthly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("m1_monthly_dataset","m1-monthly-dataset").split("_")[1:-1]
        elif "m3_quarterly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("m3_quarterly_dataset","m3-quarterly-dataset").split("_")[1:-1]
        elif "nn5_weekly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("nn5_weekly_dataset","nn5-weekly-dataset").split("_")[1:-1]
        elif "nn5_daily_dataset_without_missing_values" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("nn5_daily_dataset_without_missing_values","nn5-daily-dataset-without-missing-values").split("_")[1:-1]
        elif "m3_yearly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("m3_yearly_dataset","m3-yearly-dataset").split("_")[1:-1]
        elif "car_parts_dataset_without_missing_values" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("car_parts_dataset_without_missing_values","car-parts-dataset-without-missing-values").split("_")[1:-1]
        elif "m3_monthly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("m3_monthly_dataset","m3-monthly-dataset").split("_")[1:-1]
        elif "m1_quarterly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("m1_quarterly_dataset","m1-quarterly-dataset").split("_")[1:-1]
        elif "hospital_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("hospital_dataset","hospital-dataset").split("_")[1:-1]
        elif "solar_weekly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("solar_weekly_dataset","solar-weekly-dataset").split("_")[1:-1]
        elif "tourism_yearly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("tourism_yearly_dataset","tourism-yearly-dataset").split("_")[1:-1]
        elif "tourism_quarterly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("tourism_quarterly_dataset","tourism-quarterly-dataset").split("_")[1:-1]
        elif "tourism_monthly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("tourism_monthly_dataset","tourism-monthly-dataset").split("_")[1:-1]
        elif "electricity_weekly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("electricity_weekly_dataset","electricity-weekly-dataset").split("_")[1:-1]
        elif "electricity_hourly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("electricity_hourly_dataset","electricity-hourly-dataset").split("_")[1:-1]
        elif "australian_electricity_demand_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("australian_electricity_demand_dataset","australian-electricity-demand-dataset").split("_")[1:-1]
        elif "vehicle_trips_dataset_without_missing_values" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("vehicle_trips_dataset_without_missing_values","vehicle-trips-dataset-without-missing-values").split("_")[1:-1]
        elif "traffic_weekly_dataset" in filename and "kaggle" not in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("traffic_weekly_dataset","traffic-weekly-dataset").split("_")[1:-1]
        elif "traffic_hourly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("traffic_hourly_dataset","traffic-hourly-dataset").split("_")[1:-1]
        elif "kaggle_web_traffic_weekly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("kaggle_web_traffic_weekly_dataset","kaggle-web-traffic-weekly-dataset").split("_")[1:-1]
        elif "m4_hourly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("m4_hourly_dataset","m4-hourly-dataset").split("_")[1:-1]
        elif "kdd_cup_2018_dataset_without_missing_values" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("kdd_cup_2018_dataset_without_missing_values","kdd-cup-2018-dataset-without-missing-values").split("_")[1:-1]
        elif "m4_daily_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("m4_daily_dataset","m4-daily-dataset").split("_")[1:-1]
        elif "rideshare_dataset_without_missing_values" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("rideshare_dataset_without_missing_values","rideshare-dataset-without-missing-values").split("_")[1:-1]
        elif "m4_weekly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("m4_weekly_dataset","m4-weekly-dataset").split("_")[1:-1]
        elif "fred_md_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("fred_md_dataset","fred-md-dataset").split("_")[1:-1]
        elif "m1_yearly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("m1_yearly_dataset","m1-yearly-dataset").split("_")[1:-1]
        elif "pedestrian_counts_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("pedestrian_counts_dataset","pedestrian-counts-dataset").split("_")[1:-1]
        elif "covid_deaths_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("covid_deaths_dataset","covid-deaths-dataset").split("_")[1:-1]
        elif "m4_quarterly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("m4_quarterly_dataset","m4-quarterly-dataset").split("_")[1:-1]
        elif "m4_monthly_dataset" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("m4_monthly_dataset","m4-monthly-dataset").split("_")[1:-1]
        elif "bitcoin_dataset_without_missing_values" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("bitcoin_dataset_without_missing_values","bitcoin-dataset-without-missing-values").split("_")[1:-1]        
        elif "temperature_rain_dataset_without_missing_values" in filename:    seed, method, lr,weight,  validation_dataset_name= filename.replace("temperature_rain_dataset_without_missing_values","temperature-rain-dataset-without-missing-values").split("_")[1:-1]
        
        experiment_id = filename.split("_")[-1][:-4]

        try:
            validation_mae = re.findall(r'validation mae: +\d+.\d+', content)
            validation_mae = [float(mae.split(":")[-1]) for mae in validation_mae]
            
            test_mae = re.findall(r'test mae: +\d+.\d+', content)
            test_mae = [float(mae.split(":")[-1]) for mae in test_mae]
            
            results[(seed, method, lr, weight, validation_dataset_name, experiment_id)] = ( min(validation_mae), test_mae[ validation_mae.index( min(validation_mae) ) ] )
        except:
            results[(seed, method, lr,weight,  validation_dataset_name, experiment_id)] = ( np.inf, np.inf  )
        
    
    print("done with: ", i, "/", len(filenames))

df = pd.DataFrame(results.keys(), results.values() ).reset_index()
df.columns = ["validation_mae", "test_mae", "seed", "method", "lr", 'weight',  "validation_dataset_name","experiment_id"]

groups = df.groupby("validation_dataset_name")
for key in groups.groups.keys():
    argmin = groups.get_group(key).validation_mae.argmin()
    print(key, groups.get_group(key).iloc[argmin].test_mae)
