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

tobebashed = """#!/usr/bin/env bash
#SBATCH --job-name=jobname                           
#SBATCH --output=jobname%j.log    
#SBATCH --partition=GPU,NGPU,STUD
#SBATCH -x gpu-[017]
#SBATCH --gres=gpu:1  
set -e
source /home/shayan/anaconda3/bin/activate /home/shayan/anaconda3/envs/pgpu
cd $PWD
srun /home/shayan/anaconda3/envs/pgpu/bin/python3 nlin_net_main_seq_2_final_adv_multi.py"""
#nlinr->nlin_net_main_seq_2_final_.py<---final nlin single (mae)

#nlinr2->nlin_net_main_seq_2_final_.py<---final nlin single (but with mse)
#nlinrg->nlin_net_main_seq_2_final_musigma.py<---final nlin single (but with mse & gaussian)
#nlinradv->nlin_net_main_seq_2_final_adv.py<---final nlin single (mae and adv) 
#nlinradvw->nlin_net_main_seq_2_final_adv.py<---final nlin single (mae, adv, weighted) #0.01 pe chal chuka hai, ab 0.1, 1, 10 pe chalaya hai
#nlinradvmulti->nlin_net_main_seq_2_final_adv_multi.py<--<---final nlin single (mae, adv, multi) #0.01 pe chal chuka hai, ab 0.1, 1, 10 pe chalaya hai

#nlinrlamn->nlin_net_main_seq_2_final_lamn.py<--with last and mean baselines
counter=0
for method in           ["nlinradvmulti"]:    
    for dataset_name in [#"saugeenday_dataset",#only 1 time series, nothing to validate on
                        # "us_births_dataset",#only 1 time series, nothing to validate on
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
                        # "rideshare_dataset_without_missing_values",
                        "m4_weekly_dataset",
                        "fred_md_dataset",
                        "m1_yearly_dataset",
                        "pedestrian_counts_dataset",
                        # "sunspot_dataset_without_missing_values",#only 1 time series, nothing to validate on
                        "covid_deaths_dataset",
                        "m4_quarterly_dataset",
                        "m4_monthly_dataset",
                        "bitcoin_dataset_without_missing_values"]:
        lr =  str(float(10**np.random.uniform(np.log10(1e-5),np.log10(0.001))))
        for seed in [0,1,2]:#changed at line 146 just below pred days
            # for adversarial_weight in [0.1, 1, 10]:
            args =  " --seed " + str(seed)+\
                    " --method " + str(method)+\
                    " --lr " + str(lr)+\
                    " --validation_dataset_name " + str(dataset_name)+"\n"
                    # " --adversarial_weight " + str(adversarial_weight)+\
                
            tobebashed_final = tobebashed.replace("jobname",'experiment_'+('_'.join([x for x in args.split() if "--" not in x])).replace("/","_")+"_")
            tobebashed_final = tobebashed_final+args
            counter+=1
            with open(('_'.join([x for x in args.split() if "--" not in x])+".sh").replace("\"","").replace("/","_"),"w") as fp:#'64_81_0.01_0.3_3_128_0.5_leaky_relu'
                fp.write(tobebashed_final)        
        
sh_filenames = os.listdir()
sh_filenames = [filename for filename in sh_filenames if ".sh" in filename]
#
for filename in sh_filenames:
    os.system("sbatch "+filename)                
os.system("rm *.sh")
