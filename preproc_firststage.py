import os
import numpy as np
import pandas as pd
import preproc_script_monash

from datetime import datetime

all_filenames = os.listdir("datasets_raw")

# The name of the column containing time series values after loading data from the .tsf file into a dataframe
VALUE_COL_NAME = "series_value"

# The name of the column containing timestamps after loading data from the .tsf file into a dataframe
TIME_COL_NAME = "start_timestamp"

data_info_dict={}
all_date_time_starts_global = []
meta_data = []

frequency_mapping = {
                    'yearly': 0,
                    'quarterly': 1,
                    'monthly': 2,
                    'weekly': 3,
                    'daily': 4,
                    'hourly': 5,
                    'half_hourly': 6,
                    '10_minutes': 7,
                    'minutely': 8,
                    '4_seconds' :9
                    }

all_filenames = list(set(all_filenames) - set(['m3_other_dataset.tsf']))#the m3_other_dataset does not have a date at all
#using only the ones now from monash, since they have a valid lag and forecast horizon defined only for the datasets below
#(lags, forecast_horizon)
all_filenames = {"nn5_daily_dataset_without_missing_values.tsf": (9,56),
                "tourism_yearly_dataset.tsf": (2,4),
                "tourism_quarterly_dataset.tsf": (5,8),
                "tourism_monthly_dataset.tsf": (15,24),
                "m1_yearly_dataset.tsf": (2,6),
                "m1_quarterly_dataset.tsf": (5,8),
                "m1_monthly_dataset.tsf": (15,18),
                "m3_yearly_dataset.tsf": (2,6),
                "m3_quarterly_dataset.tsf": (5,8),
                "m3_monthly_dataset.tsf": (15,18),
                # "m3_other_dataset.tsf": (2),
                "m4_quarterly_dataset.tsf": (5,8),
                "m4_monthly_dataset.tsf": (15,18),
                "m4_weekly_dataset.tsf": (65,13),
                "m4_daily_dataset.tsf": (9,14),
                "m4_hourly_dataset.tsf": (210,48),
                "car_parts_dataset_without_missing_values.tsf": (15,12),
                "hospital_dataset.tsf": (15,12),
                "fred_md_dataset.tsf": (15,12),
                "nn5_weekly_dataset.tsf": (65,8),
                "traffic_weekly_dataset.tsf": (65,8),
                "electricity_weekly_dataset.tsf": (65,8),
                "solar_weekly_dataset.tsf": (6,5),
                "kaggle_web_traffic_weekly_dataset.tsf": (10,8),
                # "dominick_dataset.tsf": (10,8),#skip because no datetime information
                "us_births_dataset.tsf": (9,30),
                "saugeenday_dataset.tsf": (9,30),
                "sunspot_dataset_without_missing_values.tsf": (9,30),
                "covid_deaths_dataset.tsf": (9,30),
                # "weather_dataset.tsf": (9,30),,#skip because no datetime information
                "traffic_hourly_dataset.tsf": (30,168),
                "electricity_hourly_dataset.tsf": (30,168),
                "kdd_cup_2018_dataset_without_missing_values.tsf": (210,168),
                "pedestrian_counts_dataset.tsf": (210,24),
                "bitcoin_dataset_without_missing_values.tsf": (9,30),
                "vehicle_trips_dataset_without_missing_values.tsf": (9,30),
                "australian_electricity_demand_dataset.tsf": (420,336),
                "rideshare_dataset_without_missing_values.tsf": (210,168),
                "temperature_rain_dataset_without_missing_values.tsf": (9,30)}


meta_data_set_anamolies = []
for i,filename in enumerate(sorted(all_filenames.keys())):
    input_file_name='datasets_raw/'+filename
    # try:
    df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = preproc_script_monash.convert_tsf_to_dataframe(full_file_path_and_name = input_file_name, replace_missing_vals_with = 'NaN',  value_column_name =VALUE_COL_NAME)
    # df = df.fillna(0)
    
    print("Dataset: ", filename, " is loaded!")
    
    max_so_far=-np.inf
    min_so_far=np.inf
    total_changes_to_max_so_far=0
    total_series=0
    
    data_tensor = []
    
    max_index = -1
    
    data_set_anamoly_indexes_only = []    
    
    all_latest_end_dates_per_dataset_without_anamolies = []
    for index, row in df.iterrows():
        series_data = row[VALUE_COL_NAME]
        series_len = len(series_data)#df[df.series_name==row.series_name].series_value.values[0].shape[0]
        
        if min_so_far > series_len:
            min_so_far = series_len
            
        if max_so_far < series_len:
            max_so_far = series_len
            print("change in max_so_far: ", max_so_far)
            total_changes_to_max_so_far+=1
            
            print("series_with_max_len_start_time: ", row[TIME_COL_NAME])
            # if index==4098, for m4_daily_dataset:
            #     import pdb
            #     pdb.set_trace()
            # start_timestamp = pd.to_datetime('2011-12-14 12:00:00')
            # [start_timestamp + pd.DateOffset(days=i) for i in range(9933)][-1]
            # Out  [9]: Timestamp('2039-02-22 12:00:00')
            # max_so_far
            # Out  [16]: 9933
            
        #because of the bug above, calculate each time series's latest end date, based on the given frequency, and then remove them later
        #if they have vastly in the future latest end dates
        train_start_time = row[TIME_COL_NAME]
        if frequency=='daily':
            latest_end_date =train_start_time+pd.DateOffset(days=series_len-1)
        elif frequency=='monthly':
            latest_end_date = train_start_time+pd.DateOffset(months=series_len-1)    
        elif frequency=='half_hourly':
            latest_end_date = train_start_time+pd.DateOffset(minutes=(series_len-1)*30 )  
        elif frequency=='yearly':
            latest_end_date = train_start_time+pd.DateOffset(years=series_len-1)
        elif frequency=='quarterly':
            latest_end_date = train_start_time+pd.DateOffset(months=(series_len-1)*3)    
        elif frequency=='4_seconds':
            latest_end_date =train_start_time+pd.DateOffset(seconds=(series_len-1)*4)   
        elif frequency=='10_minutes':
            latest_end_date = train_start_time+pd.DateOffset(minutes=(series_len-1)*10)     
        elif frequency=='hourly':
            latest_end_date = train_start_time+pd.DateOffset(hours=series_len-1)     
        elif frequency=='weekly':
            latest_end_date = train_start_time+pd.DateOffset(weeks=series_len-1)     
        elif frequency=='minutely':
            latest_end_date = train_start_time+pd.DateOffset(minutes=series_len-1)   
        
        if latest_end_date > pd.to_datetime('2021-10-31 12:00:00'):
            print("anamoly: index: ", filename, index, latest_end_date)
            meta_data_set_anamolies.append([filename, frequency, index, train_start_time, series_len, latest_end_date])
            data_set_anamoly_indexes_only.append(index)
        else:
            all_latest_end_dates_per_dataset_without_anamolies.append(latest_end_date)
            
        total_series+=1
        
        # print("done with: ", index, "/", len(df))
        
    print("total changes in max_so_far: ", total_changes_to_max_so_far)
    print("total series seen: ", total_series)
    print("max_so_far: ", max_so_far)
    
    data_info_dict[filename.replace('.tsf','')] = ([max_so_far, total_series])
    
    
    all_date_time_starts=[]
    for index, row in df.iterrows():
        if index in data_set_anamoly_indexes_only:
            continue; #do not add the anamolous samples to the time series data
            
        #every series has it's own start time, year, month, day...
        if TIME_COL_NAME in df.columns:
            train_start_time = row[TIME_COL_NAME]
            all_date_time_starts.append([filename,
                                         index, 
                                         frequency,
                                         str(train_start_time), 
                                         train_start_time.year,
                                         train_start_time.month,
                                         train_start_time.day,
                                         train_start_time.hour,
                                         train_start_time.minute
                                         ])
            
            train_start_time_year = train_start_time.year
            train_start_time_month = train_start_time.month
            train_start_time_day = train_start_time.day
            train_start_time_hour = train_start_time.hour
            train_start_time_minute = train_start_time.minute
            dummy_date_indicator=0
            
        else:
            train_start_time = datetime.strptime('1900-01-01 00-00-00', '%Y-%m-%d %H-%M-%S') # Adding a dummy timestamp, if the timestamps are not available in the dataset or consider_time is False
            all_date_time_starts.append([filename,
                                        index, 
                                        frequency,
                                        str(train_start_time), 
                                        train_start_time.year,
                                        train_start_time.month,
                                        train_start_time.day,
                                        train_start_time.hour,
                                        train_start_time.minute
                                        ])
            dummy_date_indicator=1
        
            if frequency=='yearly':
                train_start_time_year = round(1975.535883)
                train_start_time_month = round( 7.804215)
                train_start_time_day = round(20.702132)
                train_start_time_hour = round(11.337962)
                train_start_time_minute =round( 0.000000)
            elif frequency=='quarterly':
                train_start_time_year =round(1989.950873)
                train_start_time_month = round(4.786511)
                train_start_time_day =round( 21.327306)
                train_start_time_hour = round( 11.346177)
                train_start_time_minute =round(0.000000)
            elif frequency=='monthly':
                train_start_time_year =round(1995.235452)
                train_start_time_month = round(3.245984)
                train_start_time_day = round(18.152839)
                train_start_time_hour = round(10.178657)
                train_start_time_minute = round(0.000000)
            elif frequency=='weekly':
                train_start_time_year =round( 2014.926471)
                train_start_time_month =round(6.935439)
                train_start_time_day = round(1.024119)
                train_start_time_hour =round(0.029335)
                train_start_time_minute = round(0.000000)
            elif frequency=='daily':
                train_start_time_year = round(2014.774894)
                train_start_time_month =round(6.735983)
                train_start_time_day =round(1.484129)
                train_start_time_hour = round(1.845644)
                train_start_time_minute = round(0.000000)
        
        
        series_data = np.array(row[VALUE_COL_NAME])
        series_len = len(series_data)#df[df.series_name==row.series_name].series_value.values[0].shape[0]
        max_so_far_of_dataset = data_info_dict[filename.replace('.tsf','')][0]
        
        
        
        if series_len < max_so_far_of_dataset:
            data_tensor.append( np.concatenate(( 
                                                np.array([ frequency_mapping[frequency], train_start_time_year, train_start_time_month, train_start_time_day, train_start_time_hour, train_start_time_minute  ]),
                                                series_data, 
                                                np.inf*np.ones( max_so_far_of_dataset - len(series_data)  ))) )
        else:
            data_tensor.append( np.concatenate(( 
                                                np.array([ frequency_mapping[frequency], train_start_time_year, train_start_time_month, train_start_time_day, train_start_time_hour, train_start_time_minute  ]),
                                                series_data))) 
            
    data_tensor = np.array(data_tensor)  

    
    if contain_missing_values==True:
        data_tensor = np.where(data_tensor!='NaN', data_tensor, -np.inf).astype('float')
    
    
    np.savetxt(filename.replace(".tsf",".txt"), data_tensor)   

    earliest_start_date = pd.Series([pd.to_datetime(x) for x in np.array(all_date_time_starts)[:,3]]).min()#[3 is where the original datetime info is, see line: ~149
    latest_start_date = pd.Series([pd.to_datetime(x) for x in np.array(all_date_time_starts)[:,3]]).max()#[3 is where the original datetime info is, see line: ~149
    latest_end_date= pd.Series([pd.to_datetime(x) for x in np.array(all_latest_end_dates_per_dataset_without_anamolies)]).max()#although this list is already in datetime

    print("done with: ", i, "/", len(all_filenames), data_tensor.shape[0],data_tensor.shape[1], max_so_far, min_so_far,frequency, data_tensor.dtype )
    meta_data.append([filename.replace('.tsf',''), data_tensor.shape[0],data_tensor.shape[1], max_so_far, min_so_far,frequency,
                      all_filenames[filename][0],all_filenames[filename][1], contain_missing_values, contain_equal_length,
                      dummy_date_indicator,
                      earliest_start_date, latest_start_date, latest_end_date])
    
    all_date_time_starts_global.append(all_date_time_starts)
    

all_date_time_starts_global = np.concatenate([np.array(x) for x in all_date_time_starts_global])
all_date_time_starts_df = pd.DataFrame(all_date_time_starts_global)
all_date_time_starts_df.columns=['filename',
                                'index' ,
                                'frequency',
                                'train_start_time' ,
                                'train_start_time.year',
                                'train_start_time.month',
                                'train_start_time.day',
                                'train_start_time.hour',
                                'train_start_time.minute']
all_date_time_starts_df.to_csv("meta_data_date_time_starts.csv")


meta_data_df = pd.DataFrame(meta_data)
meta_data_df.columns = ['filename', 'data_tensor.shape[0]','data_tensor.shape[1]', 'max_so_far', 'min_so_far','frequency',
                        'lag','forecast_horizon', 'contain_missing_values', 'contain_equal_length','dummy_date_indicator',
                        'earliest_start_date', 'latest_start_date', 'latest_end_date']
meta_data_df.to_csv("meta_data.csv")


meta_data_set_anamolies_df = pd.DataFrame(meta_data_set_anamolies)
meta_data_set_anamolies_df.columns = ['filename', 'frequency', 'index', 'train_start_time', 'series_len', 'latest_end_date']
meta_data_set_anamolies_df.to_csv("meta_data_set_anamolies.csv")


datasets_that_can_be_used_to_train = {}
for i in range(meta_data_df.shape[0]):
    data_set_earliest_date = meta_data_df.iloc[i,-3]#earliest start date
    
    datasets_that_can_be_used_to_train[meta_data_df.iloc[i,0]] = []
    for j in range(i+1,meta_data_df.shape[0]):
        #print(i,j)
        data_set_latest_date = meta_data_df.iloc[j,-1]
        if data_set_latest_date < data_set_earliest_date:
            print(meta_data_df.iloc[i,0], meta_data_df.iloc[j,0])
            datasets_that_can_be_used_to_train[meta_data_df.iloc[i,0]].append(meta_data_df.iloc[j,0])

for key in datasets_that_can_be_used_to_train:
    if len(datasets_that_can_be_used_to_train[key])>0:
        datasets_that_can_be_used_to_train[key] = tuple(datasets_that_can_be_used_to_train[key])
    else:
        datasets_that_can_be_used_to_train[key] = tuple([0])
        
pd.DataFrame(datasets_that_can_be_used_to_train.values(), datasets_that_can_be_used_to_train.keys()).reset_index().to_csv("dataset_that_can_be_used_to_train_info.csv")

#following is used to figure out which datasets with dummy dates have what kind of frequencies
'''
# for example only the weather dataset has no start date (or dummy date) and the frequency is still daily
all_date_time_starts_df.groupby(['train_start_time','frequency']).get_group(('1900-01-01 00:00:00','daily')).filename.unique()
Out[35]: array(['weather_dataset.tsf'], dtype=object)

all_date_time_starts_df.groupby(['train_start_time','frequency']).get_group(('1900-01-01 00:00:00','yearly')).filename.unique()
Out[36]: array(['m1_yearly_dataset.tsf'], dtype=object)

all_date_time_starts_df.groupby(['train_start_time','frequency']).get_group(('1900-01-01 00:00:00','monthly')).filename.unique()
Out[37]: 
array(['cif_2016_dataset.tsf', 'm3_monthly_dataset.tsf',
        'm1_monthly_dataset.tsf'], dtype=object)

all_date_time_starts_df.groupby(['train_start_time','frequency']).get_group(('1900-01-01 00:00:00','weekly')).filename.unique()
Out[45]: array(['dominick_dataset.tsf'], dtype=object)
all_date_time_starts_df.groupby(['train_start_time','frequency']).get_group(('1900-01-01 00:00:00','quarterly')).filename.unique()
Out[42]: array(['m1_quarterly_dataset.tsf'], dtype=object)

#for other frequencies: (note: None is for reason showing in unique, but verified not in the dataframe; smh)
# all_date_time_starts_df.frequency.unique()
# Out[51]: 
# array(['monthly', 'daily', 'half_hourly', 'yearly', 'quarterly',
#        '4_seconds', '10_minutes', 'hourly', 'weekly', 'minutely', None],
#       dtype=object)
    
    
# all_date_time_starts_df.groupby(['train_start_time','frequency']).get_group(('1900-01-01 00:00:00','hourly')).filename.unique()
# Traceback (most recent call last):

#   File "/tmp/ipykernel_15304/3873974383.py", line 1, in <cell line: 1>
#     all_date_time_starts_df.groupby(['train_start_time','frequency']).get_group(('1900-01-01 00:00:00','hourly')).filename.unique()

#   File "/home/shayan/anaconda3/envs/yellow/lib/python3.9/site-packages/pandas/core/groupby/groupby.py", line 747, in get_group
#     raise KeyError(name)

# KeyError: ('1900-01-01 00:00:00', 'hourly')
'''


#following is to figure out the average date (year, month, etc.) values for the datasets that do not have a valid start date
'''
sub_all_date_time_starts_df = all_date_time_starts_df[all_date_time_starts_df['train_start_time']!='1900-01-01 00:00:00']

sub_all_date_time_starts_df.groupby(['frequency']).get_group('yearly')[['train_start_time.year','train_start_time.month','train_start_time.day','train_start_time.hour','train_start_time.minute']].describe().iloc[1]
Out[103]: 
train_start_time.year      1975.535883
train_start_time.month        7.804215
train_start_time.day         20.702132
train_start_time.hour        11.337962
train_start_time.minute       0.000000
Name: mean, dtype: float64

sub_all_date_time_starts_df.groupby(['frequency']).get_group('quarterly')[['train_start_time.year','train_start_time.month','train_start_time.day','train_start_time.hour','train_start_time.minute']].describe().iloc[1]
Out[104]: 
train_start_time.year      1989.950873
train_start_time.month        4.786511
train_start_time.day         21.327306
train_start_time.hour        11.346177
train_start_time.minute       0.000000
Name: mean, dtype: float64

sub_all_date_time_starts_df.groupby(['frequency']).get_group('monthly')[['train_start_time.year','train_start_time.month','train_start_time.day','train_start_time.hour','train_start_time.minute']].describe().iloc[1]
Out[105]: 
train_start_time.year      1995.235452
train_start_time.month        3.245984
train_start_time.day         18.152839
train_start_time.hour        10.178657
train_start_time.minute       0.000000
Name: mean, dtype: float64

sub_all_date_time_starts_df.groupby(['frequency']).get_group('weekly')[['train_start_time.year','train_start_time.month','train_start_time.day','train_start_time.hour','train_start_time.minute']].describe().iloc[1]
Out[107]: 
train_start_time.year      2014.926471
train_start_time.month        6.935439
train_start_time.day          1.024119
train_start_time.hour         0.029335
train_start_time.minute       0.000000
Name: mean, dtype: float64

sub_all_date_time_starts_df.groupby(['frequency']).get_group('daily')[['train_start_time.year','train_start_time.month','train_start_time.day','train_start_time.hour','train_start_time.minute']].describe().iloc[1]
Out[106]: 
train_start_time.year      2014.774894
train_start_time.month        6.735983
train_start_time.day          1.484129
train_start_time.hour         1.845644
train_start_time.minute       0.000000
Name: mean, dtype: float64

'''