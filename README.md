# Forecasting Early with Meta Learning
#### Dataset preprocessing

In order to run the experiments, please download the datasets (all) from the following repository: 
[Monash Forecasting Archive](https:////forecastingdata.org/)

Please run the following script to preprocess the datasets, (make sure the paths are configured): 

```python3 preproc_firststage.py```

The script takes care of padding the different series lengths (within one dataset), adds the time features (although unused in the paper) and generates additional meta-data about the datasets, for example a csv file that arranges the datasets in chronological order based on the time features. 

In the next preprocessing stage, sliding windows are drawn randomly from the **meta-train datasets**, from a total of 100 time series in each dataset, 50 windows are drawn. If there's less than 100 time series, more windows are drawn with sampling with replacement, such that for all datasets, there is at least 5000 learning time series window samples.

```python3 preproc_secondstage.py```

The above script also generates information such as dataset ids, time series ids, normalization parameters for the time series (following the DeepAR protocol). However, this information is not used in the paper, and can be safely ignored.

For the **meta-test dataset**, please use the following script to generate few-shot training samples and few-shot test samples,

```python3 test_dataset_writing.py```

The script above resembles the meta-train script, however, for the test dataset we generate few-shot training samples until the first timestamp at lag+forecast_horizon. And the test samples are then the forecast_horizon directly after the range lag+forecast_horizon.

#### Main experiment scripts

To reproduce the experiments for the single-task NLinear model (**NLinear**), please use the script: 

```python3 nlin_net.py```

For NLinear with Adversarial training (**NLinear + Ad**),

```python3 nlin_net_adv.py```

For the Multi-task learning baseline (**MTL**): 

```python3 mtl_baseline.py```

For the Joint learning baseline (**Joint**):

```python3 baseline_joint_model.py```

For the Meta-learning Reptile single head baseline (**S-Reptile**):

```python3 baseline_reptile_singlehead.py```

For the Meta-learning Multi-headed Reptile model (**M-Reptile**): 

```python3 baseline_reptile_multihead.py```

For the proposed model (**FEML**),

```python3 prop_feml.py```

A note on reproducibility, please see the hyperparameters tuned listed in the ```runner``` scripts.

#### Results

![](https://github.com/super-shayan/FEML//blob/master/comparison_figure.png?raw=true)

![](https://github.com/super-shayan/FEML//blob/master/main_table.png?raw=true)

#### Paper citation

If you find our scripts useful, please consider citing us. 
