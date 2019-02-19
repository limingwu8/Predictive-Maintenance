# Predictive Maintenance

## Overview
This is the code for time series data analyzing. We will use LSTM to predict the value of sensor reading in the future and generate a "health index" for each component of the system and finally generate a overall "health index" for the whole system, which indicated the system's future health status. 

## Dataset
Twelve time series data from twelve different sensors which including temperature sensor, pressure sensor and vibration sensor. 
The raw dataset are located under folder ["original"](https://github.com/limingwu8/Predictive-Maintenance/tree/master/dataset/csv/original). The format is time_sensorName.csv. e.g. 1705_MAIN_FILTER_OIL_TEMP.csv indicates dataset for Main Filter Oil Temperature sensor on May, 2017. The time interval of the raw dataset is different, which means need to be processed.
The preprocessed (sampled) dataset is located under folder ["sampled"](https://github.com/limingwu8/Predictive-Maintenance/tree/master/dataset/csv/sampled). They are sampled into different time intervals.

## Models
A long short-term memory network (LSTM) was implemented for data prediction. The schematic diagram of a LSTM cell, the building block of LSTM network, is shown in the following figures.
![image1](https://github.com/limingwu8/Predictive-Maintenance/blob/master/images/LSTM-cell.png)

![image2](https://github.com/limingwu8/Predictive-Maintenance/blob/master/images/LSTM-network.png)

## Scripts
* utils/: for data reading, sampling and write into CSV
* Sensor.py: predicting the sensor value in the future
* train_batch.py: train LSTM models

## Dependencies
```
Python (3.5)
Tensorflow (>1.0)
keras
numpy
pandas
scikit-learn
datetime
matplotlib
scipy
pickle
```

## Usage
1. Change the configurations in script train_batch.py. E.g. number of epochs, dataset path
2. Run script train_batch.py, the generated results will be saved in the path that you specified.


## Results

### Prediction
This is the single step prediction result for one day sampled oil return temperature sensor.
![image3](https://github.com/limingwu8/Predictive-Maintenance/blob/master/images/OIL_RETURN_TEMPERATURE-sample_1_day.png)
This is the RMSE comparison of multi-step prediction for different time interval sampled oil return temperature sensor.
![image4](https://github.com/limingwu8/Predictive-Maintenance/blob/master/images/RMSE.png)
### Health Index
This is the predicted health index of each sensor and the whole system on November.
![image5](https://github.com/limingwu8/Predictive-Maintenance/blob/master/images/health_index_combined.png)
