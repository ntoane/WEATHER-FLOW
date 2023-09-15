from typing import Text
from numpy.core.overrides import ArgSpec
# import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd
import sys
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import torch
import yaml
# from .utils import StandardScaler

# from xarray.core.variable import Coordinate
np.random.seed(2020)

class NormScaler:
    """
    Creates a scaler object that performs MinMax scaling on a data set
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max
        # print(self.min)
        # print(self.max)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data
        # print("----------------------------------------")
        # print(data.shape)
        max = self.max.values
        max = max.reshape((1, 1, 1, 6))
        min = self.min.values
        min = min.reshape((1, 1, 1, 6))
        return data * (max - min) + min

def latlon2xyz(lat,lon):
    lat = lat*np.pi/180
    lon = lon*np.pi/180 
    x= np.cos(lat)*np.cos(lon)
    y= np.cos(lat)*np.sin(lon)
    z= np.sin(lat)
    return x,y,z

def sliding_window(df, lag, horizon, split, numOfStations,tData,lon,lat,scaler):
    """
    Converts array to times-series input-output sliding-window pairs.
    Parameters:
        df - DataFrame of weather station data
        lag - length of input sequence
        horizon - length of output sequence(forecasting horizon)
        split - points at which to split data into train, validation, and test sets
        set - indicates if df is train, validation, or test set
    Returns:
        x, y - returns x input and y output
    """
    samples = int(split)
    


    dfy = df.drop(['Rain', 'Humidity', 'Pressure', 'WindSpeed', 'WindDir'], axis=1)
    
    stations = numOfStations
    features = 6
    tfeatures = 6

    date_format = "%Y-%m-%d %H:%M:%S"
    df = df.values.reshape(samples, stations, features)
    dfy = dfy.values.reshape(samples, stations, 1)
   
    l = len (tData)
    time = []
    for w in range(l):
        temp = datetime.strptime(tData[w], date_format)
        year =temp.year
        month =temp.month
        day =temp.day
        hour =temp.hour
        lg = lon[w]
        lt = lat[w]
        time.append([year,month,day,hour,lt,lg])


    time =np.array(time)

    time = time.reshape(samples, stations, tfeatures)

    x_offsets = np.sort(np.concatenate((np.arange(-(lag - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (horizon + 1), 1))
    data = np.expand_dims(df, axis=-1)
    data = data.reshape(samples, 1, stations, features)
    data = np.concatenate([data], axis=-1)
    time = time.reshape(samples, 1, stations, tfeatures)
    time = np.concatenate([time], axis=-1)

    datay = np.expand_dims(dfy, axis=-1)
    datay = datay.reshape(samples, 1, stations, 1)
    datay = np.concatenate([datay], axis=-1)

    x, y , times = [], [], []
    min_t = abs(min(x_offsets))
    max_t = abs(samples - abs(max(y_offsets)))  # Exclusive

    # t is the index of the last observation.
    # print(data[0 + x_offsets, ...])
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        times.append(time[t + x_offsets, ...])
        y.append(datay[t + y_offsets, ...])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    times = np.stack(times, axis=0)
    x = np.squeeze(x)
    times = np.squeeze(times)
    y = np.squeeze(y, axis=2)
    print('Shape of X : ' + str(x.shape))
    print('Shape of Y : ' + str(y.shape))
   
    return x, y, times

def split_data(data, split):
        return [data[:split[0]], data[split[0]:split[1]], data[split[1]:split[2]]]


def prepare_data_split(sharedConfig, increments):
        split = [increments[0] * sharedConfig['num_nodes']['default'],
                 (increments[1]) * sharedConfig['num_nodes']['default'],
                 (increments[2]) * sharedConfig['num_nodes']['default']]
        return split

def main(h,modelConfig,filePath,outputDir,increments):
    print("Generating data")
    numOfStations = modelConfig['num_nodes']['default']
    horizon = h
    lag = modelConfig['lag_length']['default']

    data = pd.read_csv(filePath)
    latitude = np.array(data['Latitude']).flatten()
    longitude = np.array(data['Longitude']).flatten()

    time = data['DateT']
    time_data = time
    lon,lat = data['Longitude'][0:numOfStations], data['Latitude'][0:numOfStations]
    lonlat = np.array([lon, lat])
    lonlat = lonlat.reshape(2,numOfStations).T

    raw_data = data.drop(['StasName', 'DateT', 'Latitude', 'Longitude'], axis=1)

    print('Generating data for horizon ' + str(h)+ ' split ' + str(increments[0]))
    splits = prepare_data_split(modelConfig,increments)
    sets = split_data(raw_data, splits)
    t_sets = split_data(time_data,splits)
    lo = split_data(longitude,splits)
    la = split_data(latitude,splits)
    scaler = NormScaler(raw_data[:splits[0]].min(), raw_data[:splits[0]].max())

    train_x, train_y, train_context = sliding_window(scaler.transform(sets[0]), lag, h, increments[0], numOfStations, t_sets[0].values,lo[0],la[0],scaler)
    val_x, val_y, val_context = sliding_window(scaler.transform(sets[1]), lag, h, increments[1] - increments[0], numOfStations, t_sets[1].values,lo[1],la[1],scaler)
    test_x, test_y, test_context = sliding_window(scaler.transform(sets[2]), lag, h, increments[2] - increments[1], numOfStations, t_sets[2].values,lo[2],la[2],scaler)
    
    datasets =[[train_x, train_y, train_context], [val_x, val_y, val_context], [test_x,test_y, test_context]]
    subsets = ['trn','val','test']
    path = outputDir + "/horizon_{}".format(h)
    path_ = Path(path)
    path_.mkdir(exist_ok=True,parents=True)

    for i, subset in enumerate(subsets):
        with open(path+'/{}_split_{}.pkl'.format(subset,increments[0]), "wb") as f:
            save_data = {'x': datasets[i][0],
                        'y': datasets[i][1],
                        'context': datasets[i][2]}
            pickle.dump(save_data,f, protocol = 4)

    with open(path+'/{}.pkl'.format('position_info'), "wb") as f:
        save_data = {'lonlat': lonlat}
        pickle.dump(save_data, f, protocol = 4)

    return scaler

if __name__ == "__main__":
    
    modelName = 'clcrn'
    with open('../configurations/'+ modelName +'Config.yaml', 'r') as file:
            modelConfig =  yaml.safe_load(file)

    filePath = 'DataNew/Graph Neural Network Data/Graph Station Data/graph.csv'
    outputDir = "Data/CLCRN Data"

    main(modelConfig,filePath,outputDir)
