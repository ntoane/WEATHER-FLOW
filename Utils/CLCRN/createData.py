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
# from xarray.core.variable import Coordinate
np.random.seed(2020)

class NormScaler:
    """
    Creates a scaler object that performs MinMax scaling on a data set
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data

def latlon2xyz(lat,lon):
    lat = lat*np.pi/180
    lon = lon*np.pi/180 
    x= np.cos(lat)*np.cos(lon)
    y= np.cos(lat)*np.sin(lon)
    z= np.sin(lat)
    return x,y,z

def sliding_window(df, lag, horizon, split, numOfStations,tData,lon,lat):
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
    # if set == 0:
    samples = int(split)
    # print(samples)
    # if set == 1:
    #     samples = int(split[1] / 45 - split[0] / numOfStations)
    # if set == 2:
    #     samples = int(split[2] / 45 - split[1] / numOfStations)


    dfy = df.drop(['Rain', 'Humidity', 'Pressure', 'WindSpeed', 'WindDir'], axis=1)
    
    stations = numOfStations
    features = 6
    tfeatures = 6

    date_format = "%Y-%m-%d %H:%M:%S"
    # print (df.shape)
    df = df.values.reshape(samples, stations, features)
    dfy = dfy.values.reshape(samples, stations, 1)
    tData = tData [:samples*stations]
    lon = lon[:samples*stations]
    lat = lat[:samples*stations]
    l = len (tData)
    # print(tData)
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

    # print(time)

    time =np.array(time)
    # print(time.shape)

    time = time.reshape(samples, stations, tfeatures)
    # print(time.shape)

    x_offsets = np.sort(np.concatenate((np.arange(-(lag - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (horizon + 1), 1))
    # print('X offsets : ' + str(x_offsets))
    # print('Y offsets : ' + str(y_offsets))
    data = np.expand_dims(df, axis=-1)
    # print(data)
    data = data.reshape(samples, 1, stations, features)
    data = np.concatenate([data], axis=-1)
    # print(data.shape)
    time = time.reshape(samples, 1, stations, tfeatures)
    time = np.concatenate([time], axis=-1)
    # print(time.shape)

    datay = np.expand_dims(dfy, axis=-1)
    datay = datay.reshape(samples, 1, stations, 1)
    datay = np.concatenate([datay], axis=-1)

    x, y , times = [], [], []
    min_t = abs(min(x_offsets))
    max_t = abs(samples - abs(max(y_offsets)))  # Exclusive

    # t is the index of the last observation.
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
    # print(times)

    # print('Shape of Y : ' + str(y.shape))
    return x, y, times

def split_data(data, split):
        return [data[:split[0]], data[split[0]:split[1]], data[split[1]:split[2]]]


def prepare_data_split(sharedConfig, increment, k):
        split = [increment[k] * sharedConfig['num_nodes']['default'],
                 increment[k + 1] * sharedConfig['num_nodes']['default'],
                 increment[k + 2] * sharedConfig['num_nodes']['default']]
        return split

def main(modelConfig,filePath,outputDir):
    print("Generating data")
    numOfStations = modelConfig['num_nodes']['default']
    # split = modelConfig['increments']['default']
    # set = 0
    increments = modelConfig['increments']['default']
    horizon = modelConfig['horizon']['default']
    lag = modelConfig['lag_length']['default']

    data = pd.read_csv(filePath)
    # print(data['Latitude'])
    latitude = np.array(data['Latitude']).flatten()
    # print(latitude)
    longitude = np.array(data['Longitude']).flatten()

    time = data['DateT']
    time_data = time
    lon,lat = data['Longitude'][0:numOfStations], data['Latitude'][0:numOfStations]
    lonlat = np.array([lon, lat])
    lonlat = lonlat.reshape(2,numOfStations).T

    raw_data = data.drop(['StasName', 'DateT', 'Latitude', 'Longitude'], axis=1)

    for h in horizon:
        for c in range(0,len(increments)-2):
            print('Generating data for horizon ' + str(h)+ ' split ' + str(increments[c]))
            splits = prepare_data_split(modelConfig,increments, c)
            sets = split_data(raw_data, splits)
            scaler = NormScaler(raw_data.min(), raw_data.max())
            
            # seq2seq_data, seq2seq_label, context = sliding_window(scaler.transform(raw_data[:increments[c]*numOfStations]), lag, h, increments[c], numOfStations, time_data,longitude,latitude)
            
            train_x, train_y, train_context = sliding_window(scaler.transform(sets[0]), lag, h, increments[c], numOfStations, time_data,longitude,latitude)
            val_x, val_y, val_context = sliding_window(scaler.transform(sets[1]), lag, h, increments[c + 1] - increments[c], numOfStations, time_data,longitude,latitude)
            test_x, test_y, test_context = sliding_window(scaler.transform(sets[2]), lag, h, increments[c + 2] - increments[c + 1], numOfStations, time_data,longitude,latitude)
            
            # num_samples = seq2seq_data.shape[0]
            
            # num_test = round(num_samples * 0.2)
            # num_train = round(num_samples * 0.7)
            # num_val = num_samples - num_test - num_train

            # num_train =int(increments[c] )
            # num_val = int(increments[c+1] - increments[c])
            # num_test = int(increments[c+2] - increments[c+1])
            # print('Number of training samples: {}, validation samples:{}, test samples:{}'.format(train_x[0].shape, val_x[0].shape, test_x[0].shape))

            # if modelConfig['shuffle']['default'] == True :
            #     idx = np.random.permutation(np.arange(num_samples))
            #     seq2seq_data = seq2seq_data[idx]
            #     context = context[idx]
            #     seq2seq_label = seq2seq_label[idx]

            # train_x = seq2seq_data[:num_train]
            # # print(train_x.shape)
            # train_context = context[:num_train]
            # train_y = seq2seq_label[:num_train]
            # # print(train_y.shape)
            
            # val_x = seq2seq_data[num_train:num_train+num_val]
            # val_context = context[num_train:num_train+num_val]
            # val_y = seq2seq_label[num_train:num_train+num_val]

            # test_x = seq2seq_data[num_train+num_val:]
            # test_context = context[num_train+num_val:]
            # test_y = seq2seq_label[num_train+num_val:]
            datasets =[[train_x, train_y, train_context], [val_x, val_y, val_context], [test_x,test_y, test_context]]
            subsets = ['trn','val','test']
            path = outputDir + "/test/horizon_{}".format(h)
            path_ = Path(path)
            path_.mkdir(exist_ok=True,parents=True)
        
            for i, subset in enumerate(subsets):
                with open(path+'/{}_split_{}.pkl'.format(subset,increments[c]), "wb") as f:
                    save_data = {'x': datasets[i][0],
                                'y': datasets[i][1],
                                'context': datasets[i][2]}
                    pickle.dump(save_data,f, protocol = 4)

            with open(path+'/{}.pkl'.format('position_info'), "wb") as f:
                save_data = {'lonlat': lonlat}
                pickle.dump(save_data, f, protocol = 4)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--raw_dataset_dir",
    #     type=str,
    #     default='dataset_release/WeatherBench/datasets'
    # )
    # parser.add_argument(
    #     "--datasets", type=list, default=[
    #         '2m_temperature',
    #         'relative_humidity', 
    #         'component_of_wind', 
    #         'total_cloud_cover'
    #         ], help="dataset name."
    # )
    # parser.add_argument(
    #     "--attri_names", type=list, default=['t2m', 'r','uv10','tcc'], help="data name."
    # )
    # parser.add_argument(
    #     "--output_dirs", type=list, default=['data/temperature','data/humidity', 'data/component_of_wind','data/cloud_cover'], help="Output directory."
    # )
    # parser.add_argument(
    #     "--step_size", type=int, default=24
    # )
    # parser.add_argument(
    #     "--input_seq_len", type=int, default=12
    # )
    # parser.add_argument(
    #     "--output_horizon_len", type=int, default=12
    # )
    # parser.add_argument(
    #     '--start_date', type=str, default='2010-01-01'
    # )
    # parser.add_argument(
    #     '--end_date', type=str, default='2019-01-01'
    # )
    # parser.add_argument(
    #     "--shuffle", type=bool, default=False
    # )
    # args = parser.parse_args()
    
    modelName = 'clcrn'
    with open('../configurations/'+ modelName +'Config.yaml', 'r') as file:
            modelConfig =  yaml.safe_load(file)

    filePath = 'graph.csv'
    outputDir = "../data"

    main(modelConfig,filePath,outputDir)
