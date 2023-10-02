#add_window
import numpy as np
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from sklearn.preprocessing import MinMaxScaler
import Utils.gwnUtils as gwnUtils


def Add_Window_Horizon(data, window, horizon, single):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    # if single:
    #     while index < end_index:
    #         X.append(data[index:index+window])
    #         Y.append(data[index+window+horizon-1:index+window+horizon]["temperature"])
    #         index = index + 1
    # else:
    #     while index < end_index:
    #         X.append(data[index:index+window])
    #         data_temperature = data[:, :, 5]
         
    #         # print("this is data shape")
    #         # print(data_temperature[index+window+horizon-1:index+window+horizon].shape)
            
    #         Y.append(data_temperature[index+window+horizon-1:index+window+horizon].reshape((1, 45, 1)))

    #         index = index + 1



    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append( data[index+window:index+window+horizon, :, -1:] )
            index = index + 1

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


#dataloader
import torch
import numpy as np
import torch.utils.data
# from .add_window import Add_Window_Horizon
# from lib.load_dataset import load_st_dataset
# from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def min_max(data):
    norm = MinMaxScaler().fit(data)
    data = norm.transform(data)
    return data

def normalize_dataset(data, normalizer, column_wise):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        # print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:

            mean = np.mean(data, axis=(0, 1))
            std = np.std(data, axis=(0, 1))
            # mean = data.mean(axis=0, keepdims=True)
            # std = data.std(axis=0, keepdims=True)
        else:


            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        # print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    print("new shape")
    print(data.shape)
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def min_max(train, validation, test):
    """
    Performs MinMax scaling on the train, validation and test sets using the train data min and max.
    Parameters:
        train, validation, test - train, validation and test data sets
    Returns:
        train, validation, test - returns the scaled train, validation and test sets
    """

    norm = MinMaxScaler().fit(train)

    train_data = norm.transform(train)
    val_data = norm.transform(validation)
    test_data = norm.transform(test)

    return train_data, val_data, test_data



def get_dataloader(horizon, k, increment, n_stations, agcrnConfig, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(n_stations)        # B, N, D     
    
    #train data prenormalised
    split = [increment[k], increment[k + 1], increment[k + 2] ]
    data_train = data[:split[0]]
    
    #min/max of training set
    min_train = np.min(data_train, axis=(0,1))
    max_train = np.max(data_train, axis=(0,1))

    #normalise all data
    scaler = gwnUtils.NormScaler(min_train, max_train)
    data = scaler.transform(data)
    
    #split data
    split = [increment[k], increment[k + 1], increment[k + 2] ]
    data_train = data[:split[0]]
    data_val = data[split[0]:split[1]]
    data_test = data[split[1]:split[2]]


    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, agcrnConfig['lag']['default'], horizon, single)  
    x_val, y_val = Add_Window_Horizon(data_val, agcrnConfig['lag']['default'], horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, agcrnConfig['lag']['default'], horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, agcrnConfig['batch_size']['default'], shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, agcrnConfig['batch_size']['default'], shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, agcrnConfig['batch_size']['default'], shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler #, min_train, max_train





#load_dataset
import os
import numpy as np
import pandas as pd


def load_st_dataset(n_stations):
    #output B, N, D
 
    data = pd.read_csv('DataNew/Graph Neural Network Data/Graph Station Data/graph.csv')
    data = data.drop(['StasName', 'DateT', 'Latitude', 'Longitude'], axis=1)  #added latitude and longitude
    data = np.array(data)

    n_rows = data.shape[0]
    data = np.reshape(data, (int(n_rows/n_stations), n_stations, 6))

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)

    print("Data Shape:   " + str(data.shape))

    return data


#logger
import os
import logging
from datetime import datetime

def get_logger(root, name=None, debug=True):
    #when debug is true, show DEBUG and INFO in screen
    #when debug is false, show DEBUG in file and info in both screen&file
    #INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    #critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger







#metrics

'''
Always evaluate the model with MAE, RMSE, MAPE, RRSE, PNBI, and oPNBI.
Why add mask to MAE and RMSE?
    Filter the 0 that may be caused by error (such as loop sensor)
Why add mask to MAPE and MARE?
    Ignore very small values (e.g., 0.5/0.5=100%)
'''
import numpy as np
import torch

def MAE_torch(pred, true, mask_value=None):
    if mask_value != "None":
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != "None":
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != "None":
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def RRSE_torch(pred, true, mask_value=None):
    if mask_value != "None":
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))

def CORR_torch(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(dim=1)
        true = true.transpose(1, 2).unsqueeze(dim=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(dim=dims)
    true_mean = true.mean(dim=dims)
    pred_std = pred.std(dim=dims)
    true_std = true.std(dim=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(dim=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != "None":
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def PNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    indicator = torch.gt(pred - true, 0).float()
    return indicator.mean()

def oPNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    bias = (true+pred) / (2*true)
    return bias.mean()

def MARE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.div(torch.sum(torch.abs((true - pred))), torch.sum(true))

def SMAPE_torch(pred, true, mask_value=None):
    if mask_value != "None":
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred)/(torch.abs(true)+torch.abs(pred)))


def MAE_np(pred, true, mask_value=None):
    if mask_value != "None":
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE

def RMSE_np(pred, true, mask_value=None):
    if mask_value != "None":
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE

#Root Relative Squared Error
def RRSE_np(pred, true, mask_value=None):
    if mask_value != "None":
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    mean = true.mean()
    return np.divide(np.sqrt(np.sum((pred-true) ** 2)), np.sqrt(np.sum((true-mean) ** 2)))

def MAPE_np(pred, true, mask_value=None):
    if mask_value != "None":
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    epsilon = 1e-10
    return np.mean(np.absolute(np.divide((true - pred), true+epsilon)))

def PNBI_np(pred, true, mask_value=None):
    #if PNBI=0, all pred are smaller than true
    #if PNBI=1, all pred are bigger than true
    if mask_value != "None":
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = pred-true
    indicator = np.where(bias>0, True, False)
    return indicator.mean()

def oPNBI_np(pred, true, mask_value=None):
    #if oPNBI>1, pred are bigger than true
    #if oPNBI<1, pred are smaller than true
    #however, this metric is too sentive to small values. Not good!
    if mask_value != "None":
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = (true + pred) / (2 * true)
    return bias.mean()

def MARE_np(pred, true, mask_value=None):
    if mask_value != "None":
        mask = np.where(true> (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.divide(np.sum(np.absolute((true - pred))), np.sum(true))

def CORR_np(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        #B, N
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        #np.transpose include permute, B, T, N
        pred = np.expand_dims(pred.transpose(0, 2, 1), axis=1)
        true = np.expand_dims(true.transpose(0, 2, 1), axis=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(0, 1, 2, 3)
        true = true.transpose(0, 1, 2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(axis=dims)
    true_mean = true.mean(axis=dims)
    pred_std = pred.std(axis=dims)
    true_std = true.std(axis=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(axis=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation

def All_Metrics(pred, true, mask1, mask2):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = MAE_np(pred, true, mask1)
        rmse = RMSE_np(pred, true, mask1)
        mape = MAPE_np(pred, true, mask2)
        rrse = RRSE_np(pred, true, mask1)
        corr = 0
        #corr = CORR_np(pred, true, mask1)
        #pnbi = PNBI_np(pred, true, mask1)
        #opnbi = oPNBI_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae  = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        mape = MAPE_torch(pred, true, mask2)
        rrse = RRSE_torch(pred, true, mask1)
        corr = CORR_torch(pred, true, mask1)
        #pnbi = PNBI_torch(pred, true, mask1)
        #opnbi = oPNBI_torch(pred, true, mask2)
    else:
        raise TypeError
    
    rmse = math.sqrt(mse(true, pred))

    return mae, rmse, mape, rrse, corr


def mse(target, pred):
    """
    Calculates the MSE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        mse - returns MSE metric
    """
    # print(target.shape)
    return mean_squared_error(target, pred, squared=True)

def SIGIR_Metrics(pred, true, mask1, mask2):
    rrse = RRSE_torch(pred, true, mask1)
    corr = CORR_torch(pred, true, 0)
    return rrse, corr




#normalization
import numpy as np
import torch


class NScaler(object):
    def transform(self, data):
        return data
    def inverse_transform(self, data):
        return data

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

class MinMax01Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min

class ColumnMinMaxScaler():
    #Note: to use this scale, must init the min and max with column min and column max
    def __init__(self, min, max):
        self.min = min
        self.min_max = max - self.min
        self.min_max[self.min_max==0] = 1
    def transform(self, data):
        print(data.shape, self.min_max.shape)
        return (data - self.min) / self.min_max

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min_max = torch.from_numpy(self.min_max).to(data.device).type(torch.float32)
            self.min = torch.from_numpy(self.min).to(data.device).type(torch.float32)
        #print(data.dtype, self.min_max.dtype, self.min.dtype)
        return (data * self.min_max + self.min)

def one_hot_by_column(data):
    #data is a 2D numpy array
    len = data.shape[0]
    for i in range(data.shape[1]):
        column = data[:, i]
        max = column.max()
        min = column.min()
        #print(len, max, min)
        zero_matrix = np.zeros((len, max-min+1))
        zero_matrix[np.arange(len), column-min] = 1
        if i == 0:
            encoded = zero_matrix
        else:
            encoded = np.hstack((encoded, zero_matrix))
    return encoded


def minmax_by_column(data):
    # data is a 2D numpy array
    for i in range(data.shape[1]):
        column = data[:, i]
        max = column.max()
        min = column.min()
        column = (column - min) / (max - min)
        column = column[:, np.newaxis]
        if i == 0:
            _normalized = column
        else:
            _normalized = np.hstack((_normalized, column))
    return _normalized






#TrainInits
import torch
import random
import numpy as np

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_device(opt):
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt

def init_optim(model, opt):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),lr=opt.lr_init)

def init_lr_scheduler(optim, opt):
    '''
    Initialize the learning rate scheduler
    '''
    #return torch.optim.lr_scheduler.StepLR(optimizer=optim,gamma=opt.lr_scheduler_rate,step_size=opt.lr_scheduler_step)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=opt.lr_decay_steps,
                                                gamma = opt.lr_scheduler_rate)

def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')

def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024*1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024*1024.)
    return allocated_memory, cached_memory
    #print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))


def generateRandomParameters(config):
    """
    Generates a random configuration of hyper-parameters from the pre-defined search space.

    Parameters:
        config -  Conguration file of initial parameters.

    Returns:
        config - list of HPO parameters
    """
    batch_size = [32, 64, 128, 256] 
    rnn_units = [32, 64, 128]
    lag = [12, 24, 48] 
    num_layers = [1, 2, 3]
    epochs = [10, 20, 30, 40, 50]  
    lr_init = [0.0001, 0.001, 0.01]
    embed_dim = [5, 10, 15] 


    batch = batch_size[random.randint(0, len(batch_size)-1)]
    lag = lag[random.randint(0, len(lag)-1)]
    num_layers = num_layers[random.randint(0, len(num_layers)-1)]
    epoch = epochs[random.randint(0, len(epochs)-1)]
    rnn_units = rnn_units[random.randint(0, len(rnn_units)-1)]
    embed_dim = embed_dim[random.randint(0, len(embed_dim)-1)]
    lr_init = lr_init[random.randint(0, len(lr_init)-1)]


    config['embed_dim']['default'] = embed_dim
    config['lr_init']['default'] = lr_init
    config['batch_size']['default'] = batch
    config['lag']['default'] = lag
    config['num_layers']['default'] = num_layers
    config['epochs']['default'] = epoch
    config['rnn_units']['default'] = rnn_units


    return config