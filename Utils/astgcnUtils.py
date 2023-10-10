import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import os
import random
import math
import numpy.linalg as la
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def generate_execute_file_paths(base_path):#, forecast_len, station):
    target_path = f'{base_path}/Targets/target.csv'
    result_path = f'{base_path}/Predictions/result.csv'
    loss_path = f'{base_path}/Predictions/loss.csv'
    actual_vs_predicted_path = f'{base_path}/Predictions/actual_vs_predicted.csv'
    # Make sure all paths exist
    for path in [target_path, result_path, loss_path, actual_vs_predicted_path]:
        create_file_if_not_exists(path)
    return target_path, result_path, loss_path, actual_vs_predicted_path

def get_file_paths(horizon):
    return {
        "yhat": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Predictions/result.csv',
        "target": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Targets/target.csv',
        "metrics": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Metrics/metrics.txt',
        "actual_vs_predicted": f'Results/ASTGCN/{horizon} Hour Forecast/All Stations/Metrics/actual_vs_predicted.txt'
    }

def create_file_if_not_exists(file_path):
    # Get the directory from the file path
    directory = os.path.dirname(file_path)
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # If the file does not exist, create it
    if not os.path.isfile(file_path):
        open(file_path, 'w').close()
        
def dataSplit(split, series):
    train = series[0:split[0]]
    validation = series[split[0]:split[1]]
    test = series[split[1]:split[2]]
    return train, validation, test

def create_X_Y(ts: np.array, lag=1, num_nodes=1, n_ahead=1, target_index=0):
    X, Y = [], []
    if len(ts) - lag - n_ahead + 1 <= 0:
        X.append(ts)
        Y.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead + 1):
            X.append(ts[i:(i + lag)])
            Y.append(ts[i + lag + n_ahead - 1])  
    X, Y = np.array(X), np.array(Y)

    num_samples = len(X)
    time_steps = 10 
    num_samples -= num_samples % time_steps
    X = X[:num_samples]
    Y = Y[:num_samples]
    ### Reshaping to match the ASTGCN model output architecture
    X = np.expand_dims(X, axis=2)
    Y = np.reshape(Y, (Y.shape[0], -1))  # Reshape Y to match the shape of y_pred
    return X, Y

def calculate_laplacian(adj):
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return adj_normalized

def prepare_data_astgcn(split,attribute_data, time_steps, num_nodes, forecast_len):
    train_Attribute, val_attribute, test_attribute = dataSplit(split, attribute_data)
    X_attribute_train, Y_attribute_train = create_X_Y(train_Attribute, time_steps, num_nodes, forecast_len)
    return X_attribute_train, Y_attribute_train

def calculate_laplacian_astgcn(adj, num_nodes):
    # Calculate the normalized Laplacian matrix
    adj = tf.convert_to_tensor(adj, dtype=tf.float32)
    adj = tf.sparse.reorder(tf.sparse.SparseTensor(indices=tf.where(adj != 0),
                                                   values=tf.gather_nd(adj, tf.where(adj != 0)),
                                                   dense_shape=adj.shape))
    adj = tf.sparse.to_dense(adj)
    adj = tf.reshape(adj, [adj.shape[0], adj.shape[0]])
    # Calculate row sums
    rowsum = tf.reduce_sum(adj, axis=1)
    rowsum = tf.maximum(rowsum, 1e-12)  # Add small epsilon to avoid division by zero
    # Calculate the degree matrix
    degree = tf.linalg.diag(1.0 / tf.sqrt(rowsum))
    # Calculate the normalized Laplacian matrix
    laplacian = tf.eye(adj.shape[0]) - tf.matmul(tf.matmul(degree, adj), degree)
    # Rest of normalization that occured in model method before
    laplacian = tf.convert_to_tensor(laplacian, dtype=tf.float32)
    laplacian = tf.reshape(laplacian, [num_nodes, num_nodes])
    return laplacian

def split_data(input_data, increment,k):
    """Splits the input data into training, validation, and test sets."""
    splits = [increment[k], increment[k + 1], increment[k + 2]]
    standardized_train, standardized_validation, standardized_test = dataSplit(splits, input_data)
    return (standardized_train, standardized_validation, standardized_test, splits)

### Normalize input data as a whole
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data
                
# Normalizing splits of data
def normalize_splits(pre_standardize_train, pre_standardize_validation, pre_standardize_test, splits):
    min_val = np.min(pre_standardize_train)
    max_val = np.max(pre_standardize_train)
    # Normalizing the data
    train_data = (pre_standardize_train - min_val) / (max_val - min_val)
    val_data = (pre_standardize_validation - min_val) / (max_val - min_val)
    test_data = (pre_standardize_test - min_val) / (max_val - min_val)
    return train_data, val_data, test_data, splits
                

def normalize_adj(adj):
    """
    Normalize the adjacency matrix.
    """
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    adj_normalized = adj.dot(r_mat_inv).transpose().dot(r_mat_inv)
    return adj_normalized
    
def weight_variable_glorot(input_dim, output_dim, name=""):
    """
    Create a weight variable using the Glorot initialization.
    """
    # Calculate the initialization range
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    # Create a random uniform weight matrix
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                            maxval=init_range, dtype=tf.float32)
    # Return the weight variable
    return tf.Variable(initial,name=name)  

def SMAPE(actual, predicted):
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    diff = np.abs(predicted - actual)
    # Adding a small constant to avoid division by zero
    denominator = np.where(denominator==0, 1e-7, denominator)
    smape = np.mean((diff / denominator) * 100)
    return smape


def smape_std(actual, predicted):
        """
        Calculates the standard deviation of SMAPE values
        Parameters:
            actual - target values
            predicted - output values predicted by model
        Returns:
            std - returns the standard deviation of SMAPE values
        """
        smapes = abs(predicted - actual)  / ((abs(predicted) + abs(actual)) / 2) * 100
        return (np.std(smapes)) 

def MSE(target, pred):
    """
    Calculates the MSE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        mse - returns MSE metric
    """
    return mean_squared_error(target, pred, squared=True)

def generateRandomParameters(config):    
    batch_size = [8,16,32,64,128,256]
    epochs = [10,20, 30,40]
    gru_units = [33,63,93,123]
    lstm_units = [16,32,64,128,256]

    batch = batch_size[random.randint(0,len(batch_size)-1)]
    epoch = epochs[random.randint(0,len(epochs)-1)]
    gru_unit =gru_units[random.randint(0,len(gru_units)-1)]
    lstm_unit =lstm_units[random.randint(0,len(lstm_units)-1)]

    config['batch_size']['default'] = batch
    config['training_epoch']['default'] = epoch
    config['gru_units']['default'] = gru_unit
    config['lstm_neurons']['default'] = lstm_unit

    return [batch, epoch, gru_unit, lstm_unit]

### Perturbation Analysis
def MaxMinNormalization(x,Max,Min):
    x = (x-Min)/(Max-Min)
    return x