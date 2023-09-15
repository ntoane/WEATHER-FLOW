import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import torch
from scipy.sparse import linalg
from sklearn.metrics import r2_score


class StandardScaler:
    """from tensorflow.python.eafrom tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizerger import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    print("-------------------------------------------------------------------------")
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def nbhd2SparseIdx(nbhds):
    nbhds = np.array(nbhds)
    nbhds_flat = nbhds.flatten()
    element_num = nbhds_flat[nbhds_flat!=-1].shape[0]
    sparse_indices = []

    for nbhd in nbhds:
        nbhd = nbhd[nbhd != -1]
        pairs = np.zeros((nbhd.shape[0], 2))
        pairs[:, 0] = nbhd[0]
        pairs[:, 1] = nbhd
        sparse_indices.append(pairs)
    
    sparse_indices = np.array(sparse_indices)
    sparse_indices = sparse_indices.reshape(element_num, 2)
    sparse_indices = sparse_indices.T
    return torch.LongTensor(sparse_indices)

def latlon2xyz(lat,lon):
    x=-np.cos(lat)*np.cos(lon)
    y=-np.cos(lat)*np.sin(lon)
    z=np.sin(lat)
    return x,y,z

def xyz2latlon(x,y,z):
    lat=np.arcsin(z)
    lon=np.arctan2(-y,-x)
    return lat,lon

"""Transformations for samples of atmospheric rivers and tropical cyclones dataset.
"""

class ToTensor:
    """Convert raw data and labels to PyTorch tensor.
    """

    def __call__(self, item):
        """Function call operator to change type.

        Args:
            item (:obj:`numpy.array`): Numpy array that needs to be transformed.
        Returns:
            :obj:`torch.Tensor`: Sample of size (vertices, features).
        """
        return torch.tensor(item)


class Permute:
    """Permute first and second dimension.
    """

    def __call__(self, item):
        """Permute first and second dimension.

        Args:
            item (:obj:`torch.Tensor`): Torch tensor that needs to be transformed.

        Returns:
            :obj:`torch.Tensor`: Permuted input tensor.
        """
        return item.permute(1, 0)


class Normalize:
    """Normalize using mean and std.
    """

    def __init__(self, mean, std):
        """Initialization

        Args:
            mean (:obj:`numpy.array`): means of each feature
            std (:obj:`numpy.array`): standard deviations of each feature
        """
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)

    def __call__(self, item):
        """
        Args:
            item (:obj:`torch.Tensor`): Sample of size (vertices, features) to be normalized on its features.

        Returns:
            :obj:`torch.Tensor`: Normalized input tensor.
        """
        return (item - self.mean) / self.std


class Stack:
    """Stack images in torch tensor.
    """

    def __init__(self, dimension=0):
        """Initialization

        Args:
            dimension int: The dimension to be used for stacking.
        """
        self.dimension = dimension

    def __call__(self, item):
        """Stack images in torch tensor.

        Args:
            item (:obj:`torch.Tensor`): Torch tensor that needs to be transformed.

        Returns:
            :obj:`torch.Tensor`: Stacked input tensor.
        """
        return torch.stack(item, dim=self.dimension)

"""Get Means and Standard deviations for all features of a dataset.
"""

def stats_extractor(dataset):
    """Iterates over a dataset object
    It is iterated over so as to calculate the mean and standard deviation.

    Args:
        dataset (:obj:`torch.utils.data.dataloader`): dataset object to iterate over

    Returns:
        :obj:numpy.array, :obj:numpy.array : computed means and standard deviation
    """

    F, V = torch.Tensor(dataset[0][0]).shape
    summing = torch.zeros(F)
    square_summing = torch.zeros(F)
    total = 0

    for item in dataset:
        item = torch.Tensor(item[0])
        summing += torch.sum(item, dim=1)
        total += V

    means = torch.unsqueeze(summing / total, dim=1)

    for item in dataset:
        item = torch.Tensor(item[0])
        square_summing += torch.sum((item - means) ** 2, dim=1)

    stds = np.sqrt(square_summing / (total - 1))

    return torch.squeeze(means, dim=1).numpy(), stds.numpy()

def stats_extractor2(dataset):
    """Iterates over a dataset object
    It is iterated over so as to calculate the mean and standard deviation.

    Args:
        dataset (:obj:`torch.utils.data.dataloader`): dataset object to iterate over

    Returns:
        :obj:numpy.array, :obj:numpy.array : computed means and standard deviation
    """

    V,F = torch.Tensor(dataset[0][0][0]).shape
    summing = torch.zeros(F)
    square_summing = torch.zeros(F)
    total = 0

    for item in dataset:
        item = torch.Tensor(item[0])
        summing += torch.sum(item, dim=0)
        total += V

    means = torch.unsqueeze(summing / total, dim=0)

    for item in dataset:
        item = torch.Tensor(item[0])
        square_summing += torch.sum((item - means) ** 2, dim=0)

    stds = np.sqrt(square_summing / (total - 1))

    return torch.squeeze(means, dim=0).numpy(), stds.numpy()

def r2_np(preds, labels):
    return r2_score(labels, preds)

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    idx = np.where(labels>1e-2)
    preds = preds[idx]
    labels = labels[idx]
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    idx = np.where(labels>1e-2)
    preds = preds[idx]
    labels = labels[idx]
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    idx = np.where(labels>1e-2)
    preds = preds[idx]
    labels = labels[idx]
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)



def calculate_metrics(df_pred, df_test, null_val):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    mape = masked_mape_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    mae = masked_mae_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    rmse = masked_rmse_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), null_val=null_val)
    return mae, mape, rmse