import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta


def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)  # (T, N)
    
    # Generate time features
    T = feat.shape[0]
    start_time = datetime(2012, 5, 1, 0, 0, 0)
    time_features = []
    
    for i in range(T):
        current_time = start_time + timedelta(minutes=5 * i)
        hour = current_time.hour / 23.0  # Normalize hour to [0, 1]
        day = current_time.weekday() / 6.0  # Normalize day (0=Monday, 6=Sunday) to [0, 1]
        time_features.append([hour, day])
    
    time_features = np.array(time_features, dtype=dtype)  # (T, 2)
    
    # Combine features: (T, N, 3) where features are [speed, hour, day]
    N = feat.shape[1]
    combined_feat = np.zeros((T, N, 3), dtype=dtype)
    combined_feat[:, :, 0] = feat  # Traffic speed
    combined_feat[:, :, 1] = time_features[:, 0:1]  # Hour (broadcasted to all nodes)
    combined_feat[:, :, 2] = time_features[:, 1:2]  # Day (broadcasted to all nodes)
    
    return combined_feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix (T, N, 3) with [speed, hour, day]
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        # Only normalize the speed feature (index 0)
        max_val = np.max(data[:, :, 0])
        data = data.copy()
        data[:, :, 0] = data[:, :, 0] / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(train_data[i : i + seq_len])  # (seq_len, N, 3)
        # Only predict traffic speed (feature index 0)
        train_Y.append(train_data[i + seq_len : i + seq_len + pre_len, :, 0])  # (pre_len, N)
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(test_data[i : i + seq_len])  # (seq_len, N, 3)
        # Only predict traffic speed (feature index 0)
        test_Y.append(test_data[i + seq_len : i + seq_len + pre_len, :, 0])  # (pre_len, N)
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
