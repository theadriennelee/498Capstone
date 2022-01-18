import numpy as np
import pandas as pd

def load_newtimeseries(filename):
    series = pd.read_csv(filename, sep=',', header=None, index_col=None, squeeze=True, skiprows=11841)
    # skiprows
    timestamps = series.values[:, 0]
    data = series.values[:, 1]
    flag = series.values[:, 2]
    flag_dictionary = {}
    for x in range(len(timestamps)):
        if flag[x] == True:
            flag_dictionary[timestamps[x]] = True
    print("data read " + str(data))
    
    return data, flag_dictionary

def load_timeseries_with_filename(filename, params):
    series = pd.read_csv(filename, sep=',', header=0, index_col=0, squeeze=True, skipfooter=11841)
    data = series.values[:, 0]

    return load_timeseries(data, params)


def load_timeseries(data, params):
    """Load time series dataset"""

    adjusted_window = params['window_size']+ 1

    # Split data into windows
    raw = []
    for index in range(len(data) - adjusted_window):
        raw.append(data[index: index + adjusted_window])

    # Normalize data
    result = normalize_windows(raw)

    raw = np.array(raw)
    result = np.array(result)

    # Split the input dataset into train and validation
    split_ratio = round(params['train_test_split'] * result.shape[0])
    train = result[:int(split_ratio), :]
    np.random.shuffle(train)

    # x_train and y_train, for training
    x_train = train[:, :-1]
    y_train = train[:, -1]

    # x_valid and y_valid, for validation
    x_valid = result[int(split_ratio):, :-1]
    y_valid = result[int(split_ratio):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 
    x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))  

    x_valid_raw = raw[int(split_ratio):, :-1]
    y_valid_raw = raw[int(split_ratio):, -1]

    # Last window, for next time stamp prediction
    last_raw = [data[-params['window_size']:]]
    last = normalize_windows(last_raw)
    last = np.array(last)
    last = np.reshape(last, (last.shape[0], last.shape[1], 1))

    return [x_train, y_train, x_valid, y_valid, x_valid_raw, y_valid_raw, last_raw, last]

def normalize_windows(window_data):
    """Normalize data"""

    normalized_data = []
    for window in window_data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return normalized_data

def update_input_vales(data, params):
    adjusted_window = params['window_size']+ 1

    # Split data into windows
    raw = []
    for index in range(len(data) - adjusted_window):
        raw.append(data[index: index + adjusted_window])

    # Normalize data
    result = normalize_windows(raw)

    raw = np.array(raw)
    result = np.array(result)

    # Split the input dataset into train and test
    split_ratio = round(params['train_test_split'] * result.shape[0])
    train = result[:int(split_ratio), :]
    np.random.shuffle(train)

    # x_train and y_train, for training
    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_valid_raw = raw[int(split_ratio):, -1]

    return [x_train, y_train, y_valid_raw]

