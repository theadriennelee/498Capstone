import numpy as np
import pandas as pd

skipVal = 11841

def load_newtimeseries(filename):
    """
    Load testing data.

    Args:
        filename (string): Name of file that contains testing data
    Returns:
        timestamps (array[int]): Timestamps
        data (array[int]): Testing data
        flag_dictionary (dict[int]): Dictionary that contains T/F flags that
            determine if datapoint is valid or not
    """
    
    # Load last 20% of data - testing data
    series = pd.read_csv(filename, 
                         sep=',', 
                         header=None, 
                         index_col=None, 
                         squeeze=True, 
                         skiprows=skipVal)
    
    # Seperate the data into individual arrays
    timestamps = series.values[:, 0]
    data = series.values[:, 1]
    flag = series.values[:, 2]
    
    # Initialize dictionary
    flag_dictionary = {}
    for x in range(len(timestamps)):
        if flag[x] == True:
            flag_dictionary[timestamps[x]] = True
    
    return timestamps, data, flag_dictionary

def load_timeseries_with_filename(filename, params):
    """
    Load training and validation data.

    Args:
        filename (string): Name of file that contains training and validation data
        params (dict[string]): List of parameters to split the data
    Returns:
        load_timeseries() : Function that will split the data
    """
    
    # Load first 80% of data
    series = pd.read_csv(filename, 
                         sep=',', 
                         header=0, 
                         index_col=0, 
                         squeeze=True, 
                         skipfooter=skipVal)

    data = series.values[:, 0]

    return load_timeseries(data, params)


def load_timeseries(data, params):
    """
    Load timeseries dataset.

    Args:
        data (arr[int]): Data set
        params (dict[string]): List of parameters to split the data
    Returns:
        x_train (arr[int]): x training values normalized
        y_tain (arr[int]): y training values normalized
        x_valid (arr[int]): x validation values normalized
        y_valid (arr[int]): y validation values normalized
        x_valid_raw (arr[int]): x validation values raw
        y_valid_raw (arr[int]): y validation values raw
        last_raw (arr[int]): last window values raw
        last (arr[int]): last window values normalized
        data: list of data
    """

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

    return [x_train, y_train, x_valid, y_valid, x_valid_raw, y_valid_raw, last_raw, last, data]

def normalize_windows(window_data):
    """
    Normalize data

    Args:
        window_data (arr[int]): List of values to be normalized

    Returns:
        normalized_data (arr[int]): List of normalized data
    """

    normalized_data = []
    for window in window_data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return normalized_data
