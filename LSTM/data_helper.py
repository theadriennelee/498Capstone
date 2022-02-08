import numpy as np
import pandas as pd

skipVal = 11840
endVal = 15787

def set_validation_data(index):
    if skipVal < index < endVal:
        return False
    elif index == 0:
        return False
    else:
        return True

def set_testing_data(index):
    if index >= endVal:
        return False
    elif index == 0:
        return False
    else:
        return True

def load_validation_timeseries(filename):
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
    
    # Load next 20% of data - testing data
    series = pd.read_csv(filename, 
                         sep=',', 
                         header=0, 
                         index_col=0,
                         usecols=['date', 'T1', 'T1_class'],
                         squeeze=True, 
                         skiprows=lambda x: set_validation_data(x))
    
    # Seperate the data into individual arrays
    timestamps = series.index
    data = series.values[:, 0]
    flag = series.values[:, 1]
    print("timestamps " + str(timestamps))
    
    # Initialize dictionary
    flag_dictionary = {}
    for x in range(len(timestamps)):
        if flag[x] == True:
            flag_dictionary[timestamps[x]] = True
    
    return timestamps, data, flag_dictionary

def load_testing_timeseries(filename):
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
    
    # Load next 20% of data - testing data
    series = pd.read_csv(filename, 
                         sep=',', 
                         header=0, 
                         index_col=0,
                         usecols=['date', 'T1'],
                         squeeze=True, 
                         skiprows=lambda x: set_testing_data(x))
    
    timestamps = series.index
    data = series.values
    
    return timestamps, data


def load_timeseries_with_filename(filename, params):
    """
    Load training and validation data.

    Args:
        filename (string): Name of file that contains training and validation data
        params (dict[string]): List of parameters to split the data
    Returns:
        load_timeseries() : Function that will split the data
    """
    
    # Load first 60% of data
    # TODO: PASS IN T1 AS A PARAMETER
    series = pd.read_csv(filename, 
                         sep=',', 
                         header=0, 
                         index_col=0, 
                         usecols=['date', 'T1'],
                         squeeze=True, 
                         skipfooter=skipVal)

    data = series.values
    print("data " + str(data))

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
    print("last raw " + str(last_raw))
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
        if window[0] == 0:
            window[0] = 1
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return normalized_data
