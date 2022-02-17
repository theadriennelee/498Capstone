# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 12:20:53 2021

"""

# univariate cnn example
from numpy import array
from tcn import compiled_tcn
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# parameters
epochs = 10
batch_size = 4
window_size = 6
train_test_split = 0.8
validation_split = 0.1
hidden_unit = 10

n_steps = 100
n_features = 1

# splitting parameters
skipVal = 11840
endVal = 19734
midVal = 15787

update_frequency = 100
threshold = 0.03

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    """
    Split a univariate sequence into samples

    Args:
        sequence (array): Array of datapoints
        n_steps (int): Length of samples
    Returns:
        array(x) (array): Array of samples of length n_steps - 1
        array(y) (array): Array of last sample of given sequence
    """

    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def get_train_data(filename):
    """
    Load training data

    Args:
        filename (string): Name of .csv file to load data from
    Returns:
        dataset (array): 2D array of training data which includes timestamp and T1
    """
    dataset = pd.read_csv(filename, index_col=None, squeeze=True, skiprows=[i for i in range(skipVal, endVal)])
    # for threshold testing T1 
    dataset = dataset.drop(['Unnamed: 0','date','T1_class'],axis=1)
    # uncomment below to drop columns from Gaussian_T1.csv file 
    #dataset = dataset.drop(['Unnamed: 0','date','Appliances','lights','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5',
    #    'T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint',
    #    'rv1','rv2','T1_class'],axis=1)
    dataset.to_csv('train_parsed.csv')
    return (dataset)


def set_validation_data(index):
    """
    Helper function to read validation data

    Args:
        index (int): Index of datapoint
    Returns:
        boolean: Returns false if datapoint is a validation point
    """
    if skipVal < index < midVal:
        return False
    elif index == 0:
        return False
    else:
        return True

def set_testing_data(index):
    """
    Helper function to read validation data

    Args:
        index (int): Index of datapoint
    Returns:
        boolean: Returns false if datapoint is a validation point
    """
    if index >= midVal:
        return False
    elif index == 0:
        return False
    else:
        return True

def get_validation_data(filename):
    """
    Load validation data

    Args:
        filename (string): Name of .csv file to load data from
    Returns:
        dataset (array): 2D array of testing data which includes timestamp and T1
    """
    series = pd.read_csv(filename, sep=',', header=0, index_col=0, usecols=['date', 'T1'], squeeze=True, skiprows=lambda x: set_validation_data(x))
    # testset = pd.read_csv(filename, index_col=None, squeeze=True, skiprows=lambda x: set_validation_data(x))
    # for threshold testing T1 
    # testset = testset.drop(['Unnamed: 0','date','T1_class'],axis=1)
    # uncomment below to drop columns from Gaussian_T1.csv file 
    #testset = testset.drop(['Unnamed: 0','date','Appliances','lights','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5',
    #    'T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint',
    #   'rv1','rv2','T1_class'],axis=1)
    # testset.to_csv('test_parsed.csv')
    return series

def get_testing_data(filename):
    """
    Load validation data

    Args:
        filename (string): Name of .csv file to load data from
    Returns:
        dataset (array): 2D array of testing data which includes timestamp and T1
    """
    series = pd.read_csv(filename, sep=',', header=0, index_col=0, usecols=['date', 'T1'], squeeze=True, skiprows=lambda x: set_testing_data(x))
    # testset = pd.read_csv(filename, index_col=None, squeeze=True, skiprows=lambda x: set_validation_data(x))
    # for threshold testing T1 
    # testset = testset.drop(['Unnamed: 0','date','T1_class'],axis=1)
    # uncomment below to drop columns from Gaussian_T1.csv file 
    #testset = testset.drop(['Unnamed: 0','date','Appliances','lights','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5',
    #    'T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint',
    #   'rv1','rv2','T1_class'],axis=1)
    series.to_csv('test_parsed.csv')
    return series
        
def load_flag_dictionary(filename):
    """
    Create flag dictionary

    Args:
        filename (string): Name of .csv file that contains testing data
    Returns:
        timestamps (array[int]): Timestamps
        data (array[int]): Testing data
        flag_dictionary (dict[int]): Dictionary that contains T/F flags that
            determine if datapoint is valid or not
    """
    testset = pd.read_csv(filename, sep=',', header=0, index_col=0, squeeze=True, skiprows=lambda x: set_validation_data(x), usecols=['date', 'T1_class'])
    # for threshold testing T1 
    # testset = testset.drop(['date'],axis=1)
    # uncomment below to drop columns from Gaussian_T1.csv file 
    #testset = testset.drop(['date','Appliances','lights','T1','RH_1', 'T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5',
    #    'T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint',
    #    'rv1','rv2'],axis=1)

    # Seperate the data into individual arrays
    timestamps = testset.index
    flag = testset.values

    # Initialize dictionary
    flag_dictionary = {}
    for x in range(len(timestamps)):
        if flag[x] == True:
            flag_dictionary[timestamps[x]] = True

    return timestamps, flag_dictionary 

def load_test_timestamps(filename):
    """
    Create flag dictionary

    Args:
        filename (string): Name of .csv file that contains testing data
    Returns:
        timestamps (array[int]): Timestamps
        data (array[int]): Testing data
        flag_dictionary (dict[int]): Dictionary that contains T/F flags that
            determine if datapoint is valid or not
    """
    testset = pd.read_csv(filename, sep=',', header=0, index_col=0, squeeze=True, skiprows=lambda x: set_testing_data(x), usecols=['date', 'T1_class'])
    # for threshold testing T1 
    # testset = testset.drop(['date'],axis=1)
    # uncomment below to drop columns from Gaussian_T1.csv file 
    #testset = testset.drop(['date','Appliances','lights','T1','RH_1', 'T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5',
    #    'T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint',
    #    'rv1','rv2'],axis=1)

    # Seperate the data into individual arrays
    timestamps = testset.index
    flag = testset.values

    # Initialize dictionary
    flag_dictionary = {}
    for x in range(len(timestamps)):
        if flag[x] == True:
            flag_dictionary[timestamps[x]] = True

    return timestamps, flag_dictionary   


def check_abnormal_data(next_timestamp_raw, flag_dictionary, current_data, 
                        timestamp, true_positive, true_negative, 
                        false_positive, false_negative, valid_data, predicted_invalid, flags):
    """
    Compare new data against the predicted timestamp.
    Places the timestamp of the data in true_positive, true_negative, 
        false_positive or false_negative.
    Args:
        next_timestamp_raw (int): Predicted next timestamp
        flag_dictionary (dict[int]): Dictionary that contains T/F flags that
            determine if datapoint is valid or not
        current_data (int): Data to be checked
        timestamp (int): Timestamp of the data to be checked
        true_positive (arr[int]): List of true positive timestamps
        true_negative (arr[int]): List of true negative timestamps
        false_positive (arr[int]): List of false positive timestamps
        false_negative (arr[int]): List of flase negative timestamps
        valid_data (arr[int]): List of valid data
    Returns:
        valid_data (arr[int]): List of valid data
        true_positive (arr[int]): List of true positive timestamps
        true_negative (arr[int]): List of true negative timestamps
        false_positive (arr[int]): List of false positive timestamps
        false_negative (arr[int]): List of flase negative timestamps
        output_flag (boolean): Returns false if negative and true if positive
    """
    
    # Calculate thresholds
    upper_limit = next_timestamp_raw * (1 + threshold)
    lower_limit = next_timestamp_raw * (1 - threshold)

    print('Next_time_stamp_raw ' + str(next_timestamp_raw))
    print('Upper limit ' + str(upper_limit))
    print('Lower limit ' + str(lower_limit))
    print('current data array', current_data)
    
    # Compare data
    if float(current_data) > upper_limit or float(current_data) < lower_limit:
        flags.append(True)
        predicted_invalid.append([timestamp, current_data])
        if timestamp in flag_dictionary:
            true_negative.append(timestamp)
        else:
            false_negative.append(timestamp)
    else:
        flags.append(False)
        if timestamp in flag_dictionary:
            false_positive.append(timestamp)
        else:
            true_positive.append(timestamp)
        valid_data.append(current_data)
    
    return valid_data, predicted_invalid, flags, true_positive, true_negative, false_positive, false_negative

def check_abnormal_test_data(next_timestamp_raw, current_data, 
                        timestamp, predicted_invalid, flags, valid_data):
    """
    Compare new data against the predicted timestamp.
    Places the timestamp of the data in true_positive, true_negative, 
        false_positive or false_negative.
    Args:
        next_timestamp_raw (int): Predicted next timestamp
        flag_dictionary (dict[int]): Dictionary that contains T/F flags that
            determine if datapoint is valid or not
        current_data (int): Data to be checked
        timestamp (int): Timestamp of the data to be checked
        true_positive (arr[int]): List of true positive timestamps
        true_negative (arr[int]): List of true negative timestamps
        false_positive (arr[int]): List of false positive timestamps
        false_negative (arr[int]): List of flase negative timestamps
        valid_data (arr[int]): List of valid data
    Returns:
        valid_data (arr[int]): List of valid data
        true_positive (arr[int]): List of true positive timestamps
        true_negative (arr[int]): List of true negative timestamps
        false_positive (arr[int]): List of false positive timestamps
        false_negative (arr[int]): List of flase negative timestamps
        output_flag (boolean): Returns false if negative and true if positive
    """
    
    # Calculate thresholds
    upper_limit = next_timestamp_raw * (1 + threshold)
    lower_limit = next_timestamp_raw * (1 - threshold)

    print('Next_time_stamp_raw ' + str(next_timestamp_raw))
    print('Upper limit ' + str(upper_limit))
    print('Lower limit ' + str(lower_limit))
    
    # Compare data
    if float(current_data) > upper_limit or float(current_data) < lower_limit:
        flags.append(True)
        predicted_invalid.append([timestamp, current_data])
    else:
        flags.append(False)
        valid_data.append(current_data)
    
    return valid_data, predicted_invalid, flags

def calculate_mean_squared(predicted_val, actual_val):
    """

    Calculates mean squared error

    Args:
        predicted_val (int): Predicted value
        actual_val (int): Actual value

    Returns:
        mse (int): Mean squared error

    """

    mse = pow((predicted_val - actual_val), 2)
    return mse

def predict_next_timestamp(model, raw_seq):
    """

    Predicts next timestamp

    Args:
        model (model): CNN model
        raw_seq (arr): Array of training/validation data

    Returns:
        yhat: Next predicted value

    """
    x_input = []
    for x in range(n_steps):
        x_input.append(raw_seq[len(raw_seq) - n_steps + x - 1])
    x_input = array(x_input)
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    return yhat

def validate(filename, raw_seq):

    model = load_model('initial_model.h5')

    validation_data = array(get_validation_data(filename))
    timestamps, flag_dictionary = load_flag_dictionary(filename)

    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []

    mse_values = []

    valid_data = [] 
    predicted_invalid = []

    flags = []
    

    #next step: continuously update the model to keep changing as it learns 
    i = 0 
    for data in validation_data:

        # shape data
        temp_data = np.append(raw_seq, data)
        X_update, y_update = split_sequence(temp_data, n_steps)
        X_update = X_update.reshape((X_update.shape[0], X_update.shape[1], n_features))

        # predict next time stamp
        yhat_update = predict_next_timestamp(model, temp_data)

        mse = calculate_mean_squared(yhat_update, data)
        mse_values.append(mse[0][0])
        
        valid_data, predicted_invalid, flags, true_positive, true_negative, false_positive,\
            false_negative = check_abnormal_data(yhat_update, 
                                                    flag_dictionary, 
                                                    data, 
                                                    timestamps[i], 
                                                    true_positive, 
                                                    true_negative, 
                                                    false_positive, 
                                                    false_negative, 
                                                    valid_data, predicted_invalid, flags)

        # only update raw_seq is the data was found to be valid
        raw_seq = temp_data

        # if there is enough data to update the model  
        if len(valid_data) > update_frequency:
            X, y = split_sequence(raw_seq, n_steps)
            X = X.reshape((X.shape[0], X.shape[1], n_features))
            model.fit(X,y,epochs=5,verbose=1)
            model.save('initial_model_validation.h5')
            valid_data = []
        i = i + 1

    # Export data
    tp = pd.DataFrame(true_positive)
    tp.to_csv('truePositive.csv')
    tn = pd.DataFrame(true_negative)
    tn.to_csv('trueNegative.csv')
    fp = pd.DataFrame(false_positive)
    fp.to_csv('falsePositive.csv')
    fn = pd.DataFrame(false_negative)
    fn.to_csv('falseNegative.csv')
    mseExport = pd.DataFrame(mse_values)
    mseExport.to_csv('mseExport.csv')
    pred_inv = pd.DataFrame(predicted_invalid)
    pred_inv.to_csv("predicted_invalid.csv") 

    data_export_vals = {
        'Date': timestamps,
        'T1': validation_data,
        'T1_e': mse_values,
        'T1_c': flags
    }

    data_export = pd.DataFrame(data_export_vals)
    data_export.to_csv("T1_report.csv", index=False)

    # Plot graph: Current Data vs MSE
    plt.figure()
    plt.subplot(111)
    plt.plot(validation_data, label='Current Data')
    plt.plot(mse_values, label='mse')	
    plt.legend()
    plt.show()

    return raw_seq

def test(filename, raw_seq):

    model = load_model('initial_model_validation.h5')

    testing_data = array(get_testing_data(filename))  
    timestamps, flag_dictionary = load_test_timestamps(filename)

    mse_values = []

    valid_data = [] 
    predicted_invalid = []

    flags = []

    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []
    

    #next step: continuously update the model to keep changing as it learns 
    i = 0 
    for data in testing_data:

        # shape data
        temp_data = np.append(raw_seq, data)
        X_update, y_update = split_sequence(temp_data, n_steps)
        X_update = X_update.reshape((X_update.shape[0], X_update.shape[1], n_features))

        # predict next time stamp
        yhat_update = predict_next_timestamp(model, temp_data)

        mse = calculate_mean_squared(yhat_update, data)
        mse_values.append(mse[0][0])
        
        # valid_data, predicted_invalid, flags = check_abnormal_test_data(yhat_update,  
        #                                             data, 
        #                                             timestamps[i], 
        #                                             predicted_invalid,
        #                                             flags,
        #                                             valid_data)

        valid_data, predicted_invalid, flags, true_positive, true_negative, false_positive,\
            false_negative = check_abnormal_data(yhat_update, 
                                                    flag_dictionary, 
                                                    data, 
                                                    timestamps[i], 
                                                    true_positive, 
                                                    true_negative, 
                                                    false_positive, 
                                                    false_negative, 
                                                    valid_data, predicted_invalid, flags)

        # only update raw_seq is the data was found to be valid
        raw_seq = temp_data

        # if there is enough data to update the model  
        if len(valid_data) > update_frequency:
            X, y = split_sequence(raw_seq, n_steps)
            X = X.reshape((X.shape[0], X.shape[1], n_features))
            model.fit(X,y,epochs=5,verbose=1)
            model.save('initial_model_validation.h5')
            valid_data = []
        i = i + 1

    # Export data
    tp = pd.DataFrame(true_positive)
    tp.to_csv('truePositiveTest.csv')
    tn = pd.DataFrame(true_negative)
    tn.to_csv('trueNegativeTest.csv')
    fp = pd.DataFrame(false_positive)
    fp.to_csv('falsePositiveTest.csv')
    fn = pd.DataFrame(false_negative)
    fn.to_csv('falseNegativeTest.csv')
    mseExport = pd.DataFrame(mse_values)
    mseExport.to_csv('mseExport_test.csv')
    pred_inv = pd.DataFrame(predicted_invalid)
    pred_inv.to_csv("predicted_invalid_test.csv") 

    data_export_vals = {
        'Date': timestamps,
        'T1': testing_data,
        'T1_e': mse_values,
        'T1_c': flags
    }

    data_export = pd.DataFrame(data_export_vals)
    data_export.to_csv("T1_report_test.csv", index=False)

    # Plot graph: Current Data vs MSE
    plt.figure()
    plt.subplot(111)
    plt.plot(testing_data, label='Current Data')
    plt.plot(mse_values, label='mse')	
    plt.legend()
    plt.show()

    return


def train_fit():
    """Train and predict time series data"""

    # load data
    filename = "offset1_T1n.csv"
    raw_seq = array(get_train_data(filename))
    # current_data = array(get_test_data(filename)) 

    # BEGINNING OF CREATING MODEL - CAN START COMMENTING HERE
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
 
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    # create model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
    
    # fit model
    model.fit(X,y,epochs=epochs,verbose=1)

    model.save('initial_model.h5') 

    # END OF CREATING MODEL - CAN END COMMENTING HERE

    # print("Prediction: ", yhat)

    data = validate(filename, raw_seq)
    test(filename, data)
    
    # true_positive = []
    # true_negative = []
    # false_positive = []
    # false_negative = []

    # mse_values = []
    

    # #next step: continuously update the model to keep changing as it learns 
    # i = 0 
    # while (i + update_frequency) < current_data.size -1:
    #     valid_data = [] 
    #     for j in range(update_frequency):

    #         # shape data
    #         data = np.append(raw_seq, current_data[i + j])
    #         X_update, y_update = split_sequence(data, n_steps)
    #         X_update = X_update.reshape((X_update.shape[0], X_update.shape[1], n_features))

    #         # predict next time stamp
    #         yhat_update = predict_next_timestamp(model, raw_seq)

    #         mse = calculate_mean_squared(yhat_update, current_data[i + j])
    #         mse_values.append(mse)
            
    #         valid_data, true_positive, true_negative, false_positive,\
    #             false_negative, output_flag = check_abnormal_data(yhat, 
    #                                                  flag_dictionary, 
    #                                                  current_data[i + j], 
    #                                                  timestamps[i + j], 
    #                                                  true_positive, 
    #                                                  true_negative, 
    #                                                  false_positive, 
    #                                                  false_negative, 
    #                                                  valid_data)

    #         # only update raw_seq is the data was found to be valid
    #         if output_flag == True:
    #             raw_seq = data

    #     # if there is enough data to update the model  
    #     if len(valid_data) > 10:
    #         X, y = split_sequence(raw_seq, n_steps)
    #         model.fit(X,y,epochs=5,verbose=1)
    #     i = i + update_frequency

    # # Export data
    # tp = pd.DataFrame(true_positive)
    # tp.to_csv('truePositive.csv')
    # tn = pd.DataFrame(true_negative)
    # tn.to_csv('trueNegative.csv')
    # fp = pd.DataFrame(false_positive)
    # fp.to_csv('falsePositive.csv')
    # fn = pd.DataFrame(false_negative)
    # fn.to_csv('falseNegative.csv')
    # mseExport = pd.DataFrame(mse_values)
    # mseExport.to_csv('mseExport.csv') 

    # # Plot graph: Current Data vs MSE
    # plt.figure()
    # plt.subplot(111)
    # plt.plot(current_data, label='Current Data')
    # plt.plot(mse_values, label='mse')	
    # plt.legend()
    # plt.show()
                        
                
if __name__ == '__main__':
    print ("hello")
    train_fit()



