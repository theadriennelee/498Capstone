# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 17:27:38 2022

@author: Adrienne Lee
"""

# Import libraries
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

test_index = 200
update_frequency = 5

testVal = 19434

# TODO: fix file name
filename = 'data.csv'

threshold = 0.2

skipVal = 11840
endVal = 19734

def set_testing_data(index):
    """
    Helper function to read validation data

    Args:
        index (int): Index of datapoint
    Returns:
        boolean: Returns false if datapoint is a validation point
    """
    if index >= test_index:
        return False
    elif index == 0:
        return False
    else:
        return True

def calculate_mean_squared(predicted_val, actual_val):
    """

    Calculates mean squared error

    Args:
        predicted_val (int): Predicted value
        actual_val (int): Actual value

    Returns:
        mse (int): Mean squared error

    """
    # mse = math.sqrt(abs(actual_val - predicted_val))
    mse = pow((actual_val - predicted_val), 2)
    return mse

def check_abnormal_data(prediction, flag_dictionary, current_data, timestamp, true_positive, true_negative, false_positive, false_negative, flags):
    upper_limit = prediction * (1 + threshold)
    lower_limit = prediction * (1 - threshold)

    print('Next_time_stamp_raw ' + str(next_timestamp_raw))
    print('Upper limit ' + str(upper_limit))
    print('Lower limit ' + str(lower_limit))
    
    # Compare data
    output_flag = None
    if float(current_data) > upper_limit or float(current_data) < lower_limit:
        flags.append(True)
        output_flag = True
        if timestamp in flag_dictionary:
            true_negative.append(timestamp)
        else:
            false_negative.append(timestamp)
    else:
        output_flag = False
        flags.append(False)
        if timestamp in flag_dictionary:
            false_positive.append(timestamp)
        else:
            true_positive.append(timestamp)
    
    return flags, true_positive, true_negative,\
        false_positive, false_negative, output_flag

def check_abnormal_test_data(prediction, current_data, 
                        timestamp, flags, valid_data):
    """

    Compare new data against the predicted timestamp.
    Places the timestamp of the data in true_positive, true_negative, 
        false_positive or false_negative.

    Args:
        next_timestamp_raw (int): Predicted next timestamp
        current_data (int): Data to be checked
        timestamp (int): Timestamp of the data to be checked
        predicted_invalid (arr[int]): list of predicted invalid data
        flags (arr[int]): list of T/F flags
        valid_data (arr[int]): list of valid data

    Returns:
        predicted_invalid (arr[int]): list of invalid data
        flags (arr[int]): list of T/F flags
        valid_data (arr[int]): List of valid data

    """
    
    # Calculate thresholds
    upper_limit = prediction * (1 + threshold)
    lower_limit = prediction * (1 - threshold)

    print('Next_time_stamp_raw ' + str(prediction))
    print('Upper limit ' + str(upper_limit))
    print('Lower limit ' + str(lower_limit))
    
    # Compare data
    output_flag = None
    if float(current_data) > upper_limit or float(current_data) < lower_limit:
        flags.append(True)
        output_flag = True
    else:
        output_flag = False
        flags.append(False)
        valid_data.append(current_data)
    
    return flags, valid_data, output_flag
    
    # compare data

def train_predict():
    # load data
    data = pd.read_csv(filename, usecols=['date', 'T1'], index_col=0, skipfooter=testVal)
    # load timestamps and flags
    series = pd.read_csv(filename, sep=',', header=0, index_col=0, squeeze=True, skiprows=[i for i in range(skipVal, endVal)], usecols=['date', 'T1_class'])
    # seperate the data into individual arrays
    timestamps = series.index
    flag = series.values
    
    # initialize dictionary
    flag_dictionary = {}
    for x in range(len(timestamps)):
        if flag[x] == True:
            flag_dictionary[timestamps[x]] = True
    
    # TODO: set up model
    
    # START OF VALIDATION - FOR CLEAN UP PURPOSES
    i = 0
    ind = 0
    
    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []
    
    mse_values_validation = []
    predicted_invalid_validation = []
    flags_validation = []
    valid_data_validation = []
    
    for validation_data in data.values:
        # add new data point to array of data
        # TODO: add test data to array
        
        # predict next timestamp
        # TODO: predict next timestamp
        
        # calculate mse
        # TODO: pass correct variable into mse (prediction)
        mse = calculate_mean_squared(prediction, validation_data)
        mse_values_validation.append(mse)
        
        # Compare new data to predicted timestamp
        # TODO: pass correct variable into function (prediction)
        flags, true_positive, true_negative, false_positive,\
            false_negative, output_flag = check_abnormal_data(prediction, 
                                                 flag_dictionary, 
                                                 validation_data, 
                                                 timestamps[j], 
                                                 true_positive, 
                                                 true_negative, 
                                                 false_positive, 
                                                 false_negative, 
                                                 flags_validation)
        
        # upadte the model with new datapoints
        if i == update_frequency:
            # TODO: Add code to update model

            results = mod.fit()
            i = 0
        ind = ind + 1
        
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
    
    # DON'T NEED THIS UNLESS WE DO CLUSTERING
    # pred_inv = pd.DataFrame(predicted_invalid)
    # pred_inv.to_csv("predicted_invalid.csv")
    
    data_export_vals = {
            'Date': timestamps,
            'T1': validation_data,
            'T1_e': mse_values,
            'T1_c': flags
        }
    data_export = pd.DataFrame(data_export_vals)
    data_export.to_csv("T1_report_final.csv", index=False)
    
    # Plot graph: predicted VS actual
    plt.figure()
    plt.subplot(111)
    plt.plot(current_data, label='Current Data')
    plt.plot(mse_values, label='mse')    
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.subplot(111)
    plt.plot(current_data, label='Current Data')
    plt.plot(next_time_stamp_raw_values, label='Prediction')    
    plt.legend()
    plt.show()
    
    # END OF VALIDATION - FOR CLEAN UP PURPOSES
    
    
    # START OF TESTING - FOR CLEAN UP PURPOSES
    series = pd.read_csv(filename, sep=',', header=0, index_col=0, usecols=['date', 'T1'], squeeze=True, skiprows=lambda x: set_testing_data(x))
    timestamps = series.index
    testing_data = series.values
    
    mse_values_testing = []
    predicted_invalid_testing = []
    flags_testing = []
    valid_data_testing = []
    
    i = 0
    j = 0
    for test_data in testing_data:
        # add new data point to array of data
        # TODO: add test data to array
        
        # predict next timestamp
        # TODO: predict next timestamp
        
        # calculate mse
        # TODO: pass correct variable into mse (prediction)
        mse = calculate_mean_squared(prediction, validation_data)
        mse_values_testing.append(mse)
        
        # Compare new data to predicted timestamp
        # TODO: pass correct variable into function (prediction)
        flags, valid_data, output_flag = check_abnormal_test_data(prediction, test_data, timestamps[j],
                                                 flags_testing,   
                                                 valid_data_testing)
        
        # this is used for clustering
        if output_flag == True:
            predicted_invalid_testing.append([timestam[s[j], j]])
            
        # upadte the model with new datapoints
        if i == update_frequency:
            # TODO: Add code to update model

            results = mod.fit()
            i = 0
        j = j + 1
        
    # Export data
    mseExport = pd.DataFrame(mse_values)
    mseExport.to_csv('mseExport_test.csv')
    pred_inv = pd.DataFrame(predicted_invalid)
    pred_inv.to_csv("predicted_invalid_test.csv")
    
    data_export_vals = {
            'Date': timestamps,
            'T1': current_data,
            'T1_e': mse_values_testing,
            'T1_c': flags_testing
        }
    data_export = pd.DataFrame(data_export_vals)
    data_export.to_csv("T1_report_final_test.csv", index=False)
    
    # Plot graph: predicted VS actual
    plt.figure()
    plt.subplot(111)
    plt.plot(current_data, label='Current Data')
    plt.plot(mse_values, label='mse')	
    plt.legend()
    plt.show()

if __name__ = '__main__':
    print("hello")
    train_predict()