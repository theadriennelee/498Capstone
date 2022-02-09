import sys
import json
import math
import build_model
import data_helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csv import reader
from sklearn.metrics import mean_squared_error
from keras.models import load_model

threshold = 0.02
update_frequency = 50

def forecast(x_valid_raw, y_valid_raw, predicted, model, last_window, 
             last_window_raw):
    """

    Predicts the next timestamp

    Args:
        x_valid_raw (arr[int]): Raw x test values
        y_valid_raw (arr[int]): Raw y test values
        predicted (arr[int]): Predicted values during validation
        model (keras.model): LSTM model
        last_window (arr[int]): History of data normalized
        last_window_raw (arr[int]): Raw history of data

    Returns:
        next_timestamp_raw (int): Predicted next timestamp
        rms (int): RMS value

    """
    
    predicted_raw = []
    
    for i in range(len(x_valid_raw)):
        predicted_val = (predicted[i] + 1) * x_valid_raw[i][0]
        predicted_raw.append(predicted_val)

    # Plot graph: predicted VS actual
    # plt.figure()
    # plt.subplot(111)
    # plt.plot(predicted_raw, label='Predicted')
    # plt.plot(y_valid_raw, label='Actual')	
    # plt.legend()
    # plt.show()

    rms = mean_squared_error(y_valid_raw, predicted_raw, squared=False)
    print("rms " + str(rms))
    
    next_timestamp = build_model.predict_next_timestamp(model, last_window)
    next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw[0][0]
    print('The next timestamp forecasting is: {}'.format(next_timestamp_raw))
    
    return [next_timestamp_raw, rms]

def check_abnormal_data(next_timestamp_raw, flag_dictionary, current_data, 
                        timestamp, true_positive, true_negative, 
                        false_positive, false_negative, valid_data, 
                        predicted_invalid, flags):
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
        predicted_invalid (arr[int]): list of timestamps that were predicted to
            be invalid
        flags (arr[int]): list of T/F flags

    Returns:
        valid_data (arr[int]): List of valid data
        predicted_invalid (arr[int]): list of invalid data
        flags (arr[int]): list of T/F flags
        true_positive (arr[int]): List of true positive timestamps
        true_negative (arr[int]): List of true negative timestamps
        false_positive (arr[int]): List of false positive timestamps
        false_negative (arr[int]): List of flase negative timestamps

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
    
    return valid_data, predicted_invalid, flags, true_positive, true_negative,\
        false_positive, false_negative

def check_abnormal_test_data(next_timestamp_raw, current_data, 
                        timestamp, predicted_invalid, flags, valid_data):
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
    
    return predicted_invalid, flags, valid_data

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

def validate(train_file, data, params, current_params):   
    """

    Validate model using validation data

    Args:
        train_file (string): .csv file with validation dataset
        data (arr[int]): list of data used to create model
        params (arr[int]): list of parameters used to validate model
        current_params (arr[int]): list of parameters used to update the model

    Returns:
        data (arr[int]): updated list of data used to create and validate model

    """
    
    model = load_model('initial_model.h5')

    # Load testing time series dataset
    timestamps, current_data, flag_dictionary = data_helper.load_validation_timeseries(train_file)
    
    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []
    
    mse_values = []
    predicted_invalid = []
    flags = []
    valid_data = []
    
    next_time_stamp_raw_values = []
    
    j = 0
    
    for test_data in current_data:
            
        # add new data point into the array of data
        temp_data = np.append(data, test_data)
        
        # Split data into train and validation set
        x_train_update, y_train_update, x_valid_update, y_valid_update,\
            x_valid_raw_update, y_valid_raw_update,\
            last_window_raw_update, last_window_update, temp_data = data_helper.load_timeseries(temp_data, 
                                                                                           params)
        
        # Check model against validation set
        predicted = build_model.predict_next_timestamp(model, 
                                                       x_valid_update)
        
        # Predict the next timestamp
        next_timestamp_raw, rms = forecast(x_valid_raw_update, 
                                           y_valid_raw_update, 
                                           predicted, model, 
                                           last_window_update,
                                           last_window_raw_update)
        
        next_time_stamp_raw_values.append(next_timestamp_raw)
        
        mse = calculate_mean_squared(next_timestamp_raw, test_data)
        mse_values.append(mse)
        
        # Compare new data to predicted timestamp
        valid_data, predicted_invalid, flags, true_positive, true_negative, false_positive,\
            false_negative = check_abnormal_data(next_timestamp_raw, 
                                                 flag_dictionary, 
                                                 test_data, 
                                                 timestamps[j], 
                                                 true_positive, 
                                                 true_negative, 
                                                 false_positive, 
                                                 false_negative, 
                                                 valid_data,
                                                 predicted_invalid, flags)

        data = temp_data
               
        # If there is enough data to update the model
        # Load time series dataset, and split it into train and validation
        if len(valid_data) > update_frequency:
            x_train_update, y_train_update, x_valid_update, y_valid_update, x_valid_raw_update, y_valid_raw_update,\
                last_window_raw_update, last_window_update, data = data_helper.load_timeseries(valid_data, current_params)

            # # Update model
            # model = load_model('initial_model.h5')
        
            model.fit(
                x_train_update,
                y_train_update,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_split=params['validation_split'])
            model.save('initial_model_validation.h5')
            predicted = build_model.predict_next_timestamp(model, x_valid_update)

            print("predicted " + str(predicted))
            valid_data = []
        
        j = j + 1
    
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
            'T1': current_data,
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
    
    return data

def test(train_file, data, params, current_params):
    """

    Test model using testing data

    Args:
        train_file (string): .csv file with validation dataset
        data (arr[int]): list of data used to create model
        params (arr[int]): list of parameters used to test model
        current_params (arr[int]): list of parameters used to update the model

    """
    
    model = load_model('initial_model_validation.h5')
    

    # Load testing time series dataset
    timestamps, current_data = data_helper.load_testing_timeseries(train_file)
    
    mse_values = []
    predicted_invalid = []
    flags = []
    valid_data = []
    
    j = 0
    
    for test_data in current_data:
            
        # add new data point into the array of data
        temp_data = np.append(data, test_data)
        
        # Split data into train and validation set
        x_train_update, y_train_update, x_valid_update, y_valid_update,\
            x_valid_raw_update, y_valid_raw_update,\
            last_window_raw_update, last_window_update, temp_data = data_helper.load_timeseries(temp_data, 
                                                                                           params)
        
        # Check model against validation set
        predicted = build_model.predict_next_timestamp(model, 
                                                       x_valid_update)
        
        # Predict the next timestamp
        next_timestamp_raw, rms = forecast(x_valid_raw_update, 
                                           y_valid_raw_update, 
                                           predicted, model, 
                                           last_window_update,
                                           last_window_raw_update)
        
        mse = calculate_mean_squared(next_timestamp_raw, test_data)
        mse_values.append(mse)
        
        # Compare new data to predicted timestamp
        predicted_invalid, flags, valid_data = check_abnormal_test_data(next_timestamp_raw, test_data, 
                                                                                     timestamps[j], 
                                                                                     predicted_invalid, 
                                                                                     flags, 
                                                                                     valid_data)
        
        data = temp_data
                
        # If there is enough data to update the model
        # Load time series dataset, and split it into train and validation
        if len(valid_data) > update_frequency:
            x_train_update, y_train_update, x_valid_update, y_valid_update, x_valid_raw_update, y_valid_raw_update,\
                last_window_raw_update, last_window_update, data = data_helper.load_timeseries(valid_data, current_params)

            # Update model
            model = load_model('initial_model_validation.h5')
        
            model.fit(
                x_train_update,
                y_train_update,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_split=params['validation_split'])
            model.save('initial_model_test.h5')
            predicted = build_model.predict_next_timestamp(model, x_valid_update)

            print("predicted " + str(predicted))
            valid_data = []
        
        j = j + 1
    
    # Export data
    mseExport = pd.DataFrame(mse_values)
    mseExport.to_csv('mseExport_test.csv')
    pred_inv = pd.DataFrame(predicted_invalid)
    pred_inv.to_csv("predicted_invalid_test.csv")
    
    print('timestamps ' + str(len(timestamps)))
    print('current data ' + str(len(current_data)))
    print('mse ' + str(len(mse_values)))
    print('flag ' + str(len(flags)))
    
    data_export_vals = {
            'Date': timestamps,
            'T1': current_data,
            'T1_e': mse_values,
            'T1_c': flags
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
    
    return
    

def train_predict():
    """Train and predict time series data"""

    # Load command line arguments 
    # train_file = sys.argv[1]
    # parameter_file = sys.argv[2]
    # current_file = sys.argv[3]
    # current_parameter_file = sys.argv[4]
    train_file = "./data/threshold_T1.csv"
    parameter_file = "./training_config.json"
    current_parameter_file = "./training_config_update.json"

    # Load training parameters
    params = json.loads(open(parameter_file).read())
    current_params = json.loads(open(current_parameter_file).read())

    # Load time series dataset, and split it into train and validation
    x_train, y_train, x_valid, y_valid, x_valid_raw, y_valid_raw,\
        last_window_raw, last_window, data = data_helper.load_timeseries_with_filename(train_file, params)


    # # Build RNN (LSTM) model
    # lstm_layer = [1, params['window_size'], params['hidden_unit'], 1]
    # model = build_model.rnn_lstm(lstm_layer, params)

    # # Train RNN (LSTM) model with train set
    # model.fit(
    #     x_train,
    #     y_train,
    #     batch_size=params['batch_size'],
    #     epochs=params['epochs'],
    #     validation_split=params['validation_split'])
    # model.save('initial_model.h5')

    # # Check the model against validation set
    # predicted = build_model.predict_next_timestamp(model, x_valid)        

    # # Predict the next timestamp
    # next_timestamp_raw, rms = forecast(x_valid_raw, 
    #                                     y_valid_raw, 
    #                                     predicted,
    #                                     model, 
    #                                     last_window, 
    #                                     last_window_raw)
    
    data = validate(train_file, data, params, current_params)
    
    test(train_file, data, params, current_params)


if __name__ == '__main__':
    print("hello")
    train_predict()
