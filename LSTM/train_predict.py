import sys
import json
import build_model
import data_helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csv import reader
from sklearn.metrics import mean_squared_error
from keras.models import load_model

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
                        false_positive, false_negative, valid_data):
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

    """
    
    # Calculate thresholds
    threshold = 0.02
    upper_limit = next_timestamp_raw * (1 + threshold)
    lower_limit = next_timestamp_raw * (1 - threshold)

    print('Next_time_stamp_raw ' + str(next_timestamp_raw))
    print('Upper limit ' + str(upper_limit))
    print('Lower limit ' + str(lower_limit))

    # for i in range(len(current_data)):
    #     if float(current_data[i]) > upper_limit or float(current_data[i]) < lower_limit:
    #         if timestamps[i] in flag_dictionary:
    #             true_negative.append(timestamps[i])
    #         else:
    #             false_negative.append(timestamps[i])
    #             valid_data.append(current_data[i])
    #     else:
    #         if timestamps[i] in flag_dictionary:
    #             false_positive.append(timestamps[i])
    #         else:
    #             true_positive.append(timestamps[i])
    #             valid_data.append(current_data[i])
    
    # Compare data
    if float(current_data) > upper_limit or float(current_data) < lower_limit:
        if timestamp in flag_dictionary:
            true_negative.append(timestamp)
        else:
            false_negative.append(timestamp)
            valid_data.append(current_data)
    else:
        if timestamp in flag_dictionary:
            false_positive.append(timestamp)
        else:
            true_positive.append(timestamp)
            valid_data.append(current_data)
    
    return valid_data, true_positive, true_negative, false_positive, false_negative

def train_predict():
    """Train and predict time series data"""

    # Load command line arguments 
    # train_file = sys.argv[1]
    # parameter_file = sys.argv[2]
    # current_file = sys.argv[3]
    # current_parameter_file = sys.argv[4]
    train_file = "./data/distorted.csv"
    parameter_file = "./training_config.json"
    # current_file = "./data/test1.csv"
    current_parameter_file = "./training_config_update.json"

    # Load training parameters
    params = json.loads(open(parameter_file).read())
    current_params = json.loads(open(current_parameter_file).read())

    # Load time series dataset, and split it into train and validation
    x_train, y_train, x_valid, y_valid, x_valid_raw, y_valid_raw,\
        last_window_raw, last_window, data = data_helper.load_timeseries_with_filename(train_file, params)


    # Build RNN (LSTM) model
    lstm_layer = [1, params['window_size'], params['hidden_unit'], 1]
    model = build_model.rnn_lstm(lstm_layer, params)

    # Train RNN (LSTM) model with train set
    model.fit(
        x_train,
        y_train,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        validation_split=params['validation_split'])
    model.save('initial_model.h5')

    # Check the model against validation set
    predicted = build_model.predict_next_timestamp(model, x_valid)        

    # Predict the next timestamp
    next_timestamp_raw, rms = forecast(x_valid_raw, 
                                       y_valid_raw, 
                                       predicted, 
                                       model, 
                                       last_window, 
                                       last_window_raw)

    # Load testing time series dataset
    timestamps, current_data, flag_dictionary = data_helper.load_newtimeseries(train_file)

    # Update model every 50 timestamps
    update_frequency = 50
    
    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []
    
    i = 0
    
    # For every timestamp, update the next predicted value
    # With the updated predicted value, compare the value and the datapoint 
        # and place it in tp, tn, fp or fn
    # For every 50 timestamps, update the model with the new valid data
    while (i + update_frequency) < current_data.size - 1:
        valid_data = []
        for j in range(update_frequency):
            
            # add new data point into the array of data
            data = np.append(data, current_data[i + j])
            
            # Split data into train and validation set
            x_train_update, y_train_update, x_valid_update, y_valid_update,\
                x_valid_raw_update, y_valid_raw_update,\
                last_window_raw_update, last_window_update, data = data_helper.load_timeseries(data, 
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
            
            # Compare new data to predicted timestamp
            valid_data, true_positive, true_negative, false_positive,\
                false_negative = check_abnormal_data(next_timestamp_raw, 
                                                     flag_dictionary, 
                                                     current_data[i + j], 
                                                     timestamps[i + j], 
                                                     true_positive, 
                                                     true_negative, 
                                                     false_positive, 
                                                     false_negative, 
                                                     valid_data)

        # If there is enough data to update the model
        # Load time series dataset, and split it into train and validation
        if len(valid_data) > 10:
            x_train_update, y_train_update, x_valid_update, y_valid_update, x_valid_raw_update, y_valid_raw_update,\
                last_window_raw_update, last_window_update, data = data_helper.load_timeseries(valid_data, current_params)

            # Update model
            model = load_model('initial_model.h5')
        
            model.fit(
                x_train_update,
                y_train_update,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_split=params['validation_split'])
            model.save('initial_model.h5')
            predicted = build_model.predict_next_timestamp(model, x_valid_update)

            print("predicted " + str(predicted))
        
        i = i + update_frequency
    
    # Export data
    tp = pd.DataFrame(true_positive)
    tp.to_csv('truePositive_002_50_75.csv')
    tn = pd.DataFrame(true_negative)
    tn.to_csv('trueNegative_002_50_75.csv')
    fp = pd.DataFrame(false_positive)
    fp.to_csv('falsePositive_002_50_75.csv')
    fn = pd.DataFrame(false_negative)
    fn.to_csv('falseNegative_002_50_75.csv')
    



if __name__ == '__main__':
    # python3 train_predict.py ./data/sales.csv ./training_config.json
    # python train_predict.py ./data/test2.csv ./training_config.json ./data/test3.csv\
    #python train_predict.py ./data/test2.csv ./training_config.json ./data/test3.csv ./training_config_update.json 
    #type_threshold_epoch_updateFreq
    print("hello")
    train_predict()
