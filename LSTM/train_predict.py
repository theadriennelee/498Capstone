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


def forecast(x_test_raw, y_test_raw, predicted, model, last_window, last_window_raw):
    predicted_raw = []
    
    for i in range(len(x_test_raw)):
        predicted_val = (predicted[i] + 1) * x_test_raw[i][0]
        predicted_raw.append(predicted_val)

    # Plot graph: predicted VS actual
    plt.figure()
    plt.subplot(111)
    plt.plot(predicted_raw, label='Predicted')
    plt.plot(y_test_raw, label='Actual')	
    plt.legend()
    plt.show()

    rms = mean_squared_error(y_test_raw, predicted_raw, squared=False)
    print("rms " + str(rms))
    
    next_timestamp = build_model.predict_next_timestamp(model, last_window)
    next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw[0][0]
    print('The next timestamp forecasting is: {}'.format(next_timestamp_raw))
    
    return [next_timestamp_raw, rms]

def check_abnormal_data(next_timestamp_raw, rms, current_data):
    threshold = 1
    upper_limit = next_timestamp_raw * (1 + threshold)
    lower_limit = next_timestamp_raw * (1 - threshold)
    # upper_limit = next_timestamp_raw + rms
    # lower_limit = next_timestamp_raw - rms
    print('Next_time_stamp_raw ' + str(next_timestamp_raw))
    print('Upper limit ' + str(upper_limit))
    print('Lower limit ' + str(lower_limit))

    valid_data = []
    for data in current_data:
        if float(data) > upper_limit or float(data) < lower_limit:
            print("Abnormal data: " + str(data))
        else:
            valid_data.append(data)
    
    return valid_data

def train_predict():
    """Train and predict time series data"""

    # Load command line arguments 
    # train_file = sys.argv[1]
    # parameter_file = sys.argv[2]
    # current_file = sys.argv[3]
    # current_parameter_file = sys.argv[4]
    train_file = "./data/distorted.csv"
    parameter_file = "./training_config.json"
    current_file = "./data/test1.csv"
    current_parameter_file = "./training_config.json"

    # Load training parameters
    params = json.loads(open(parameter_file).read())
    current_params = json.loads(open(current_parameter_file).read())

    # Load time series dataset, and split it into train and test
    x_train, y_train, x_test, y_test, x_test_raw, y_test_raw,\
        last_window_raw, last_window = data_helper.load_timeseries_with_filename(train_file, params)


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

    # Check the model against test set
    predicted = build_model.predict_next_timestamp(model, x_test)        

    next_timestamp_raw, rms = forecast(x_test_raw, y_test_raw, predicted, model, last_window, last_window_raw)

    current_data, flag_dictionary = data_helper.load_newtimeseries(train_file)

    update_frequency = 50
    for i in range(0, current_data.size - 1, update_frequency):
        updating_values = []
        for j in range(update_frequency):
            updating_values.append(current_data[i + j]) 
        print("updating values " + str(updating_values))

        valid_data = check_abnormal_data(next_timestamp_raw, rms, updating_values)

        # Load time series dataset, and split it into train and test
        x_train_update, y_train_update, x_test_update, y_test_update, x_test_raw_update, y_test_raw_update,\
            last_window_raw_update, last_window_update = data_helper.load_timeseries(valid_data, current_params)

        model = load_model('initial_model.h5')
        
        model.fit(
            x_train_update,
            y_train_update,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            validation_split=params['validation_split'])
        model.save('initial_model.h5')
        predicted_update = build_model.predict_next_timestamp(model, x_test_update)

        print("predicted " + str(predicted_update))

        next_timestamp_raw_updated, rms_updated = forecast(x_test_raw_update, y_test_raw_update, predicted_update, model, last_window_update, last_window_raw_update)        



if __name__ == '__main__':
    # python3 train_predict.py ./data/sales.csv ./training_config.json
    # python train_predict.py ./data/test2.csv ./training_config.json ./data/test3.csv\
    #python train_predict.py ./data/test2.csv ./training_config.json ./data/test3.csv ./training_config_update.json 
    print("hello")
    train_predict()
