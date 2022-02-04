# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 12:20:53 2021

@author: yujin
"""

# univariate cnn example
from numpy import array
from tcn import TCN, compiled_tcn, tcn_full_summary
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model 
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import data_helper

params = {"epochs": 50, 
          "batch_size": 2,
          "window_size": 6,
          "train_test_split": 0.8,
          "validation_split": 0.1, 
          "hidden_unit": 10}

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        

    return array(X), array(y)

def get_train_data(filename):
    #dataset = pd.read_csv("train.csv")
    dataset = pd.read_csv(filename, index_col=None, squeeze=True, skiprows=[i for i in range(11840,19734)]) #last 20% as test set - skip those rows
# get the dataset to only have relevant columns - for now just the timestep and t1 datapoint 
    dataset = dataset.drop(['Unnamed: 0','date','Appliances','lights','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint','rv1','rv2','T1_class'],axis=1)
    dataset.to_csv('train_parsed.csv')
    return (dataset)

skipVal = 11840
endVal = 15787

def set_validation_data(index):
    if skipVal < index < endVal:
        return False
    elif index == 0:
        return False
    else:
        return True

def get_test_data(filename):
    # split the dataset into 60 training 20 validation and 20 test (last 20% for test has the distortions applied)
    testset = pd.read_csv(filename, index_col=None, squeeze=True, skiprows=lambda x: set_validation_data(x)) #test set is only the last 20% 
    testset = testset.drop(['Unnamed: 0','date','Appliances','lights','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint','rv1','rv2','T1_class'],axis=1)
    testset.to_csv('test_parsed.csv')
    return(testset)
        
def load_flag_dictionary(filename):
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
    testset = pd.read_csv(filename, index_col=None, squeeze=True, skiprows=lambda x: set_validation_data(x)) #test set is only the last 20% 
    testset = testset.drop(['date','Appliances','lights','T1','RH_1', 'T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint','rv1','rv2'],axis=1)

    # Seperate the data into individual arrays
    timestamps = testset.values[:, 0]
    flag = testset.values[:, 1]

    # Initialize dictionary
    flag_dictionary = {}
    for x in range(len(timestamps)):
        if flag[x] == True:
            flag_dictionary[timestamps[x]] = True

    return timestamps, flag_dictionary    
    
def get_normalized_data(filename, params):
    data = pd.read_csv(filename, index_col=None, squeeze=True, skiprows=[i for i in range(15782,19734)]) 
    adjusted_window = params['window_size']+1 
    raw = []
    for index in range(len(data) - adjusted_window):
        raw.append(data[index: index + adjusted_window])
        
    result = normalize_windows(raw)
    
    raw = np.array(raw)
    result = np.array(result)
    
    #split into training and validation
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
    normalized_data = []
    for window in window_data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return normalized_data
    
def predict_next_timestamp(model, history):
    """Predict the next time stamp given a sequence of history data"""

    prediction = model.predict(history)
    prediction = np.reshape(prediction, (prediction.size,))
    return prediction 

    
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

    #rms = mean_squared_error(y_valid_raw, predicted_raw, squared=False)
    #print("rms " + str(rms))
    
    next_timestamp = predict_next_timestamp(model, last_window)
    next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw[0][0]
    print('The next timestamp forecasting is: {}'.format(next_timestamp_raw))
    
    return [next_timestamp_raw]

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



#print(dataset.head())
   # print("Columns: ", dataset.columns)
    #print('Shape of dataset: ',dataset.shape)    

def create_model(x_train):
    
    #tcn_layer = TCN(input_shape=(n_steps, n_features))
    model = compiled_tcn(
        return_sequences=False,
        num_feat=x_train.shape[2],
        num_classes=0,
        nb_filters=24,
        kernel_size=8,
        dilations=[2 ** i for i in range(9)],
        nb_stacks=1,
        max_len=x_train.shape[1],
        use_skip_connections=False,
        use_weight_norm=True,
        regression=True,
        dropout_rate=0
    )
#    model = Sequential()
#
#    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
#    #model.add(TCN(nb_filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))
#    model.add(MaxPooling1D(pool_size=2))
#    model.add(Flatten())
#    model.add(Dense(50, activation='relu'))
#    model.add(Dense(1))
#    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

    return model 

def train_fit():
    # define input sequence
    raw_seq = array(get_train_data("gaussian_t1.csv"))
    # choose a number of time steps
    x_input = array(get_test_data("gaussian_t1.csv"))
    x_input_array = array(x_input)
    n_steps = 100
    #n_steps = len(x_input)
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    
    flag_dictionary = load_flag_dictionary("gaussian_t1.csv")    
    
    # summarize the data
    #for i in range(len(X)):
    #    print(X[i], y[i])
    
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    #print("X reshaped", X)
    # dividing dataset into training set, cross validation set, and test set
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    #train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.2,    random_state=42)

    train_X, valid_X, train_label, valid_label = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    #define model
    model = create_model(train_X) 
    # fit model
    model.fit(train_X, train_label, batch_size=5, epochs=5, verbose=1, validation_data=(valid_X, valid_label))
    model.save('initial_model.h5') 
    #model.fit(X, y, epochs=10, verbose=1)
    # demonstrate prediction
    #test = array([70, 80, 90])
    #print("x_input before reshape: ", x_input)
    #x_input = x_input.reshape((1, n_steps, n_features))
    #print("x_input: ", x_input)
    yhat = model.predict(valid_X, verbose=0)
    print("Prediction first: ", yhat)

    #test_eval = model.evaluate(array(test_X), array(test_Y), verbose=0)
    #print('Test loss:', test_eval[0])
    #print('Test accuracy:', test_eval[1])

    #keras.utils.plot_model(model, to_file='tcn_model.png', show_shapes=True, show_dtype=True, show_layer_names=True)
    print(model.summary())
    
    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []
    
    X_update, y_update = split_sequence(x_input, n_steps)
#    timestamps, current_data, flag_dictionary = data_helper.load_newtimeseries("gaussian_t1.csv")

    #next step: continuously update the model to keep changing as it learns 
    i = 0 
    update_frequency = 50 
    while (i + update_frequency) < x_input.size -1:
        valid_data = [] 
        for j in range(update_frequency):
            data = []
            data = np.append(raw_seq, x_input[0,:])
            
            #split data into training and validation again 
            train_X, valid_X, train_label, valid_label = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            
            X, y = split_sequence(raw_seq, n_steps)

            #add in same thing as above? and get rid of below use of load_timeseries which returns something different 
           # x_train_update, x_valid_update, y_train_update, y_valid_update = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            
            #x_train_update, y_train_update, x_valid_update, y_valid_update, x_valid_raw_update, y_valid_raw_update, last_window_raw_update, last_window_update, data = data_helper.load_timeseries(data, params)
            x_input = x_input.reshape((1, n_steps, n_features))
            predicted = model.predict(x_valid_update, verbose=0)
            print("Prediction inside for: ", predicted)
            #predictedArray = x_input_array
            
            next_timestamp_raw, rms = forecast(x_valid_raw_update, 
                                               y_valid_raw_update, 
                                               x_input_array, model, 
                                               last_window_update,
                                               last_window_raw_update)
            
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
            
            if len(valid_data) > 10:
                train_X, valid_X, train_label, valid_label = train_test_split(X, y,    test_size=0.2, random_state=13)
                # update model
                model = load_model('initial_model.h5') 
                model.fit(train_X, train_label, batch_size=5, epochs=5, verbose=1,   validation_data=(valid_X, valid_label))
                model.save('initial_model.h5')
                x_input = x_input.reshape((1, n_steps, n_features))
                predicted = model.predict(x_input, verbose=0)
                print("Prediction inside if: ", predicted)
                        
                
if __name__ == '__main__':
    print ("hello")
    train_fit()



