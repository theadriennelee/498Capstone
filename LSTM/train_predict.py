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

def train_predict():
	"""Train and predict time series data"""

	# Load command line arguments 
	train_file = sys.argv[1]
	parameter_file = sys.argv[2]
	current_file = sys.argv[3]

	threshold = 0.2

	# Load training parameters
	params = json.loads(open(parameter_file).read())

	# Load time series dataset, and split it into train and test
	x_train, y_train, x_test, y_test, x_test_raw, y_test_raw,\
		last_window_raw, last_window = data_helper.load_timeseries(train_file, params)

	print("x train " + str(x_train))
	print("y train " + str(y_train))


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
	predicted_raw = []
	# rms = []
	for i in range(len(x_test_raw)):
		predicted_val = (predicted[i] + 1) * x_test_raw[i][0]
		predicted_raw.append(predicted_val)
		# rms.append(mean_squared_error(y_test_raw[i], predicted_val, squared=False))
		# accuracy.append(abs(predicted_val - y_test_raw[i]) / y_test_raw[i] * 100)


	# final_rms = sum(rms) / len(rms)

	rms = mean_squared_error(y_test_raw, predicted_raw, squared=False)
	print("rms " + str(rms))

	# Plot graph: predicted VS actual
	plt.figure()
	plt.subplot(111)
	plt.plot(predicted_raw, label='Predicted')
	plt.plot(y_test_raw, label='Actual')	
	plt.legend()
	plt.show()
    
	# Plot graph: accuracy
	# plt.figure()
	# plt.subplot(111)
	# plt.plot(accuracy, label='Accuracy')
	# plt.legend()
	# plt.show()

	# Predict next time stamp 
	next_timestamp = build_model.predict_next_timestamp(model, last_window)
	next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw[0][0]
	print('The next time stamp forecasting is: {}'.format(next_timestamp_raw))

	current_data = data_helper.load_newtimeseries(current_file)
	upper_limit = next_timestamp_raw * (1 + rms)
	lower_limit = next_timestamp_raw * (1 - rms)
	print('Next_time_stamp_raw ' + str(next_timestamp_raw))
	print('Upper limit ' + str(upper_limit))
	print('Lower limit ' + str(lower_limit))

	for data in current_data:
		counter = 0
		if float(data) > upper_limit or float(data) < lower_limit:
			print("Abnormal data: " + str(data))

	x_train_update, y_train_update = data_helper.update_input_vales(current_data, params)

	model = load_model('initial_model.h5')
	model.fit(
		x_train_update,
		y_train_update,
		batch_size=params['batch_size'],
		epochs=params['epochs'],
		validation_split=params['validation_split'])
	predicted = build_model.predict_next_timestamp(model, x_test)        
	predicted_raw = []

	for i in range(len(x_test_raw)):
		predicted_val = (predicted[i] + 1) * x_test_raw[i][0]
		predicted_raw.append(predicted_val)
	# Predict next time stamp 
	next_timestamp = build_model.predict_next_timestamp(model, last_window)
	next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw[0][0]
	print('The next time stamp forecasting is: {}'.format(next_timestamp_raw))



if __name__ == '__main__':
	# python3 train_predict.py ./data/sales.csv ./training_config.json
	# python train_predict.py ./data/test2.csv ./training_config.json ./data/test3.csv
	train_predict()
