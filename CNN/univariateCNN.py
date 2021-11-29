# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 12:20:53 2021

@author: yujin
"""

# univariate cnn example
from numpy import array
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
 
dataset = pd.read_csv("train.csv")
print(dataset.head())
#print("columns: ", dataset.columns)
print('shape of dataset: ',dataset.shape)    

# define input sequence
#raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
raw_seq = array(dataset)
# choose a number of time steps
test_set = pd.read_csv("test1.csv")
x_input = array(test_set)
n_steps = len(x_input)
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# summarize the data
#for i in range(len(X)):
#	print(X[i], y[i])
    
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
    
# dividing dataset into training set, cross validation set, and test set
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)
train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)

# partition the data into training and validation 
train_X, valid_X, train_label, valid_label = train_test_split(X, y, test_size=0.2, random_state=13)
print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

# fit model
model.fit(train_X, train_label, batch_size=64, epochs=10, verbose=1, validation_data=(valid_X, valid_label))
#model.fit(X, y, epochs=10, verbose=1)
# demonstrate prediction
#test = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print("Prediction: ", yhat)

test_eval = model.evaluate(array(test_X), array(test_Y), verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

keras.utils.plot_model(model, "my_model.png", show_shapes=True)
