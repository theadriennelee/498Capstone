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
from keras.models import load_model 
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
 
#dataset = pd.read_csv("train.csv")
dataset = pd.read_csv("gaussian_t1.csv", skiprows=[i for i in range(15782,19734)]) #last 20% as test set - skip those rows
# get the dataset to only have relevant columns - for now just the timestep and t1 datapoint 
dataset = dataset.drop(['Unnamed: 0','date','Appliances','lights','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint','rv1','rv2','T1_class'],axis=1)
dataset.to_csv('parsed.csv')

# split the dataset into 60 training 20 validation and 20 test (last 20% for test has the distortions applied)
testset = pd.read_csv("gaussian_t1.csv", index_col=None, skiprows=[i for i in range(1,15782)]) #test set is only the last 20% 
testset = testset.drop(['Unnamed: 0','date','Appliances','lights','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint','rv1','rv2','T1_class'],axis=1)
testset.to_csv('testparsed.csv')

#print(dataset.head())
print("Columns: ", dataset.columns)
print('Shape of dataset: ',dataset.shape)    

# define input sequence
#raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
raw_seq = array(dataset)
# choose a number of time steps
#test_set = pd.read_csv("testparsed.csv")
#x_input = array(test_set)
x_input = array(testset)
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
model.fit(train_X, train_label, batch_size=64, epochs=1, verbose=1, validation_data=(valid_X, valid_label))
model.save('initial_model.h5') 
#model.fit(X, y, epochs=10, verbose=1)
# demonstrate prediction
#test = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print("Prediction: ", yhat)

test_eval = model.evaluate(array(test_X), array(test_Y), verbose=0)
print('Test loss:', test_eval[0])
#print('Test accuracy:', test_eval[1])

keras.utils.plot_model(model, "my_model.png", show_shapes=True)


#next step: continuously update the model to keep changing as it learns ? 
model = load_model('initial_model.h5') 
model.fit(train_X, train_label, batch_size=64, epochs=1, verbose=1, validation_data=(valid_X, valid_label)) #same conditions as initial fit 




