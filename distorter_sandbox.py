import Distorter
import os
import numpy as np

path = os.path.join("data.csv")
columns = ["T1", "T2", "T3"]
distorter = Distorter.Distorter()
distorter.gaussian_variance = 1
distorter.read_data(path, columns)
distorter.apply_distortion("gaussian", 3, 0.005, 0.001)
distorter.distorted_data.to_csv('distorted.csv')

columns = distorter.columns
window_size = 10

all_x = []
all_y = []

for col in columns:
	data = distorter.distorted_data[col]
	x_vals = []
	y_vals = []
	for i in range(data.size):
		endpoint = i + window_size
		if endpoint >= data.size:
			break
		x_vals.append(list(data[i:endpoint]))
		y_vals.append(data[endpoint])
	all_x.append(x_vals)
	all_y.append(y_vals)

multiinput_x = np.array(all_x)
print(multiinput_x)