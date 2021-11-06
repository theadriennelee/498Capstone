import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

"""
Sample code

path = os.path.join("data.csv")
columns = ["T1", "T2", "T3"]
test = Distorter()
test.read_data(path, columns)
test.apply_distortion("gaussian", 3, 0.005, 0.001)
test.view_chart("regular")
test.view_chart("distorted")
"""

class Distorter(object):
    """
    Object for
    """
    def __init__(self):
        # General variables
        self.data = None
        self.distorted_data = None
        self.length = None
        self.columns = None
        self.split = 0.8
        self.train = None
        self.test = None

        # Distortion-specific variables
        self.gaussian_mean = 0
        self.gaussian_variance = 5


    def read_data(self, path, columns):
        """
        Read the CSV into
        Args:
            path: path of csv
            columns: columns of interest
        """
        self.data = pd.read_csv(path)
        self.distorted_data = self.data.copy(deep=True)
        self.length = len(self.data.index)
        self.columns = columns

        for col in self.columns:
            if col not in self.data.columns:
                raise ValueError("Column '%s' not found in data", col)

            # Generate the label column
            label_col = "{}_class".format(col)
            self.distorted_data[label_col] = pd.Series(np.zeros(self.length, dtype=bool))


    def apply_distortion(self, type, count, size, variance, columns=None):
        """
        Apply distortions to a column
        Args:
            type:
            count:
            size:
            variance:
        """
        if self.data is None:
            print("Read in data first")
            return
        if not columns:
            columns = self.columns
        if any([col not in self.columns for col in columns]):
            raise ValueError("The following columns were not found "
                             "in the data: %s",
                             ", ".join([col not in self.columns for col in columns]))

        for col in columns:
            mask = self.anomaly_mask(count, size, variance)
            self.distorted_data["{}_class".format(col)] = self.distorted_data["{}_class".format(col)] ^ mask
            if type == "gaussian":
                noise = self.gaussian(self.distorted_data[col], self.gaussian_mean, self.gaussian_variance, mask)
            self.distorted_data[col] = self.distorted_data[col] + noise


    def anomaly_mask(self, count, size, variance):
        """
        Define an anomaly mask

        Args:
            count: number of anomalies to generate
            size_mu: mean anomaly size
            size_sigma: anomaly size standard deviation

        Returns:
             Series:
        """
        mask = pd.Series(np.zeros(self.length), dtype=bool)
        for x in range(count):
            anomaly_size = abs(np.random.normal(size, variance) * self.length)
            anomaly_center = int(random.randrange(int(self.length * self.split + anomaly_size / 2), self.length - 1))
            mask_start = max(0, int(anomaly_center - anomaly_size / 2))
            mask_end = min(self.length, int(anomaly_center + anomaly_size / 2))
            mask[mask_start:mask_end] = True

        return mask


    def gaussian(self, data, mean, variance, noise_mask):
        """
        Apply gaussian noise
        Args:
            data: data to apply to gaussian to
            mean: gaussian mean
            variance: gaussian variance
            noise_mask: noise mask

        Returns:
            data with gaussian noise
        """
        # Create the noise
        noise = pd.Series(np.random.normal(mean, variance, self.length))

        return data + noise * noise_mask


    def view_chart(self, data):
        """
        View the graphed data
        Args:
            data: which data to view
        """
        ax = plt.gca()
        for col in self.columns:
            if data == "distorted":
                self.distorted_data.plot(kind='line', x='date', y=col, ax=ax)
            elif data == "regular":
                self.data.plot(kind='line', x='date', y=col, ax=ax)
            else:
                raise ValueError("Please choose between the values 'regular' or 'distorted'")
        plt.show()