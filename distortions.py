import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


class Distorter(object):
    """
    Object for applying distortions to a dataset
    """
    def __init__(self, train_size=0.6, val_size=0.2):
        if train_size <= 0 or val_size < 0:
            ValueError("Train size must be non-zero, neither size can be negative")
        if train_size + val_size >= 1:
            ValueError("Train and validation size must not exceed 1")
        # General variables
        self.data = None
        self.distorted_data = None
        self.length = None
        self.columns = None

        # Set variables
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = 1 - self.train_size - self.val_size

        # Sets
        self.train = None
        self.val = None
        self.test = None

        # Anomaly defaults
        self.anomaly_count = 10
        self.anomaly_size = 0.005
        self.anomaly_variance = 0.001
        self.start = None
        self.end = None


    def read_data(self, path, columns):
        """
        Read the CSV into the object. Must specify columns of interest;
        the object will be able to apply distortions to and track the
        class of those columns

        Args:
            path: path of csv
            columns: relevant columns
        """
        raw_data = pd.read_csv(path)
        if any([col for col in columns if col not in raw_data.columns]):
            ValueError("Columns specified must match to columns in data")

        self.data = raw_data
        self.distorted_data = self.data.copy(deep=True)
        self.length = len(self.data.index)
        self.columns = columns
        self.start = self.train_size * self.length
        self.end = self.length

        for col in self.columns:
            # Generate the label column
            label_col = "{}_class".format(col)
            self.distorted_data[label_col] = pd.Series(np.zeros(self.length, dtype=bool))


    def distort(self, mean, variance, columns, overwrite=False, **kwargs):
        """
        Applies a distortion to the selected columns

        Args:
            columns: which columns to distort
            type: distortion type
            overwrite: whether to add or overwrite with noise
            **kwargs: custom args

        Returns:

        """
        if self.data is None:
            ValueError("Read in data first!")
        if isinstance(columns, str):
            columns = [columns]

        # If no columns are given, default to use all columns
        if not columns:
            columns = self.columns

        # The following variables are the valid keyword arguments
        # Anomaly count
        count = kwargs["count"] if "count" in kwargs else self.anomaly_count

        # Variables for anomaly size
        anomaly_size = kwargs["anomaly_size"] if "anomaly_size" in kwargs \
            else self.anomaly_size
        anomaly_variance = kwargs["anomaly_variance"] if "anomaly_variance" in kwargs \
            else self.anomaly_variance

        # Variables to set bounds for anomaly size; upper bound must be greater
        # Setting these will overwrite anomaly size and variance, and distribute
        # sizes in this range evenly
        upper_bound = kwargs["upper_bound"] if "upper_bound" in kwargs else None
        lower_bound = kwargs["lower_bound"] if "lower_bound" in kwargs else None

        # Variables to set bounds for anomaly positions
        start = kwargs["start"] if "start" in kwargs else self.start
        end = kwargs["end"] if "end" in kwargs else self.end

        for col in columns:
            # Create the anomaly mask
            mask = self.anomaly_mask(count, anomaly_size, anomaly_variance, start, end, upper_bound, lower_bound)

            # Mark the anomalies in the class column
            self.distorted_data["{}_class".format(col)] = self.distorted_data["{}_class".format(col)] ^ mask

            # Generate the anomaly
            noise = pd.Series(np.random.normal(mean, variance, self.length))
            if overwrite:
                self.distorted_data[col] = self.distorted_data[col] * ~mask + noise * mask
            else:
                self.distorted_data[col] = self.distorted_data[col] + noise * mask


    def anomaly_mask(self, count, size, variance, start, end,
                     upper_bound=None, lower_bound=None):
        """
        Create an anomaly mask for certain distortion types. Specify an
        average and variance for anomaly sizes. If upper and lower bound
        are specified, the anomaly sizes will be randomly distributed
        between those two values (size and variance will be ignored)

        Args:
            count: number of anomalies to generate
            size: mean anomaly size
            variance: anomaly size standard deviation
            start: where to start anomalies
            end: where to end anomalies
            upper_bound: upper bound on anomaly size
            lower_range: lower bound on anomaly size

        Returns:
             Series:
        """
        mask = pd.Series(np.zeros(self.length), dtype=bool)

        # Get the anomaly coverage

        # Create the anomaly mask
        for x in range(count):
            if upper_bound is not None and lower_bound is not None:
                anomaly_size = random.randrange(lower_bound, upper_bound)
            else:
                anomaly_size = abs(np.random.normal(size, variance) * self.length)
            anomaly_center = int(random.randrange(int(start + anomaly_size / 2), end - 1))
            mask_start = max(0, int(anomaly_center - anomaly_size / 2))
            mask_end = min(self.length, int(anomaly_center + anomaly_size / 2))
            mask[mask_start:mask_end] = True

        return mask


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


    def uniform_noise(self, variance):
        """
        Apply gaussian noise to the entire time series

        Args:
            variance: variance
        """
        for col in self.columns:
            noise = pd.Series(np.random.normal(0, variance, self.length))
            self.distorted_data[col] = self.distorted_data[col] + noise


    def gaussian_noise(self, variance, **kwargs):
        """
        Add gaussian noise anomalies

        Args:
            variance: variance
        """
        self.distort(0, variance, self.columns, **kwargs)


    def offset(self, value, **kwargs):
        """
        Add offset anomalies

        Args:
            value: offset
        """
        self.distort(value, 0, self.columns, **kwargs)


    def zero(self, **kwargs):
        """
        Add zero anomalies

        Args:
            value: offset
        """
        self.distort(0, 0, self.columns, overwrite=True, **kwargs)


    def pure_noise(self, mean, variance, **kwargs):
        """
        Add pure gaussian noise anomalies

        Args:
            value: offset
        """
        self.distort(mean, variance, self.columns, overwrite=True, **kwargs)


    def point(self, mean, variance, **kwargs):
        """
        Add point anomalies

        Args:
            value: offset
        """
        self.distort(mean, variance, self.columns, upper_bound=2, lower_bound=1, **kwargs)