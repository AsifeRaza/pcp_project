# Imports
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from load_data import load_data

# Loading the data
subjects = load_data()
# print(subjects[0]["X"].shape)
# print(subjects[0]["eye"].shape)
# print(subjects[0]["eye"].flatten().shape)
#
# print(type(subjects[0]["X"]))
# print(type(subjects[0]["eye"]))

# Make all subject data equal-sized

# Splitting data into eyes-open/eyes closed based on 'eyes' data
class EyeOpenCloseSplitter(BaseEstimator, TransformerMixin):
    """

    Splits a single-subject EEG signal into eye-open and eye-closed segments
    using a binary eyes-open/eyes-closed signal.

    The transformer converts a 2D EEG signal of shape (T, n_channels) to two
    separate signals of shape (window_size, n_channels).

    """

    def __init__(self):
        pass

    def fit(self, data, eyes):
        return self

    def transform(self, data, eyes):
        """
        Single-subject EEG splitter.

        Parameters
        ----------
        data : np.ndarray
            EEG signal of shape (T, n_channels)

        eyes : np.ndarray
            Binary mask of shape (1,T)
            1 = eyes open
            0 = eyes closed

        Returns
        -------
        X_open : np.ndarray
        X_close : np.ndarray
        """
        eyes = eyes.flatten()               #changing the shape (1,T) to T
        X_open = data[eyes == 1]
        X_close = data[eyes == 0]

        return X_open, X_close

eye_open_close_splitter = EyeOpenCloseSplitter()
eyes_open, eyes_closed = eye_open_close_splitter.transform(subjects[0]["X"], subjects[0]["eye"])

print("Data - eyes-open: ", eyes_open.shape)
print("Data - eyes-open: ", eyes_closed.shape, "\n")

# Class "Window" - single subject
class EEGWindower(BaseEstimator, TransformerMixin):
    """
    Splits continuous EEG signals into non-overlapping fixed-size windows.

    This transformer converts a 2D EEG signal of shape (T, n_channels)
    into a 3D tensor of shape (n_windows, window_size, n_channels)
    by partitioning the time axis into equal segments.

    """

    def __init__(self, window_size=500):
        self.window_size = window_size

    def fit(self):
         return self

    def transform(self, data):
        """
        Transform EEG signal into fixed-size windows.

        Parameters
        ----------
        data : numpy.ndarray of shape (T, n_channels)
            Continuous EEG signal.

        Returns
        -------
        numpy.ndarray of shape (n_windows, window_size, n_channels)
            Windowed EEG segments.
        """

        T, n_channels = data.shape
        n_windows = T // self.window_size
        data_trimmed = data[:n_windows * self.window_size]

        windows = data_trimmed.reshape(
            n_windows,
            self.window_size,
            n_channels
        )
        return windows

windowing_eyes_open = EEGWindower()
windowed_eyes_open = windowing_eyes_open.transform(eyes_open)
print("Eyes-open - (n_windows, window_size, n_channels): ",windowed_eyes_open.shape)

windowing_eyes_closed = EEGWindower()
windowed_eyes_closed = windowing_eyes_closed.transform(eyes_closed)
print("Eyes-closed - (n_windows, window_size, n_channels): ",windowed_eyes_closed.shape, "\n")

# covariance on eye-open/eye-close data
class FeaturesWithCovariance(BaseEstimator, TransformerMixin):
    """
    Extracts covariance-based features from EEG windows.

    Converts each EEG window of shape (n_windows, window_size, n_channels)
    into a flattened covariance feature vector.

    Output shape:
        (n_windows, n_channels * n_channels)
    """

    def fit(self, data):
        return self

    def transform(self, data):
        """
        Parameters
        ----------
        data : np.ndarray
            Shape (n_windows, window_size, n_channels)

        Returns
        -------
        np.ndarray
            Shape (n_windows, n_channels * n_channels)
        """

        n_windows = data.shape[0]
        n_channels = data.shape[2]

        features = np.zeros((n_windows, n_channels * n_channels))

        for i, window in enumerate(data):
            cov = np.cov(window.T)      #(n_channels, n_channels)
            features[i] = cov.flatten()

        return features

cov_eyes_open = FeaturesWithCovariance()
features_eyes_open = cov_eyes_open.transform(windowed_eyes_open)
print("(n_windows,covariance_vector): ",features_eyes_open.shape)

# all data together in shape X: (n_windows, features), y: (n_windows,labels) wrt to subjects