import numpy as np
from scipy import signal

def filter_data(data, l_freq=1, h_freq=40):
    """Apply a bandpass FIR filter to the EEG data."""
    return data.apply(lambda x: signal.filtfilt(*signal.butter(4, [l_freq, h_freq], btype='band', fs=256), x), axis=0)
