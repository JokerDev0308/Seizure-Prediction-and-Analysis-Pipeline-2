import numpy as np
import pandas as pd
from scipy import signal

def filter_data(data, l_freq=1, h_freq=40, fs=256, order=4):
    """Apply a bandpass FIR filter to the EEG data.
    
    Args:
        data (pd.DataFrame): Input EEG data as a pandas DataFrame.
        l_freq (float): Lower frequency bound for the filter (default is 1 Hz).
        h_freq (float): Upper frequency bound for the filter (default is 40 Hz).
        fs (int): Sampling frequency of the data (default is 256 Hz).
        order (int): Order of the filter (default is 4).
    
    Returns:
        pd.DataFrame: Filtered EEG data as a pandas DataFrame.
    
    Raises:
        ValueError: If input data is not a DataFrame or is empty.
    """
    
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Input data must be a non-empty pandas DataFrame.")
    
    # Create the bandpass filter
    b, a = signal.butter(order, [l_freq, h_freq], btype='band', fs=fs)
    
    # Apply the filter to each column in the DataFrame
    filtered_data = data.apply(lambda x: signal.filtfilt(b, a, x), axis=0)
    
    return filtered_data
