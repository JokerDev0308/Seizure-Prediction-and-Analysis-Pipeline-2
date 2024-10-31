import numpy as np
import pandas as pd

def extract_features(data):
    """Extract statistical features from the data."""
    mean_power = data.abs().mean()  # Calculate mean power
    std_dev = data.std()  # Calculate standard deviation
    return pd.concat([mean_power, std_dev], axis=0)

def prepare_labels(part_info_dict):
    """Prepare labels based on seizure windows for supervised learning."""
    labels = []  # Initialize a list for labels
    for file, info in part_info_dict.items():
        labels.extend([1] * len(info['Seizures Window']))  # Seizure label
        labels.extend([0] * (len(info['Channels']) - len(info['Seizures Window'])))  # Non-seizure
    return np.array(labels)  # Convert to NumPy array and return
