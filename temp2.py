import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import pyedflib
import pywt
from scipy import signal
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from urllib.request import urlretrieve
from tqdm import tqdm  # Import tqdm for progress bars

# Set random seed for reproducibility
np.random.seed(100)

def get_participant_info():
    """Retrieve participant information from the PhysioNet database.
    
    Returns:
        dict: A dictionary containing participant file names and their associated information.
    """
    # Get a list of all records in the CHB-MIT database
    records_list = wfdb.io.get_record_list('chbmit', records='all')
    # Extract unique participant codes from the record list
    part_codes = sorted(set(record.split('/')[0] for record in records_list))

    part_info_dict = {}  # Initialize dictionary to hold participant information

    # Loop through each participant code to retrieve summary info
    for part_code in part_codes:
        # Construct URL to the summary file
        url = f"https://physionet.org/physiobank/database/chbmit/{part_code}/{part_code}-summary.txt"
        filename = f"./{part_code}-summary.txt"  # Local filename for summary
        urlretrieve(url, filename)  # Download the summary file
        
        # Read the contents of the summary file
        with open(filename, encoding='UTF-8') as f:
            content = f.readlines()
        
        channels = []  # List to store channel names
        file_name = ""  # Current file name being processed
        file_info_dict = {}  # Dictionary to hold info for the current file

        # Parse the contents of the summary file
        for line in content:
            # Check for channel information
            if 'Channel' in line:
                channel = line.split(': ')[-1].strip()  # Extract channel name
                channels.append(channel)  # Add channel to list
            # Check for file name
            elif 'File Name' in line:
                if file_name:
                    # Save the previous file's info
                    part_info_dict[file_name] = file_info_dict
                # Extract and set the current file name
                file_name = re.findall(r'\w+\d+_\d+|\w+\d+\w+_\d+', line)[0]
                file_info_dict = {'Channels': list(set(channels)), 'Seizures Window': []}  # Initialize info dict
            # Check for seizure start/end times
            elif 'Seizure Start Time' in line or 'Seizure End Time' in line:
                file_info_dict['Seizures Window'].append(int(re.findall(r'\d+', line)[-1]))  # Add time to window list

        # Save the last file's info
        if file_name:
            part_info_dict[file_name] = file_info_dict

    return part_info_dict  # Return the compiled participant information

def load_data(file, selected_channels=[]):
    """Load EEG data from a specified file.
    
    Args:
        file (str): The name of the EDF file to load.
        selected_channels (list): List of channels to extract from the file.
        
    Returns:
        DataFrame: A DataFrame containing the loaded EEG data.
    """
    try:
        # Determine the folder based on the file name
        folder = file.split("_")[0]
        # Construct the URL to download the EDF file
        url = f"https://physionet.org/physiobank/database/chbmit/{folder}/{file}.edf"
        filename = f"./data/{file}.edf"  # Local filename for the EDF file
        urlretrieve(url, filename)  # Download the EDF file
        
        f = pyedflib.EdfReader(filename)  # Open the EDF file
        # If no specific channels are selected, use all available channels
        if not selected_channels:
            selected_channels = f.getSignalLabels()
        
        # Initialize a buffer to store the signals
        sigbufs = np.zeros((f.getNSamples()[0], len(selected_channels)))
        # Read the signals for each selected channel
        for i, channel in enumerate(selected_channels):
            sigbufs[:, i] = f.readSignal(f.getSignalLabels().index(channel))
        
        # Create a DataFrame with the loaded data
        df = pd.DataFrame(sigbufs, columns=selected_channels).astype('float32')
        # Add a time column to the DataFrame
        df['Time'] = np.linspace(0, len(df) / f.getSampleFrequencies()[0], len(df), endpoint=False)
        df.set_index('Time', inplace=True)  # Set time as the index
        return df  # Return the DataFrame
    except Exception as e:
        print(f"Error loading {file}: {e}")  # Print error if loading fails
        return pd.DataFrame()  # Return an empty DataFrame

def filter_data(data, l_freq=1, h_freq=40):
    """Apply a bandpass FIR filter to the EEG data.
    
    Args:
        data (DataFrame): The input EEG data.
        l_freq (float): The lower frequency cut-off for the filter.
        h_freq (float): The upper frequency cut-off for the filter.
        
    Returns:
        DataFrame: The filtered EEG data.
    """
    # Apply the bandpass filter to each channel of the data
    return data.apply(lambda x: signal.filtfilt(*signal.butter(4, [l_freq, h_freq], btype='band', fs=256), x), axis=0)

def wavelet_decompose(data):
    """Perform wavelet decomposition on the data.
    
    Args:
        data (array-like): Input data to be decomposed.
        
    Returns:
        list: List of wavelet coefficients.
    """
    coeffs = pywt.wavedec(data, 'db4', level=5)  # Perform wavelet decomposition
    return coeffs  # Return the coefficients

def extract_features(data):
    """Extract statistical features from the data.
    
    Args:
        data (DataFrame): Input EEG data.
        
    Returns:
        Series: A Series containing mean power and standard deviation features.
    """
    mean_power = data.abs().mean()  # Calculate mean power
    std_dev = data.std()  # Calculate standard deviation
    # Combine the features into a single Series
    return pd.concat([mean_power, std_dev], axis=0)

def prepare_labels(part_info_dict):
    """Prepare labels based on seizure windows for supervised learning.
    
    Args:
        part_info_dict (dict): Dictionary containing participant information.
        
    Returns:
        np.ndarray: Array of labels indicating seizure (1) and non-seizure (0).
    """
    labels = []  # Initialize a list for labels
    for file, info in part_info_dict.items():
        # Append seizure labels for the number of seizure windows
        labels.extend([1] * len(info['Seizures Window']))  # Seizure label
        # Append non-seizure labels
        labels.extend([0] * (len(info['Channels']) - len(info['Seizures Window'])))  # Non-seizure
    return np.array(labels)  # Convert to NumPy array and return

def train_svm(X, y):
    """Train an SVM model to classify seizure events.
    
    Args:
        X (DataFrame): Feature set for training.
        y (array-like): Labels corresponding to the feature set.
        
    Returns:
        tuple: A tuple containing the trained model, accuracy, and test predictions.
    """
    clf = SVC(kernel='linear', C=1)  # Initialize SVM classifier
    # Split the dataset into training and test sets
    X_rem, X_test, y_rem, y_test = train_test_split(X, y, test_size=0.25, random_state=100, stratify=y)

    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)  # Initialize cross-validation
    scores = []  # List to store scores for each fold

    # Perform cross-validation
    for train_index, val_index in k_fold.split(X_rem, y_rem):
        X_train, X_val = X_rem.iloc[train_index], X_rem.iloc[val_index]  # Split into train/validation sets
        y_train, y_val = y_rem.iloc[train_index], y_rem.iloc[val_index]
        clf.fit(X_train, y_train)  # Fit the model to the training data
        scores.append(clf.score(X_val, y_val))  # Store validation score

    y_pred = clf.predict(X_test)  # Predict on the test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    
    return clf, accuracy, y_test, y_pred  # Return the trained model, accuracy, and predictions

def plot_confusion_matrix(y_true, y_pred):
    """Plot the confusion matrix to visualize the classification results.
    
    Args:
        y_true (array-like): True labels of the test set.
        y_pred (array-like): Predicted labels from the model.
    """
    con_mat = confusion_matrix(y_true, y_pred)  # Calculate confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=con_mat)  # Prepare display
    disp.plot()  # Plot the confusion matrix
    plt.title('Confusion Matrix')  # Set the title
    plt.show()  # Show the plot

# Main execution
if __name__ == "__main__":
    # Create a directory for data if it doesn't exist
    os.makedirs('./data', exist_ok=True)

    # Load participant info and data
    part_info_dict = get_participant_info()  # Get information about participants
    data_frames = []  # Initialize a list to hold DataFrames of loaded data

    # Show progress bar while loading data
    for file in tqdm(part_info_dict.keys(), desc="Loading Data"):
        df = load_data(file, part_info_dict[file]['Channels'])  # Load data for each file
        if not df.empty:
            filtered_data = filter_data(df)  # Apply filtering to the data
            data_frames.append(filtered_data)  # Add filtered data to the list
        else:
            print(f"No data loaded for {file}.")  # Print a message if loading fails

    # Combine all loaded data into a single DataFrame
    combined_data = pd.concat(data_frames)
    # Extract features from the combined data
    features = extract_features(combined_data)

    # Prepare data for SVM
    X = pd.DataFrame(features)  # Use the extracted features as input
    y = prepare_labels(part_info_dict)  # Prepare labels for the input

    # Train the SVM model
    svm_model, accuracy, y_test, y_pred = train_svm(X, y)

    # Print the accuracy and plot the confusion matrix
    print(f'Accuracy: {accuracy * 100:.2f}%')  # Print accuracy
    plot_confusion_matrix(y_test, y_pred)  # Plot confusion matrix
