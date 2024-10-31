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
    """Retrieve participant information from the PhysioNet database."""
    records_list = wfdb.io.get_record_list('chbmit', records='all')
    part_codes = sorted(set(record.split('/')[0] for record in records_list))

    part_info_dict = {}

    for part_code in part_codes:
        url = f"https://physionet.org/physiobank/database/chbmit/{part_code}/{part_code}-summary.txt"
        filename = f"./data/{part_code}/{part_code}-summary.txt"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        urlretrieve(url, filename)
        print(f"{part_code} data downloaded successfully.")
        with open(filename, encoding='UTF-8') as f:
            content = f.readlines()
        
        channels = []
        file_name = ""
        file_info_dict = {}

        for line in content:
            if 'Channel' in line:
                channel = line.split(': ')[-1].strip()
                channels.append(channel)
            elif 'File Name' in line:
                if file_name:
                    part_info_dict[file_name] = file_info_dict
                file_name = re.findall(r'\w+\d+_\d+|\w+\d+\w+_\d+', line)[0]
                file_info_dict = {'Channels': list(set(channels)), 'Seizures Window': []}
            elif 'Seizure Start Time' in line or 'Seizure End Time' in line:
                file_info_dict['Seizures Window'].append(int(re.findall(r'\d+', line)[-1]))

        if file_name:
            part_info_dict[file_name] = file_info_dict

    return part_info_dict

def load_data(file, selected_channels=[]):
    """Load EEG data from a file."""
    try:
        folder = file.split("_")[0]
        url = f"https://physionet.org/physiobank/database/chbmit/{folder}/{file}.edf"
        filename = f"./data/{folder}/{file}.edf"
        urlretrieve(url, filename)
        
        f = pyedflib.EdfReader(filename)
        if not selected_channels:
            selected_channels = f.getSignalLabels()
        
        sigbufs = np.zeros((f.getNSamples()[0], len(selected_channels)))
        for i, channel in enumerate(selected_channels):
            sigbufs[:, i] = f.readSignal(f.getSignalLabels().index(channel))
        
        df = pd.DataFrame(sigbufs, columns=selected_channels).astype('float32')
        df['Time'] = np.linspace(0, len(df) / f.getSampleFrequencies()[0], len(df), endpoint=False)
        df.set_index('Time', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return pd.DataFrame()

def filter_data(data, l_freq=1, h_freq=40):
    """Apply FIR filter to the data."""
    return data.apply(lambda x: signal.filtfilt(*signal.butter(4, [l_freq, h_freq], btype='band', fs=256), x), axis=0)

def wavelet_decompose(data):
    """Perform wavelet decomposition."""
    coeffs = pywt.wavedec(data, 'db4', level=5)
    return coeffs

def extract_features(data):
    """Extract mean power and standard deviation features."""
    mean_power = data.abs().mean()
    std_dev = data.std()
    return pd.concat([mean_power, std_dev], axis=0)

def prepare_labels(part_info_dict):
    """Prepare labels based on seizure windows."""
    labels = []
    for file, info in part_info_dict.items():
        labels.extend([1] * len(info['Seizures Window']))  # Seizure label
        labels.extend([0] * (len(info['Channels']) - len(info['Seizures Window'])))  # Non-seizure
    return np.array(labels)

def train_svm(X, y):
    """Train SVM model and return trained model and accuracy."""
    clf = SVC(kernel='linear', C=1)
    X_rem, X_test, y_rem, y_test = train_test_split(X, y, test_size=0.25, random_state=100, stratify=y)

    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    scores = []

    for train_index, val_index in k_fold.split(X_rem, y_rem):
        X_train, X_val = X_rem.iloc[train_index], X_rem.iloc[val_index]
        y_train, y_val = y_rem.iloc[train_index], y_rem.iloc[val_index]
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_val, y_val))

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, accuracy, y_test, y_pred

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    con_mat = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=con_mat)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)

    # Load participant info and data
    part_info_dict = get_participant_info()
    data_frames = []

    # Show progress bar while loading data
    for file in tqdm(part_info_dict.keys(), desc="Loading Data"):
        df = load_data(file, part_info_dict[file]['Channels'])
        if not df.empty:
            filtered_data = filter_data(df)
            data_frames.append(filtered_data)
        else:
            print(f"No data loaded for {file}.")

    # Combine data and extract features
    combined_data = pd.concat(data_frames)
    features = extract_features(combined_data)

    # Prepare data for SVM
    X = pd.DataFrame(features)  # Adjust as necessary based on your feature extraction
    y = prepare_labels(part_info_dict)

    # Train the SVM model
    svm_model, accuracy, y_test, y_pred = train_svm(X, y)

    # Print accuracy and plot confusion matrix
    print(f'Accuracy: {accuracy * 100:.2f}%')
    plot_confusion_matrix(y_test, y_pred)
