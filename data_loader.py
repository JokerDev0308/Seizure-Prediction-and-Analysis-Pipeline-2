import re
import pandas as pd
import pyedflib
import wfdb
import numpy as np
from urllib.request import urlretrieve
import os

def get_participant_info():
    """Retrieve participant information from the PhysioNet database."""
    records_list = wfdb.io.get_record_list('chbmit', records='all')
    part_codes = sorted(set(record.split('/')[0] for record in records_list))

    part_info_dict = {}

    for part_code in part_codes:
        url = f"https://physionet.org/physiobank/database/chbmit/{part_code}/{part_code}-summary.txt"
        filename = f"./database/{part_code}/{part_code}-summary.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        urlretrieve(url, filename)
        
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
    """Load EEG data from a specified file, downloading it if not present locally."""
    folder = file.split("_")[0]
    local_filename = f"./database/{folder}/{file}.edf"
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    # Check if the file already exists
    if not os.path.isfile(local_filename):
        print(f"Downloading {file}...")
        url = f"https://physionet.org/physiobank/database/chbmit/{folder}/{file}.edf"
        urlretrieve(url, local_filename)
    else:
        print(f"Loading {file} from local storage.")

    try:
        f = pyedflib.EdfReader(local_filename)
        all_channels = f.getSignalLabels()
        
        # Print available channels for debugging
        #print("Available channels in the file:", all_channels)
        
        # Filter out any unexpected channel names
        all_channels = [channel for channel in all_channels if 'Channels in EDF Files:' not in channel]
        
        if not selected_channels:
            selected_channels = all_channels
        
        # Check if selected channels exist in the EDF file
        selected_channels = [channel for channel in selected_channels if channel in all_channels]
        
        if not selected_channels:
            print("No valid channels found.")
            return pd.DataFrame()
        
        sigbufs = np.zeros((f.getNSamples()[0], len(selected_channels)))
        for i, channel in enumerate(selected_channels):
            sigbufs[:, i] = f.readSignal(all_channels.index(channel))
        
        df = pd.DataFrame(sigbufs, columns=selected_channels).astype('float32')
        df['Time'] = np.linspace(0, len(df) / f.getSampleFrequencies()[0], len(df), endpoint=False)
        df.set_index('Time', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return pd.DataFrame()



