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
    """Load EEG data from a specified file."""
    try:
        folder = file.split("_")[0]
        url = f"https://physionet.org/physiobank/database/chbmit/{folder}/{file}.edf"
        filename = f"./database/{folder}/{file}.edf"
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
