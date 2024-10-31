import os
import pandas as pd
from tqdm import tqdm

from data_filter import filter_data
from feature_extraction import extract_features, prepare_labels
from svm_training import train_svm
from utils import plot_confusion_matrix

# Set random seed for reproducibility
import numpy as np
np.random.seed(100)

# Create a directory for data if it doesn't exist
os.makedirs('./database', exist_ok=True)

if __name__ == "__main__":
    # Mock participant info for demonstration purposes
    part_info_dict = {
        'chb01_01': {'Channels': ['FP1-F7', 'F7-T7', 'T7-P7'], 'label': 1},
        'chb01_02': {'Channels': ['FP1-F7', 'F7-T7', 'T7-P7'], 'label': 0},
        'chb01_03': {'Channels': ['FP1-F7', 'F7-T7', 'T7-P7'], 'label': 1},
        'chb01_04': {'Channels': ['FP1-F7', 'F7-T7', 'T7-P7'], 'label': 0},
    }
    
    # Select only 4 participants to process
    selected_participants = list(part_info_dict.keys())[:7]
    
    data_frames = []  # Initialize a list to hold DataFrames of loaded data
    features_list = []  # Initialize a list to hold extracted features

    # Show progress bar while loading data
    for file in tqdm(selected_participants, desc="Loading Data"):
        # Simulated loading of data
        df = pd.DataFrame(np.random.randn(100, len(part_info_dict[file]['Channels'])), 
                          columns=part_info_dict[file]['Channels'])
        
        if not df.empty:
            filtered_data = filter_data(df)  # Apply filtering to the data
            data_frames.append(filtered_data)  # Add filtered data to the list
            
            # Extract features for this participant
            features = extract_features(filtered_data)
            features_list.append(features)  # Add the features to the list
        else:
            print(f"No data loaded for {file}.")  # Print a message if loading fails

    # Combine all features into a single DataFrame
    X = pd.DataFrame(features_list)

    # Prepare labels
    labels = prepare_labels(part_info_dict, selected_participants)  # Ensure this returns the correct labels

    # Check sizes
    print(f'Number of features extracted: {X.shape[0]}')  # Should match the number of participants
    print(f'Number of labels: {len(labels)}')

    # Ensure sizes match before proceeding
    if X.shape[0] != len(labels):
        raise ValueError(f"Feature set and label set sizes do not match: {X.shape[0]} vs {len(labels)}")

    # Train the SVM model
    svm_model, accuracy, y_test, y_pred = train_svm(X, labels)

    # Print the accuracy and plot the confusion matrix
    print(f'Accuracy: {accuracy * 100:.2f}%')  # Print accuracy
    plot_confusion_matrix(y_test, y_pred)  # Plot confusion matrix
