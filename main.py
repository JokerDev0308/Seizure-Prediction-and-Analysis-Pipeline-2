import os
import pandas as pd
from tqdm import tqdm
import numpy as np

from data_loader import get_participant_info, load_data  # Assuming these functions exist
from data_filter import filter_data
from feature_extraction import extract_features, prepare_labels
from svm_training import train_svm
from utils import plot_confusion_matrix

# Set random seed for reproducibility
np.random.seed(100)

# Create a directory for data if it doesn't exist
os.makedirs('./database', exist_ok=True)

if __name__ == "__main__":
    # Load participant info
    part_info_dict = get_participant_info()  # Get information about participants
    selected_participants = list(part_info_dict.keys())  # Use all participants

    data_frames = []  # Initialize a list to hold DataFrames of loaded data

    # Show progress bar while loading data
    for file in tqdm(selected_participants, desc="Loading Data"):
        df = load_data(file, part_info_dict[file]['Channels'])  # Load data for each file

        if not df.empty:
            filtered_data = filter_data(df)  # Apply filtering to the data
            data_frames.append(filtered_data)  # Add filtered data to the list
        else:
            print(f"No data loaded for {file}.")  # Print a message if loading fails

    # Combine all loaded data into a single DataFrame
    if data_frames:
        combined_data = pd.concat(data_frames, ignore_index=True)
    else:
        raise ValueError("No data frames to concatenate.")

    # Extract features from the combined data
    features = []
    labels = []

    for file in selected_participants:
        extracted_features, label = extract_features(combined_data, part_info_dict[file]['label'])
        features.append(extracted_features)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    print(f'Number of features extracted: {features.shape[0]}')
    print(f'Number of labels: {len(labels)}')

    # Check for size match
    if features.shape[0] != len(labels):
        raise ValueError(f"Feature set and label set sizes do not match: {features.shape[0]} vs {len(labels)}")

    # Train the SVM model
    svm_model, accuracy, y_test, y_pred = train_svm(features, labels)

    # Print the accuracy and plot the confusion matrix
    print(f'Accuracy: {accuracy * 100:.2f}%')  # Print accuracy
    plot_confusion_matrix(y_test, y_pred)  # Plot confusion matrix


   
