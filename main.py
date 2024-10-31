import os
import pandas as pd
from tqdm import tqdm

from data_loader import get_participant_info, load_data
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
