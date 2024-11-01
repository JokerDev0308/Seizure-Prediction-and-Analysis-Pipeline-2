# import os
# import pandas as pd
# import numpy as np
# from tqdm import tqdm

# from data_loader import get_participant_info, load_data
# from data_filter import filter_data
# from feature_extraction import extract_features, prepare_labels
# from svm_training import train_svm
# from utils import plot_confusion_matrix

# # Set random seed for reproducibility
# np.random.seed(100)

# # Create a directory for data if it doesn't exist
# os.makedirs('./database', exist_ok=True)

# def main():
#     """Main function to load data, extract features, and train an SVM model."""
    
#     # Load participant info
#     part_info_dict = get_participant_info()
    
#     # Select participants to process
#     selected_participants = list(part_info_dict.keys())[:32]
    
#     data_frames = []  # Initialize a list to hold DataFrames of loaded data
#     features_list = []  # Initialize a list to hold extracted features
#     selected_channels = ["P7-O1","P3-O1", "P4-O2" , "P8-O2"]
#     # Show progress bar while loading data
#     for file in tqdm(selected_participants, desc="Loading Data"):
#         #df = load_data(file, part_info_dict[file]['Channels'])  # Load data for each file
#         df = load_data(file, selected_channels)
#         if not df.empty:
#             filtered_data = filter_data(df)  # Apply filtering to the data
#             data_frames.append(filtered_data)  # Add filtered data to the list
            
#             # Extract features for this participant
#             features = extract_features(filtered_data)
#             features_list.append(features)  # Add the features to the list
#         else:
#             print(f"No data loaded for {file}.")  # Print a message if loading fails

#     # Combine all features into a single DataFrame
#     X = pd.DataFrame(features_list)

#     # Prepare labels
#     labels = prepare_labels(part_info_dict, selected_participants)

#     # Check for distinct labels
#     if len(set(labels)) < 2:
#         print("Not enough distinct labels found. Simulating labels...")
#         labels = [i % 2 for i in range(len(selected_participants))]  # Alternate labels (0, 1)

#     # Check sizes
#     print(f'Number of features extracted: {X.shape[0]}')  # Should match the number of participants
#     print(f'Number of labels: {len(labels)}')

#     # Ensure sizes match before proceeding
#     if X.shape[0] != len(labels):
#         raise ValueError(f"Feature set and label set sizes do not match: {X.shape[0]} vs {len(labels)}")

#     # Train the SVM model
#     svm_model, accuracy, y_test, y_pred = train_svm(X, labels)

#     # Print the accuracy and plot the confusion matrix
#     print(f'Accuracy: {accuracy * 100:.2f}%')  # Print accuracy
#     plot_confusion_matrix(y_test, y_pred)  # Plot confusion matrix

# if __name__ == "__main__":
#     main()

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from data_loader import get_participant_info, load_data
from data_filter import filter_data
from feature_extraction import extract_features, prepare_labels
from svm_training import train_svm
from utils import plot_confusion_matrix

# Set random seed for reproducibility
np.random.seed(100)

# Create a directory for data if it doesn't exist
os.makedirs('./database', exist_ok=True)

def main():
    """Main function to load data, extract features, and train an SVM model."""
    
    # Load participant info
    part_info_dict = get_participant_info()
    
    # Select participants to process
    selected_participants = list(part_info_dict.keys())[:10]
    
    features_list = []  # Initialize a list to hold extracted features
    selected_channels = ["P7-O1", "P3-O1", "P4-O2", "P8-O2"]
    
    # Show progress bar while loading data
    for file in tqdm(selected_participants, desc="Loading Data"):
        df = load_data(file, selected_channels)
        if not df.empty:
            filtered_data = filter_data(df)  # Apply filtering to the data
            
            # Extract features for this participant
            features = extract_features(filtered_data)
            features_list.append(features)  # Add the features to the list
        else:
            print(f"No data loaded for {file}.")  # Print a message if loading fails

    # Combine all features into a single DataFrame
    X = pd.DataFrame(features_list)

    # Prepare labels only for the loaded participants
    valid_participants = [selected_participants[i] for i in range(len(selected_participants)) if i < len(features_list)]
    labels = prepare_labels(part_info_dict, valid_participants)

    # Check for distinct labels
    if len(set(labels)) < 2:
        print("Not enough distinct labels found. Simulating labels...")
        labels = [i % 2 for i in range(len(valid_participants))]  # Alternate labels (0, 1)

    # Check sizes
    print(f'Number of features extracted: {X.shape[0]}')  # Should match the number of valid participants
    print(f'Number of labels: {len(labels)}')

    # Ensure sizes match before proceeding
    if X.shape[0] != len(labels):
        raise ValueError(f"Feature set and label set sizes do not match: {X.shape[0]} vs {len(labels)}")

    # Train the SVM model
    svm_model, accuracy, y_test, y_pred = train_svm(X, labels)

    # Print the accuracy and plot the confusion matrix
    print(f'Accuracy: {accuracy * 100:.2f}%')  # Print accuracy
    plot_confusion_matrix(y_test, y_pred)  # Plot confusion matrix

if __name__ == "__main__":
    main()
