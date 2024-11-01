import numpy as np

def extract_features(data):
    """Extract features from the filtered EEG data.
    
    Args:
        data (pd.DataFrame): Filtered EEG data.
    
    Returns:
        np.ndarray: Extracted features for each participant.
    """
    # Placeholder: Implement actual feature extraction logic
    features = np.mean(data, axis=0).values  # Example: mean of each channel
    return features

def prepare_labels(part_info_dict, selected_participants):
    """Prepare labels based on participant information.
    
    Args:
        part_info_dict (dict): Dictionary containing participant info.
        selected_participants (list): List of participants.
    
    Returns:
        list: List of labels corresponding to each participant.
    """
    # labels = []
    # for participant in selected_participants:
    #     # Placeholder: Implement logic to derive labels from part_info_dict
    #     labels.append(part_info_dict[participant]['label'])  # Example placeholder
    # return labels
    labels = []
    for participant in selected_participants:
        # Check if 'label' exists; otherwise, handle appropriately
        if 'label' in part_info_dict[participant]:
            labels.append(part_info_dict[participant]['label'])
        else:
            # Placeholder: Assign a default label or handle the missing label case
            labels.append(0)  # or some other default value

    return labels