import numpy as np

def extract_features(data):
    """Extract mean power and standard deviation features from the data."""
    mean_power = data.abs().mean().values  # Mean power for each channel
    std_dev = data.std().values  # Standard deviation for each channel
    features = np.concatenate((mean_power, std_dev))
    return features

def prepare_labels(part_info_dict, selected_participants):
    """Prepare labels based on participant info."""
    labels = []
    for participant in selected_participants:
        # Check if label exists, otherwise assign a default
        if 'label' in part_info_dict[participant]:
            labels.append(part_info_dict[participant]['label'])
        else:
            print(f"No label found for {participant}. Assigning default label.")
            labels.append(0)  # Assign a default label if none found
    return labels
