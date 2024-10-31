# Seizure Prediction and Analysis Pipeline

## Summary

This Program is a machine learning application that analyzes electroencephalogram (EEG) data to predict seizures in patients with epilepsy. By leveraging data from the CHB-MIT database, the program employs signal processing techniques and machine learning algorithms to identify patterns associated with seizure events.

## Purpose

The primary purpose of this program is to assist researchers and clinicians in the early detection of seizures. Accurate prediction can significantly improve patient management and safety, enabling timely interventions. This program demonstrates how to integrate data preprocessing, feature extraction, and classification for seizure prediction using EEG data.

## Features

- **Data Retrieval**: Automatically downloads EEG data and corresponding metadata from the CHB-MIT database.
- **Data Preprocessing**: Applies a bandpass filter to the EEG signals to enhance relevant frequency components.
- **Feature Extraction**: Computes statistical features from the filtered EEG data, such as mean power and standard deviation.
- **Model Training**: Utilizes Support Vector Machine (SVM) for classification, with performance evaluation using cross-validation.
- **Visualization**: Displays a confusion matrix to visualize classification results.

## Requirements

Before running the program, ensure you have the following Python packages installed:

- `numpy`
- `pandas`
- `matplotlib`
- `wfdb`
- `pyedflib`
- `pywt`
- `scipy`
- `scikit-learn`
- `tqdm`

You can install these packages using pip:

```bash
pip install numpy pandas matplotlib wfdb pyedflib pywt scipy scikit-learn tqdm
```

## Usage

1. **Clone the Repository** (if applicable):

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Run the Program**:

   Save the program files in a directory (e.g., `eeg_seizure_prediction`). Execute the script using:

   ```bash
   python main.py
   ```

3. **Data Directory**:

   The program will automatically create a `./data` directory to store the downloaded EEG data files.

4. **Monitor Progress**:

   During execution, a progress bar will display the status of data loading. The accuracy of the SVM model will be printed to the console, and a confusion matrix will be displayed upon completion.

## Program Structure

### Key Modules

- **`main.py`**: Orchestrates the workflow by calling functions from other modules.
- **`data_loader.py`**: Contains functions for retrieving participant information and loading EEG data.
- **`data_filter.py`**: Contains functions for applying filtering techniques to the EEG data.
- **`feature_extraction.py`**: Contains functions for extracting statistical features and preparing labels for training.
- **`svm_training.py`**: Contains functions for training the SVM model and evaluating its performance.
- **`utils.py`**: Contains utility functions for plotting and visualizing results.

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The CHB-MIT database was developed as part of the Massachusetts Institute of Technology's research on epilepsy.
- Thanks to the developers of the libraries used in this project.
