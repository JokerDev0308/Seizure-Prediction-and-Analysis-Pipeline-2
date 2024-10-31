from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def train_svm(X, y):
    # Check for sufficient classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError("Need at least two classes for SVM.")

    n_samples = len(y)
    n_classes = len(unique_classes)
    
    # Ensure enough samples for stratification
    if n_samples < n_classes:
        raise ValueError("Not enough samples for stratification.")

    # Set a minimum test size to ensure all classes are represented
    test_size = max(1, n_classes)  # At least one sample per class
    test_size = min(test_size, n_samples // 2)  # Ensure it's not more than half the samples

    # Perform train-test split
    X_rem, X_test, y_rem, y_test = train_test_split(X, y, test_size=test_size, random_state=100, stratify=y)

    # Train SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_rem, y_rem)

    # Make predictions
    y_pred = svm_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return svm_model, accuracy, y_test, y_pred
