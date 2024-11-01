from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_svm(X, y, kernel='linear', test_size=None, random_state=100):
    """Train an SVM model on the provided data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        kernel (str): The kernel type to be used in the SVM algorithm (default is 'linear').
        test_size (int, optional): The number of test samples; if None, defaults to at least one per class.
        random_state (int): Seed for reproducibility (default is 100).

    Returns:
        svm_model (SVC): The trained SVM model.
        accuracy (float): Accuracy of the model on the test set.
        y_test (np.ndarray): True labels for the test set.
        y_pred (np.ndarray): Predicted labels for the test set.

    Raises:
        ValueError: If there are not enough samples for stratification or if fewer than two classes are present.
    """
    
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError("Need at least two classes for SVM.")

    n_samples = len(y)
    n_classes = len(unique_classes)

    if n_samples < n_classes:
        raise ValueError("Not enough samples for stratification.")

    if test_size is None:
        test_size = max(1, n_classes)
        test_size = min(test_size, n_samples // 2)

    X_rem, X_test, y_rem, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_rem, y_rem)

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    return svm_model, accuracy, y_test, y_pred

