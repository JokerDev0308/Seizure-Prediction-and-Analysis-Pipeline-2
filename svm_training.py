import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def train_svm(X, y):
    """Train an SVM model to classify seizure events."""
    clf = SVC(kernel='linear', C=1)  # Initialize SVM classifier
    X_rem, X_test, y_rem, y_test = train_test_split(X, y, test_size=0.25, random_state=100, stratify=y)

    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)  # Initialize cross-validation
    scores = []  # List to store scores for each fold

    for train_index, val_index in k_fold.split(X_rem, y_rem):
        X_train, X_val = X_rem.iloc[train_index], X_rem.iloc[val_index]  # Split into train/validation sets
        y_train, y_val = y_rem.iloc[train_index], y_rem.iloc[val_index]
        clf.fit(X_train, y_train)  # Fit the model to the training data
        scores.append(clf.score(X_val, y_val))  # Store validation score

    y_pred = clf.predict(X_test)  # Predict on the test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    
    return clf, accuracy, y_test, y_pred  # Return the trained model, accuracy, and predictions
