import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, normalize=None, cmap=plt.cm.Blues):
    """Plot the confusion matrix to visualize the classification results.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        normalize (str, optional): If 'true', the confusion matrix will be normalized.
        cmap (colormap, optional): Colormap to use for the matrix (default is plt.cm.Blues).
    
    Raises:
        ValueError: If the length of y_true and y_pred do not match.
    """
    
    if len(y_true) != len(y_pred):
        raise ValueError("Length of true labels and predicted labels must match.")
    
    con_mat = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=con_mat)
    disp.plot(cmap=cmap)
    plt.title('Confusion Matrix')
    plt.show()
