import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred):
    """Plot the confusion matrix to visualize the classification results."""
    con_mat = confusion_matrix(y_true, y_pred)  # Calculate confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=con_mat)  # Prepare display
    disp.plot()  # Plot the confusion matrix
    plt.title('Confusion Matrix')  # Set the title
    plt.show()  # Show the plot
