import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_auc(y_true, y_prob, save_path):
    """
    Plots the ROC-AUC curve and saves it to a file.

    Parameters:
        y_true (ndarray): An array of true labels (0 for negative class, 1 for positive class)
        y_prob (ndarray): An array of predicted probabilities for the positive class
        save_path (str): A file path to save the plot

    Returns:
        None
    """
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # Save plot
    plt.savefig(save_path)
