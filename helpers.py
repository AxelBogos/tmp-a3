import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_auc(all_labels, all_probs, output_path):
    # Calculate the ROC curve and AUC for training data
    train_fpr, train_tpr, _ = roc_curve(all_labels['train_labels'], all_probs['train_probs'])
    train_auc = auc(train_fpr, train_tpr)

    # Calculate the ROC curve and AUC for validation data
    val_fpr, val_tpr, _ = roc_curve(all_labels['val_labels'], all_probs['val_probs'])
    val_auc = auc(val_fpr, val_tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(train_fpr, train_tpr, color='blue', lw=2, label='Train ROC curve (area = %0.2f)' % train_auc)
    plt.plot(val_fpr, val_tpr, color='green', lw=2, label='Validation ROC curve (area = %0.2f)' % val_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()
