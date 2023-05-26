
from model_training import L1Dist


import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_data):
    y_true = []
    y_pred_labels = []

    for batch in test_data:
        test_input, test_val, labels = batch
        y_true.extend(labels)

        y_pred = model.predict([test_input, test_val])
        y_pred_labels.extend([1 if pred > 0.5 else 0 for pred in y_pred])

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred_labels = np.array(y_pred_labels)

    # Classification report
    report = classification_report(y_true, y_pred_labels)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_labels)
    print("Confusion Matrix:")
    print(cm)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_labels)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()

    return y_true, y_pred_labels

