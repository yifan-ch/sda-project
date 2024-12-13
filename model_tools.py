"""
Functions that are used by multiple models.
"""

import numpy as np


def split_df(df, frac=0.5):
    """
    Randomly split data into two parts based on the fraction.
    """

    p1 = df.sample(frac=frac)  # frac
    p2 = df.drop(p1.index)  # 1-frac

    return p1, p2


def predict_class(y_pred, threshold):
    """
    Convert predicted probabilities to class labels.
    """
    return (y_pred >= threshold).astype(int)


def calculate_metrics(y_true, y_pred_labels):
    """
    Calculate accuracy, precision, recall, f1, and rates.
    """

    # True Positives
    TP = np.sum((y_true == 1) & (y_pred_labels == 1))
    # False Positives
    FP = np.sum((y_true == 0) & (y_pred_labels == 1))
    # False Negatives
    FN = np.sum((y_true == 1) & (y_pred_labels == 0))
    # True Negatives
    TN = np.sum((y_true == 0) & (y_pred_labels == 0))

    # Accuracy calculation
    recall = TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (FN + TP)

    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP)
    f1 = 2 * ((precision * recall) / (precision + recall))

    return accuracy, precision, recall, f1, TPR, FPR, FNR, TNR
