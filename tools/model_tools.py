"""
Functions that are used by multiple models.
"""

from .data_tools import status
import numpy as np
import pandas as pd


def split_df(df, frac=0.5, rs=None):
    """
    Randomly split data into two parts based on the fraction.
    """

    p1 = df.sample(frac=frac, random_state=rs)  # frac
    p2 = df.drop(p1.index)  # 1-frac

    return p1, p2


def split_training_test(df, random_state, frac_training=0.5):
    # Split_df the data for status 0 (64 total samples)
    df_0 = status(df, 0)
    df_0_training, df_0_test = split_df(
        df_0, frac_training, random_state
    )  # Divide evenly between training and testing

    # Split_df the data for status 1 (188 total samples)
    df_1 = status(df, 1)
    df_1_test = df_1.sample(
        n=len(df_0_test), random_state=random_state
    )  # Select 32 samples for the test set
    df_1_training = df_1.drop(df_1_test.index)  # The rest go into the training set

    # Combine the training data
    df_training = pd.concat([df_0_training, df_1_training], axis=0)
    y_training = df_training["status"].values.reshape(-1, 1)
    X_training = df_training.drop(["status"], axis=1).values

    # Combine the test data
    df_test = pd.concat([df_0_test, df_1_test], axis=0)
    y_test = df_test["status"].values.reshape(-1, 1)
    X_test = df_test.drop(["status"], axis=1).values

    return X_training, y_training, X_test, y_test


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
