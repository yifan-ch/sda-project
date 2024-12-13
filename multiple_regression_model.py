import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import data_model

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


def calculate_correlation_matrix_with_pandas(X, z_scores):
    # Convert X back to DataFrame for easier handling if needed
    X_df = pd.DataFrame(X, columns=z_scores.columns)
    return X_df.corr()


def calculate_vif_with_statsmodels(X):
    vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return vif


def perform_regression_sklearn(X, y):
    # Create a LinearRegression object
    model = LinearRegression()

    # Fit the model
    model.fit(X, y)

    # Get coefficients and intercept
    # print("Intercept:", model.intercept_)
    # print("Coefficients:", model.coef_)

    # Return the fitted model
    return model


def perform_regression_statsmodels(X, y):
    # Add a constant term to the model (intercept)
    X_with_intercept = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X_with_intercept).fit()

    # Print the summary of the regression
    # print(model.summary())

    return model


# randomly split data into two parts based on the fraction.
def split(df, frac=0.5):
    p1 = df.sample(frac=frac)  # frac
    p2 = df.drop(p1.index)  # 1-frac

    return p1, p2


def predict(y_pred, threshold):
    return (y_pred >= threshold).astype(int)


def calculate_metrics(y_true, y_pred_labels):
    """
    Calculate accuracy, false positives, false negatives, true positives, and true negatives.
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


def test_regression_sklearn(z_scores, frac_training=0.5, thres=0.5):
    # Split the data for status 0 (64 total samples)
    df_0 = data_model.status(z_scores, 0)
    df_0_training, df_0_test = split(
        df_0, frac_training
    )  # Divide evenly between training and testing

    # Split the data for status 1 (188 total samples)
    df_1 = data_model.status(z_scores, 1)
    df_1_test = df_1.sample(n=len(df_0_test))  # Select 32 samples for the test set
    df_1_training = df_1.drop(df_1_test.index)  # The rest go into the training set

    # training data
    df_training = pd.concat([df_0_training, df_1_training], axis=0)
    y_training = df_training["status"].values
    X_training = df_training.drop(["status"], axis=1).values

    # test data
    df_test = pd.concat([df_0_test, df_1_test], axis=0)
    y_test = df_test["status"].values
    X_test = df_test.drop(["status"], axis=1).values

    model = perform_regression_sklearn(X_training, y_training)

    # predict
    y_pred = predict(model.predict(X_test), thres)
    # mae = mean_absolute_error(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # rmse = root_mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    metrics = calculate_metrics(y_test, y_pred)

    # return (mae, mse, rmse, r2), accuracy
    return metrics
