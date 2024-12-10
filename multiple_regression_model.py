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

    return p2, p1


def test_regression_sklearn(df_z_scores, frac_training=0.5, tresh=0.5):
    # split the data into two fractions
    df_0_training, df_0_test = split(data_model.status(df_z_scores, 0), frac_training)  # status 0
    df_1_training, df_1_test = split(data_model.status(df_z_scores, 1), frac_training)  # status 1

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
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # calc accuracy on a simple treshhold
    accuracy = np.sum([[y_pred >= tresh] == y_test]) / len(y_test)

    return mae, mse, rmse, r2, accuracy
