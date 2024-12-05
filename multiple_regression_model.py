import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


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
