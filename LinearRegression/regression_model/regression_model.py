import numpy as np


def add_intercept(X):
    """
    Add a column of ones to X for the intercept.
    """
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


def fit_linear_regression(X, y):
    """
    Fit linear regression using OLS closed-form solution.
    beta = (X^T X)^(-1) X^T y
    """
    X = add_intercept(X)

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    XtY = X.T @ y

    beta = XtX_inv @ XtY

    return beta


def predict(X, beta):
    X = add_intercept(X)
    return X @ beta


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)