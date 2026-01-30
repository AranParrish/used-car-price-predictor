from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from src.eval_metrics import evaluate_model
import pandas as pd
import numpy as np


def linear_reg_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Function to create a linear regression model trained on the input data.

    Args:
        X_train - features training set (i.e. model inputs)
        y_train - target training set (i.e. expected outputs mapped to the inputs)

    Returns:
        A trained instance of a linear regression model.

    Raises:
        TypeError if:
            - X_train is not a pandas DataFrame
            - y_train is not a pandas Series
        ValueError if:
            - either input contains non-numeric values
            - either input contains missing values
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")

    if not X_train.select_dtypes(exclude=["number"]).empty:
        raise ValueError("X_train must only contain numeric values")
    if not pd.api.types.is_numeric_dtype(y_train):
        raise ValueError("y_train must only contain numeric values")
    if any(data.isna().any().any() for data in (X_train, y_train)):
        raise ValueError("Input data must not contain missing values")

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_linear_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Evaluate a trained scikit-learn regressor model on called out test data.

    Args:
        model - a trained scikit-learn regressor model.
        X_test - the test features data.
        y_test - the test target data.

    Returns:
        Evaluation metrics for the model (MSE, RMSE, MAE, R2).

    Raises:
        TypeError if:
            - model is not a scikit-learn linear regressor model
            - X_test is not a pandas DataFrame
        ValueError if:
            - X_test and y_test have differing number of observations
            - y_test is not one dimensional
    """
    if not hasattr(model, "predict"):
        raise TypeError("Model must be an sklearn regressor")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame")
    if not isinstance(y_test, (pd.Series, np.ndarray)):
        raise TypeError("y_test must be a pandas Series or 1D numpy array")

    if y_test.ndim != 1:
        raise ValueError("y_test must be 1-dimensional")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of observations")

    y_pred = model.predict(X_test)
    y_pred = np.asarray(y_pred).ravel()
    return evaluate_model(y_test, y_pred)
