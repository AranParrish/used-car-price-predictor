from sklearn.linear_model import LinearRegression
from src.eval_metrics import evaluate_model
import pandas as pd


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
    model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """
    Evaluate a trained scikit-learn linear regression model on called out test data.

    Args:
        model - the trained scikit-learn linear regression model.
        X_test - the test features data.
        y_test - the test target data.

    Returns:
        Evaluation metrics for the model (MSE, RMSE, MAE, R2).

    Raises:
        TypeError if:
            - model is not a scikit-learn linear regression class
            - X_test is not a pandas DataFrame
            - y_test is not a pandas Series
        ValueError if:
            - the scikit-learn model has not been trained on any data
            - X_test and y_test have differing number of observations
    """
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred)
