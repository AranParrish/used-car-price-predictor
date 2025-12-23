from sklearn.linear_model import LinearRegression
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
            - X_train is not a pandas Dataframe
            - y_train is not a pandas Series
        ValueError if:
            - either input contains non-numeric values
            - either input contains missing values
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas Dataframe")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")

    if not X_train.select_dtypes(exclude=["number"]).empty:
        raise ValueError("X_train must only contain numeric values")

    if not pd.api.types.is_numeric_dtype(y_train):
        raise ValueError("y_train must only contain numeric values")

    if X_train.isna().any().any():
        raise ValueError("X_train must not contain missing values")

    if y_train.isna().any().any():
        raise ValueError("y_train must not contain missing values")

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
