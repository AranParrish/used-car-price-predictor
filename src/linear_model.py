from sklearn.linear_model import LinearRegression
import pandas as pd


def linear_reg_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> LinearRegression:
    """
    Function to create a linear regression model trained on the input data.

    Args:
        X_train - features training set (i.e. model inputs)
        y_train - target training set (i.e. expected outputs mapped to the inputs)

    Returns:
        A trained instance of a linear regression model.

    Raises:
        TypeError if:
            - Either input is not a pandas dataframe
            - Either input contains non-numeric values
            - The length of the inputs do not match
    """
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.DataFrame):
        raise TypeError("Inputs must both be a pandas dataframe")

    if (
        not X_train.select_dtypes(include=["object", "string"]).empty
        or not y_train.select_dtypes(include=["object", "string"]).empty
    ):
        raise TypeError("Inputs must not contain non-numeric values.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train lengths must match")

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
