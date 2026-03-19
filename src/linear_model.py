from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from src.eval_metrics import evaluate_model
import pandas as pd
import numpy as np
from numpy.random import RandomState
from typing import Hashable


def linear_preprocessing(
    X_num_train_scaled: pd.DataFrame,
    X_num_test_scaled: pd.DataFrame,
    X_cat_train: pd.DataFrame,
    X_cat_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    Function to preprocess a DataFrame before being used for linear regression ML models.

    Applies "one hot encoding" to categorical features data and then combines with numerical features data.

    Args:
        X_num_train_scaled: A cleaned and scaled DataFrame of numerical features training data.
        X_num_test_scaled: A cleaned and scaled DataFrame of numerical features testing data.
        X_cat_train: A cleaned DataFrame of categorical features training data.
        X_cat_test: A cleaned DataFrame of categorical features testing data.

    Returns:
        A tuple of one-hot encoded training and testing DataFrames, and the OneHotEncoder instance used to transform the data (only fitted on the training data)

    Raises:
        TypeError if:
            - any input is not a pandas DataFrame.
            - X_num_train_scaled or X_num_test_scaled contains non-numeric values.
            - X_cat_train or X_cat_test contains numeric values.
        ValueError if:
            - any input DataFrame is empty.
            - any input contains invalid rows.
            - any input contains 'object' dtype.
            - the numerical data appears unscaled (heuristic check).
            - either train/test pair has mismatched columns.
            - either the training datasets or the testing datasets have mismatched indices.
    """
    input_map = {
        "X_num_train_scaled": (X_num_train_scaled, "numeric"),
        "X_num_test_scaled": (X_num_test_scaled, "numeric"),
        "X_cat_train": (X_cat_train, "categorical"),
        "X_cat_test": (X_cat_test, "categorical"),
    }

    for name, (data, dtype) in input_map.items():
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if data.empty:
            raise ValueError(f"{name} is an empty DataFrame")

        if dtype == "numeric":
            if not data.select_dtypes(exclude=["number"]).empty:
                raise TypeError(f"{name} must not contain non-numeric values")
            if data.isna().any().any():
                raise ValueError(f"{name} contains invalid values")

        if dtype == "categorical":
            if not data.select_dtypes(include=["number"]).empty:
                raise TypeError(f"{name} must not contain numeric values")
            if not data.select_dtypes(include=["object"]).empty:
                raise TypeError(
                    f"{name} contains generic 'object' dtypes: cast to an explicit type first"
                )
            if data.map(lambda x: str(x).strip() == "").any().any():
                raise ValueError(f"{name} contains empty strings or whitespace")

    if not X_num_train_scaled.columns.equals(X_num_test_scaled.columns):
        raise ValueError(
            "X_num_train_scaled and X_num_test_scaled columns do not match"
        )
    if not X_cat_train.columns.equals(X_cat_test.columns):
        raise ValueError("X_cat_train and X_cat_test columns do not match")

    if not X_num_train_scaled.index.equals(X_cat_train.index):
        raise ValueError("X_num_train_scaled and X_cat_train indices do not match")
    if not X_num_test_scaled.index.equals(X_cat_test.index):
        raise ValueError("X_num_test_scaled and X_cat_test indices do not match")

    cols_to_encode = [
        col for col in X_cat_train.columns if X_cat_train[col].nunique() > 1
    ]

    if not cols_to_encode:
        return X_num_train_scaled, X_num_test_scaled, None

    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    train_ohe = ohe.fit_transform(X_cat_train[cols_to_encode])
    test_ohe = ohe.transform(X_cat_test[cols_to_encode])

    ohe_cols = ohe.get_feature_names_out()
    X_cat_train_encoded = pd.DataFrame(
        train_ohe, columns=ohe_cols, index=X_cat_train.index
    )
    X_cat_test_encoded = pd.DataFrame(
        test_ohe, columns=ohe_cols, index=X_cat_test.index
    )

    X_train = pd.concat([X_num_train_scaled, X_cat_train_encoded], axis=1)
    X_test = pd.concat([X_num_test_scaled, X_cat_test_encoded], axis=1)

    return X_train, X_test, ohe


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
            - input lengths are mismatched
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train contain differing number of observations")
    if not X_train.select_dtypes(exclude=["number"]).empty:
        raise ValueError("X_train must only contain numeric values")
    if not pd.api.types.is_numeric_dtype(y_train):
        raise ValueError("y_train must only contain numeric values")
    if X_train.isna().any().any():
        raise ValueError("X_train contains missing values")
    if y_train.isna().any().any():
        raise ValueError("y_train contains missing values")

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# def evaluate_linear_model(
#     model: RegressorMixin,
#     X_test: pd.DataFrame,
#     y_test: pd.Series | np.ndarray,
# ) -> dict[str, float]:
#     """
#     Evaluate a trained scikit-learn regressor model on called out test data.

#     Args:
#         model - a trained scikit-learn regressor model.
#         X_test - the test features data.
#         y_test - the test target data.

#     Returns:
#         Evaluation metrics for the model (MSE, RMSE, MAE, R2).

#     Raises:
#         TypeError if:
#             - model is not a scikit-learn linear regressor model
#             - X_test is not a pandas DataFrame
#         ValueError if:
#             - X_test and y_test have differing number of observations
#             - y_test is not one dimensional
#     """
#     if not hasattr(model, "predict"):
#         raise TypeError("Model must be an sklearn regressor")
#     if not isinstance(X_test, pd.DataFrame):
#         raise TypeError("X_test must be a pandas DataFrame")
#     if not isinstance(y_test, (pd.Series, np.ndarray)):
#         raise TypeError("y_test must be a pandas Series or 1D numpy array")

#     if y_test.ndim != 1:
#         raise ValueError("y_test must be 1-dimensional")
#     if len(X_test) != len(y_test):
#         raise ValueError("X_test and y_test must have the same number of observations")

#     y_pred = model.predict(X_test)
#     y_pred = np.asarray(y_pred).ravel()
#     return evaluate_model(y_test, y_pred)
