import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
from datetime import datetime


def linear_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess a Dataframe before being used for linear regression ML models.

    Applies "one hot encoding" to categorical columns and returns a fully numeric Dataframe.

    Args:
        df - a cleaned Dataframe without any invalid row values.

    Returns:
        A fully numeric Dataframe, suitable for use with ML models.

    Raises:
        TypeError if input data is not a pandas Dataframe.
        ValueError if input data contains invalid rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas Dataframe")

    if df.isna().values.any():
        raise ValueError("Input data contains invalid rows")

    processed_df = df.copy(deep=True)
    cat_cols = processed_df.select_dtypes(include=["object", "string"]).columns
    processed_df = pd.get_dummies(
        processed_df, columns=cat_cols, drop_first=True, dtype="int32"
    )
    return processed_df


def linear_train_test_datasets(
    df: pd.DataFrame,
    target_col: str | int | float | tuple | datetime,
    test_size: int | float = 0.2,
    random_seed: RandomState | int = 42,
) -> list[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Function to split given dataset into test and training sets for linear regression ML models

    Args:
        df - Numeric pandas Dataframe containing full dataset (features and target)
        target_col - String name of the target column (i.e. y values, all remaining columns used as features)
        (Optional) test_size - Proportion of data to use as test set, remaining data used for training set.
                    Can be given as a proportion (between 0.0 and 1.0) or absolute integer number of samples.
                    Default value of 0.2.
        (Optional) random_seed - Seed value to ensure repeatable split of data (i.e. to ensure same data splits for comparing ML models). Default value of 42.

    Returns:
        Four datasets as a list - two training sets (of features and target data) and two testing sets (of features and target data)

    Raises:
        TypeError if input data is not a pandas Dataframe
        ValueError if:
            - input target column is not present in the Dataframe
            - input data contains non-numeric columns
            - input data does not contain any features
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input dataset must be a pandas Dataframe")

    if not df.select_dtypes(include=["object", "string", "boolean"]).empty:
        raise ValueError("Input Dataframe must not contain non-numeric columns")

    if len(df.columns) < 2:
        raise ValueError(
            "Dataframe must contain at least one feature column and one target column"
        )

    if target_col not in df.columns:
        raise ValueError("Target column not in input dataset")

    y = df[target_col]
    X = df.drop(columns=target_col, axis=1)

    return train_test_split(X, y, test_size=test_size, random_state=random_seed)


def embeddings_preprocessing(
    df: pd.DataFrame, target_col: str | int | float | datetime | tuple
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """
    Function to preprocess a Dataframe being used for a PyTorch Neural Network ML model.

    Builds embeddings layers for categorical columns.

    Args:
        df - a cleaned Dataframe without any invalid row values.
        target_col - name of column containing target values.

    Returns:
        A tuple containing a Dataframe of numerical features, a Dataframe of categorical features,
        a Series containing target values, and a dictionary of mappings for the categorical features.

    Raises:
        TypeError if input data is not a pandas Dataframe.
        ValueError if target_col is not found in the input data.
        ValueError if input data contains invalid rows (e.g. NA values).

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas Dataframe")

    if target_col not in df.columns:
        raise ValueError("Target column not found in input data")

    if df.isna().values.any():
        raise ValueError("Input data contains invalid rows")

    y = df[target_col].reset_index(drop=True)
    X = df.drop(columns=target_col)
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object", "string", "boolean"]).columns
    X_num = X[num_cols].reset_index(drop=True)
    X_cat = pd.DataFrame()
    mappings = {}

    for col in cat_cols:
        cat_Series = X[col].astype("category")
        X_cat[col] = cat_Series.cat.codes
        mappings[col] = dict(enumerate(cat_Series.cat.categories))

    return X_num, X_cat, y, mappings


def split_and_tensorise(
    X_num: pd.DataFrame,
    X_cat: pd.DataFrame,
    y: pd.Series,
    test_size: int | float = 0.2,
    random_seed: RandomState | int = 42,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Function to create training and testing datasets as torch tensors for use in a PyTorch model.

    Args:
        X_num - Pandas Dataframe of numerical features data.
        X_cat - Pandas Dataframe of categorical features data (as an embeddings layer).
        y - Pandas Series of target data.
        (Optional) test_size - Proportion of data to use as test set, remaining data used for training set.
                    Can be given as a proportion (between 0.0 and 1.0) or absolute integer number of samples.
                    Default value of 0.2.
        (Optional) random_seed - Seed value to ensure repeatable split of data (i.e. to ensure same data splits for comparing ML models). Default value of 42.

    Returns:
        A tuple containing six datasets (training and testing sets for X_num, X_cat, and y).

    Raises:
        TypeError if:
            - either input features are not a pandas Dataframe
            - the target data is not a pandas Series
        ValueError if any input data contains non-numeric values
    """
    if not all(isinstance(data, pd.DataFrame) for data in (X_num, X_cat)):
        raise TypeError("Input features must be pandas Dataframe")

    if not isinstance(y, pd.Series):
        raise TypeError("Target must be a pandas Series")

    if not X_num.select_dtypes(exclude=["number"]).empty:
        raise ValueError("X_num must only contain numeric values")

    if not X_cat.select_dtypes(exclude=["integer"]).empty:
        raise ValueError("X_cat must only contain integer category codes")

    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = (
        train_test_split(X_num, X_cat, y, test_size=test_size, random_state=random_seed)
    )

    return (
        torch.tensor(X_num_train.values, dtype=torch.float32),
        torch.tensor(X_num_test.values, dtype=torch.float32),
        torch.tensor(X_cat_train.values, dtype=torch.long),
        torch.tensor(X_cat_test.values, dtype=torch.long),
        torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1),
    )
