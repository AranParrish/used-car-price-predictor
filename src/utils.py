import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
from datetime import datetime


def linear_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess a dataframe before being used for linear regression ML models.

    Applies "one hot encoding" to categorical columns and returns a fully numeric dataframe.

    Args:
        df - a cleaned dataframe without any invalid row values.

    Returns:
        A fully numeric dataframe, suitable for use with ML models.

    Raises:
        TypeError if input data is not a pandas dataframe.
        ValueError if input data contains invalid rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas dataframe")

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
) -> tuple:
    """
    Function to split given dataset into test and training sets for linear regression ML models

    Args:
        df - Numeric pandas dataframe containing full dataset (features and target)
        target_col - String name of the target column (i.e. y values, all remaining columns used as features)
        test_size - Proportion of data to use as test set, remaining data used for training set.
                    Can be given as a proportion (between 0.0 and 1.0) or absolute integer number of samples.
                    Default value of 0.2.
        random_seed - Seed value to ensure repeatable split of data (i.e. to ensure same data splits for comparing ML models). Default value of 42.

    Returns:
        Four datasets as a tuple - two training sets (of features and target data) and two testing sets (of features and target data)

    Raises:
        TypeError if:
            - input data is not a pandas dataframe
            - input data contains non-numeric columns
            - input data does not contain any features
            - the random_seed is not an integer
        ValueError if:
            - input target column is not present in the dataframe
            - test_size is not a proportion between 0.0 and 1.0 or a valid number of samples
            - random_seed is outside the range [0, 2***32 - 1]
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input dataset must be a pandas dataframe")

    if not df.select_dtypes(include=["object", "string", "boolean"]).empty:
        raise TypeError("Input dataframe must not contain non-numeric columns")

    if len(df.columns) < 2:
        raise TypeError(
            "Dataframe must contain at least one feature column and one target column"
        )

    if isinstance(test_size, float) and not 0.0 <= test_size <= 1.0:
        raise ValueError(
            "test_size must be a float between 0.0 and 1.0 or an integer number of samples"
        )

    if isinstance(test_size, int) and not 0 <= test_size <= len(df):
        raise ValueError(
            "test_size must be a float between 0.0 and 1.0 or an integer number of samples"
        )

    if not isinstance(random_seed, (int, RandomState)):
        raise TypeError("random_seed must be an integer or numpy RandomState instance")

    if isinstance(random_seed, int) and not 0 <= random_seed <= 2**32 - 1:
        raise ValueError("random_seed must be an integer in the range [0, 2**32 - 1]")

    try:
        y = df[[target_col]]
        X = df.drop(columns=target_col, axis=1)
    except KeyError:
        raise ValueError("Target column not in input dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    return X_train, X_test, y_train, y_test


def embeddings_preprocessing(
    df: pd.DataFrame, target_col: str | int | float | datetime | tuple
) -> tuple:
    """
    Function to preprocess a dataframe being used for a PyTorch Neural Network ML model.

    Builds embeddings layers for categorical columns.

    Args:
        df - a cleaned dataframe without any invalid row values.
        target_col - name of column containing target values.

    Returns:
        A tuple containing a dataframe of numerical features, a dataframe of categorical features,
        a dataframe containing target values, and a dictionary of mappings for the categorical features.

    Raises:
        TypeError if input data is not a pandas dataframe.
        ValueError if target_col is not found in the input data.
        ValueError if input data contains invalid rows (e.g. NA values).

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas dataframe")

    if target_col not in df.columns:
        raise ValueError("Target column not found in input data")

    if df.isna().values.any():
        raise ValueError("Input data contains invalid rows")

    y = df[[target_col]].reset_index(drop=True)
    X = df.drop(columns=target_col)
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object", "string", "boolean"]).columns
    X_num = X[num_cols].reset_index(drop=True)
    X_cat = pd.DataFrame()
    mappings = {}

    for col in cat_cols:
        cat_series = X[col].astype("category")
        X_cat[col] = cat_series.cat.codes
        mappings[col] = dict(enumerate(cat_series.cat.categories))

    return X_num, X_cat, y, mappings


def split_and_tensorise(
    X_num: pd.DataFrame,
    X_cat: pd.DataFrame,
    y: pd.DataFrame,
    test_size: int | float = 0.2,
    random_seed: RandomState | int = 42,
) -> tuple:
    """
    Function to create training and testing datasets as torch tensors for use in a PyTorch model.

    Args:
        X_num - Pandas dataframe of numerical features data.
        X_cat - Pandas dataframe of categorical features data (as an embeddings layer).
        y - Pandas dataframe of target feature data.

    Returns:
        A tuple containing six datasets (training and testing sets for X_num, X_cat, and y).

    Raises:
        TypeError if:
            - any input is not a pandas dataframe
            - any input contains non-numeric values
        ValueError if the length of the inputs do not match
    """
    # if not all(isinstance(data, pd.DataFrame) for data in (X_num, X_cat, y)):
    #     raise TypeError("Input data must all be a pandas dataframe")

    # if (
    #     not X.select_dtypes(include=["object", "string"]).empty
    #     or not y.select_dtypes(include=["object", "string"]).empty
    # ):
    #     raise TypeError("Inputs must not contain non-numeric values")

    # if len(X) != len(y):
    #     raise ValueError("X and y lengths must match")

    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = (
        train_test_split(X_num, X_cat, y, test_size=test_size, random_state=random_seed)
    )

    return (
        torch.tensor(X_num_train.values, dtype=torch.float32),
        torch.tensor(X_num_test.values, dtype=torch.float32),
        torch.tensor(X_cat_train.values, dtype=torch.long),
        torch.tensor(X_cat_test.values, dtype=torch.long),
        torch.tensor(y_train.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32),
    )
