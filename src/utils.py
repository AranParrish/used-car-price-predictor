import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from numpy.random import RandomState


def linear_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess a dataframe before being used for linear regression ML models.

    Applies "one hot encoding" to categorical columns and returns a fully numeric dataframe.

    Args:
        df - a cleaned dataframe without any invalid row values.

    Returns:
        A fully numeric dataframe, suitable for use with ML models.

    Raises:
        TypeError if input is not a pandas dataframe.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas dataframe")

    processed_df = df.copy(deep=True)
    cat_cols = processed_df.select_dtypes(include=["object", "string"]).columns
    processed_df = pd.get_dummies(
        processed_df, columns=cat_cols, drop_first=True, dtype="int32"
    )
    return processed_df


def linear_train_test_datasets(
    df: pd.DataFrame,
    target_col: str,
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


def tensor_converter(X: pd.DataFrame, y: pd.DataFrame) -> tuple:
    """
    Function to convert training or testing datasets into torch tensors for use in a PyTorch model.

    Args:
        X - Numeric pandas dataframe of input features data.
        y - Numeric pandas dataframe of target features data.

    Returns:
        Two tensor datasets as a tuple.

    Raises:
        TypeError if:
            - either input is not a pandas dataframe
            - either input contains non-numeric values
        ValueError if the length of the inputs do not match
    """
    if not all(isinstance(data, pd.DataFrame) for data in (X, y)):
        raise TypeError("Inputs must both be a pandas dataframe")

    if (
        not X.select_dtypes(include=["object", "string"]).empty
        or not y.select_dtypes(include=["object", "string"]).empty
    ):
        raise TypeError("Inputs must not contain non-numeric values")

    if len(X) != len(y):
        raise ValueError("X and y lengths must match")

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    return X_tensor, y_tensor
