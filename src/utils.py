import pandas as pd
from typing import Hashable
from collections.abc import Mapping
import torch
from sklearn.model_selection import train_test_split
from numpy.random import RandomState


def linear_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess a DataFrame before being used for linear regression ML models.

    Applies "one hot encoding" to categorical columns and returns a fully numeric DataFrame.

    Args:
        df - a cleaned DataFrame without any invalid row values.

    Returns:
        A DataFrame with categorical columns one-hot encoded (excluding "boolean" type), suitable for use with sklearn ML models.

    Raises:
        TypeError if input data is not a pandas DataFrame.
        ValueError if input data contains invalid rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.isna().any().any():
        raise ValueError("Input data contains invalid rows")

    processed_df = df.copy(deep=True)
    cat_cols = processed_df.select_dtypes(include=["object", "string"]).columns
    processed_df = pd.get_dummies(
        processed_df, columns=cat_cols, drop_first=True, dtype="int32"
    )
    return processed_df


def linear_train_test_datasets(
    df: pd.DataFrame,
    target_col: Hashable,
    test_size: int | float = 0.2,
    random_seed: RandomState | int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Function to split given dataset into test and training sets for linear regression ML models

    Args:
        df - Numeric pandas DataFrame containing full dataset (features and target)
        target_col - column containing target values (i.e. y values, all remaining columns used as features)
        (Optional) test_size - Proportion of data to use as test set, remaining data used for training set.
                    Can be given as a proportion (between 0.0 and 1.0) or absolute integer number of samples.
                    Default value of 0.2.
        (Optional) random_seed - Seed value to ensure repeatable split of data (i.e. to ensure same data splits for comparing ML models). Default value of 42.

    Returns:
        Four datasets as a list - two training sets (of features and target data) and two testing sets (of features and target data)

    Raises:
        TypeError if input data is not a pandas DataFrame
        ValueError if:
            - input target column is not present in the DataFrame
            - input data contains non-numeric columns
            - input data does not contain any features
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input dataset must be a pandas DataFrame")

    if not df.select_dtypes(exclude=["number"]).empty:
        raise ValueError("Input DataFrame must not contain non-numeric columns")

    if len(df.columns) < 2:
        raise ValueError(
            "DataFrame must contain at least one feature column and one target column"
        )

    if target_col not in df.columns:
        raise ValueError("Target column not in input dataset")

    y = df[target_col]
    X = df.drop(columns=target_col)

    return train_test_split(X, y, test_size=test_size, random_state=random_seed)


def embeddings_preprocessing(
    df: pd.DataFrame, target_col: Hashable
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, Mapping[Hashable, Mapping]]:
    """
    Function to preprocess a DataFrame being used for a PyTorch Neural Network ML model.

    Builds embeddings layers for categorical columns.

    Args:
        df - a cleaned DataFrame without any invalid row values.
        target_col - column containing target values.

    Returns:
        A tuple containing a DataFrame of numerical features, a DataFrame of categorical features,
        a Series containing target values, and a mapping of the categorical features.

    Raises:
        TypeError if input data is not a pandas DataFrame.
        ValueError if target_col is not found in the input data.
        ValueError if input data contains invalid rows (e.g. NA values).

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    if target_col not in df.columns:
        raise ValueError("Target column not found in input data")

    if df.isna().values.any():
        raise ValueError("Input data contains invalid rows")

    y = df[target_col].reset_index(drop=True)
    X = df.drop(columns=target_col)
    num_cols = X.select_dtypes(include=["number"]).columns
    bool_cols = X.select_dtypes(include=["boolean"]).columns
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    X_num = (
        X[num_cols.union(bool_cols)]
        .astype({col: "int8" for col in bool_cols})
        .reset_index(drop=True)
    )
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
        X_num - Pandas DataFrame of numerical features data.
        X_cat - Pandas DataFrame of categorical features data (as an embeddings layer).
        y - Pandas Series of target data.
        (Optional) test_size - Proportion of data to use as test set, remaining data used for training set.
                    Can be given as a proportion (between 0.0 and 1.0) or absolute integer number of samples.
                    Default value of 0.2.
        (Optional) random_seed - Seed value to ensure repeatable split of data (i.e. to ensure same data splits for comparing ML models). Default value of 42.

    Returns:
        A tuple containing six datasets (training and testing sets for X_num, X_cat, and y).

    Raises:
        TypeError if:
            - either input features are not a pandas DataFrame
            - the target data is not a pandas Series
        ValueError if:
            - any input data contains non-numeric values
            - any input data contains missing values
    """
    if not all(isinstance(data, pd.DataFrame) for data in (X_num, X_cat)):
        raise TypeError("Input features must be pandas DataFrame")

    if not isinstance(y, pd.Series):
        raise TypeError("Target must be a pandas Series")

    if not X_num.select_dtypes(exclude=["number"]).empty:
        raise ValueError("X_num must only contain numeric values")

    if not X_cat.apply(pd.api.types.is_integer_dtype).all():
        raise ValueError("X_cat must only contain integer category codes")

    if any(data.isna().any().any() for data in (X_num, X_cat, y)):
        raise ValueError("Input data must not contain missing values")

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


def fastai_embedding_dims(
    mappings: Mapping[Hashable, Mapping],
    max_dim: int = 50,
) -> list[int]:
    """
    Function to generate embedding dimensions using the FastAI heuristic approach.

    Args:
        mappings - map of column names to associated mappings of categorical codes.
        (Optional) max_dim - user-defined limit on the embedding dimension. Default value of 50.

    Returns:
        List of embedding dimensions corresponding to the order of mappings.values().

    Raises:
        TypeError if:
            - column names map is not a Mapping type
            - any categorical codes map is not a Mapping type
            - max_dim is not an integer
        ValueError if max_dim is not a positive integer.
    """
    if not isinstance(mappings, Mapping):
        raise TypeError("mappings must be a Mapping type")

    if not all(isinstance(mapping, Mapping) for mapping in mappings.values()):
        raise TypeError("Each categorical codes map must be a Mapping type")

    if not isinstance(max_dim, int):
        raise TypeError("max_dim must be an integer")

    if max_dim < 1:
        raise ValueError("max_dim must be a positive integer")

    return [
        max(1, min(max_dim, int(len(categories) ** 0.5)))
        for categories in mappings.values()
    ]
