import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.random import RandomState
from typing import Hashable

logger = logging.getLogger(__name__)


def load_data(data_dir: Path) -> pd.DataFrame:
    """
    Function to load CSV data from given folder path

    Args:
        data_dir - absolute folder path of source data

    Returns:
        Pandas dataframe with source data combined and cleaned

    Raises:
        ValueError if input data directory does not contain CSV files
    """

    # Mapping of expected columns and their types
    types_map = {
        "brand": "string",
        "model": "string",
        "year": "int64",
        "price": "int64",
        "transmission": "string",
        "mileage": "int64",
        "fuelType": "string",
        "tax": "int64",
        "mpg": "float64",
        "engineSize": "float64",
    }

    # Read in source data and combine into a single dataframe
    dfs = {}
    for csv_file in data_dir.glob("*.csv"):
        brand = csv_file.stem
        df = pd.read_csv(csv_file)
        # Rename "tax(£)" column to "tax"
        if "tax(£)" in df.columns:
            df.rename(columns={"tax(£)": "tax"}, inplace=True)
        # Check all required columns are present, otherwise skip and warn user
        if (set(types_map) - {"brand"}).issubset(df.columns):
            df["brand"] = brand
            dfs[brand] = df
        else:
            missing_cols = [
                col
                for col in set(types_map.keys() - {"brand"})
                if col not in df.columns
            ]
            logger.warning(
                "Skipping %s as missing columns %s", csv_file.name, missing_cols
            )
    try:
        combined_df = pd.concat(dfs.values(), ignore_index=True, sort=False)
    except ValueError:
        raise ValueError(f"No valid CSV files found at {data_dir}")

    # Drop any extra columns
    if len(combined_df.columns) > len(types_map):
        cols_to_drop = set(combined_df.columns) - set(types_map.keys())
        combined_df.drop(columns=cols_to_drop, axis=1, inplace=True)

    # Drop invalid rows and cast columns to mapped types
    invalid_rows_counters = 0
    for column, dtype in types_map.items():
        if dtype in ("int64", "float64"):
            combined_df[column] = pd.to_numeric(combined_df[column], errors="coerce")
            invalid_rows_counters += combined_df.isna().any(axis=1).sum()
            combined_df = combined_df.dropna(subset=column)
            combined_df[column] = combined_df[column].astype(dtype)
        else:
            combined_df[column] = combined_df[column].astype(dtype)

    if invalid_rows_counters > 0:
        logger.warning(
            f"Dropped {invalid_rows_counters} rows from combined dataset due to invalid value(s) in numeric columns"
        )

    return combined_df


def split_datasets(
    df: pd.DataFrame,
    target_col: Hashable,
    test_size: int | float = 0.2,
    random_seed: RandomState | int = 42,
) -> dict[str, dict]:
    """
    Function to split given dataset into test/train sets and then further breakdown into num/cat/target data.

    Args:
        df - cleansed pandas DataFrame containing full dataset (features and target)
        target_col - column containing target values (i.e. y values, all remaining columns used as features)
        (Optional) test_size - Proportion of data to use as test set, remaining data used for training set.
                    Can be given as a proportion (between 0.0 and 1.0) or absolute integer number of samples.
                    Default value of 0.2.
        (Optional) random_seed - Seed value to ensure repeatable split of data. Default value of 42.

    Returns:
        A nested dictionary splitting by test/train at the first level and then into num/cat/target.

    Raises:
        TypeError if input data is not a pandas DataFrame
        ValueError if:
            - input target column is not present in the DataFrame
            - input data does not contain any features
            - input data does not contain both categorical and numerical features
            - input data contains any invalid rows
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if len(df.columns) < 3:
        raise ValueError(
            "df must contain at least two feature columns and one target column"
        )
    if target_col not in df.columns:
        raise ValueError("Target column not in df")
    if df.isna().any().any():
        raise ValueError("df contains invalid rows")

    y = df[target_col].copy()
    X = df.drop(columns=target_col)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if not bool_cols.empty:
        for col in bool_cols:
            X[col] = X[col].astype("int16")
    X_num = X.select_dtypes(include=[np.number]).copy()
    X_cat = X.select_dtypes(exclude=[np.number]).copy()

    if X_num.empty or X_cat.empty:
        raise ValueError("df must contain both numerical and categorical features")

    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = (
        train_test_split(X_num, X_cat, y, test_size=test_size, random_state=random_seed)
    )

    for output in (X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test):
        output = output.reset_index(drop=True, inplace=True)

    return {
        "train": {
            "X_num": X_num_train,
            "X_cat": X_cat_train,
            "y": y_train,
        },
        "test": {
            "X_num": X_num_test,
            "X_cat": X_cat_test,
            "y": y_test,
        },
    }


def scale_num_data(
    X_num_train: pd.DataFrame, X_num_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Function to scale numerical training and testing data.
    Fits only on training data to prevent leakage.

    Args:
        X_train - cleansed pandas DataFrame of numerical features training data
        X_test - cleansed pandas DataFrame of numerical features testing data

    Returns:
        A tuple containing scaled numerical train and test DataFrames as well as the accompanying fitted Scaler.

    Raises:
        TypeError if either input is not a pandas DataFrame.
        TypeError if either input contains non-numeric features.
        ValueErrof if either input contains invalid rows.
        ValueError if there are differing columns in X_num_train and X_num_test.
    """
    if not all(isinstance(input, pd.DataFrame) for input in (X_num_train, X_num_test)):
        raise TypeError("Inputs must both be pandas DataFrames")
    if (
        not X_num_train.select_dtypes(exclude=[np.number]).empty
        or not X_num_test.select_dtypes(exclude=[np.number]).empty
    ):
        raise TypeError("Inputs must only contain numeric features")

    if X_num_train.isna().any().any() or X_num_test.isna().any().any():
        raise ValueError("Inputs must not contain invalid rows")
    if not X_num_train.columns.equals(X_num_test.columns):
        raise ValueError("Inputs contain differing columns or column order")

    cols_to_scale = [
        col for col in X_num_train.columns if X_num_train[col].nunique() > 2
    ]

    X_num_train_scaled = X_num_train.copy()
    X_num_test_scaled = X_num_test.copy()
    scaler = StandardScaler()
    X_num_train_scaled[cols_to_scale] = scaler.fit_transform(X_num_train[cols_to_scale])
    X_num_test_scaled[cols_to_scale] = scaler.transform(X_num_test[cols_to_scale])

    return X_num_train_scaled, X_num_test_scaled, scaler
