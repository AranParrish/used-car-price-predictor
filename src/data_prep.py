import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Function to split given dataset into test and training sets

    Args:
        df - Numeric pandas DataFrame containing full dataset (features and target)
        target_col - column containing target values (i.e. y values, all remaining columns used as features)
        (Optional) test_size - Proportion of data to use as test set, remaining data used for training set.
                    Can be given as a proportion (between 0.0 and 1.0) or absolute integer number of samples.
                    Default value of 0.2.
        (Optional) random_seed - Seed value to ensure repeatable split of data. Default value of 42.

    Returns:
        Four datasets as a list - two training sets (of features and target data) and two testing sets (of features and target data)

    Raises:
        TypeError if input data is not a pandas DataFrame
        ValueError if:
            - input target column is not present in the DataFrame
            - input data does not contain any features
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input dataset must be a pandas DataFrame")

    if len(df.columns) < 2:
        raise ValueError(
            "DataFrame must contain at least one feature column and one target column"
        )
    if target_col not in df.columns:
        raise ValueError("Target column not in input dataset")

    y = df[target_col]
    X = df.drop(columns=target_col)

    return train_test_split(X, y, test_size=test_size, random_state=random_seed)
