import pandas as pd
from pathlib import Path
import logging

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
        if set(types_map.keys() - {"brand"}).issubset(set(df.columns)):
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
