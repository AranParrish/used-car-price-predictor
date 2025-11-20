import pandas as pd
from pathlib import Path


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

    # Read in source data and combine into a single dataframe
    dfs = {}
    for csv_file in data_dir.glob("*.csv"):
        brand = csv_file.stem
        df = pd.read_csv(csv_file)
        df["brand"] = brand
        dfs[brand] = df
    try:
        combined_df = pd.concat(dfs.values(), ignore_index=True, sort=False)
    except ValueError:
        raise ValueError(f"No CSV files found at {data_dir}")

    # Combine "tax" and "tax(£)" column values into a single column
    if "tax(£)" in combined_df.columns and "tax" in combined_df.columns:
        combined_df.fillna({"tax(£)": 0}, inplace=True)
        combined_df.fillna({"tax": 0}, inplace=True)
        combined_df["tax"] = combined_df["tax"] + combined_df["tax(£)"]
        combined_df.drop(labels="tax(£)", axis=1, inplace=True)
    elif "tax(£)" in combined_df.columns:
        combined_df.rename(columns={"tax(£)": "tax"}, inplace=True)

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
    # Drop any extra columns
    if len(combined_df.columns) > len(types_map):
        cols_to_drop = set(combined_df.columns) - set(types_map.keys())
        combined_df.drop(columns=cols_to_drop, axis=1, inplace=True)

    # Drop invalid rows and cast columns to mapped types
    for column, dtype in types_map.items():
        if dtype in ("int64", "float64"):
            combined_df[column] = pd.to_numeric(combined_df[column], errors="coerce")
            combined_df = combined_df.dropna(subset=column)
            combined_df[column] = combined_df[column].astype(dtype)
        else:
            combined_df[column] = combined_df[column].astype(dtype)

    return combined_df
