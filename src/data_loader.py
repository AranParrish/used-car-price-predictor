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
        TBD
    """

    # Read in source data and combine into a single dataframe
    dfs = {}
    for csv_file in data_dir.glob("*.csv"):
        brand = csv_file.stem
        df = pd.read_csv(csv_file)
        df["brand"] = brand
        dfs[brand] = df
    combined_df = pd.concat(dfs.values(), ignore_index=True, sort=False)

    # Combine "tax" and "tax(£)" column values into a single column
    combined_df.fillna({"tax(£)": 0}, inplace=True)
    combined_df.fillna({"tax": 0}, inplace=True)
    combined_df["tax"] = combined_df["tax"] + combined_df["tax(£)"]
    combined_df.drop(labels="tax(£)", axis=1, inplace=True)

    # Ensure correct types for all columns
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
    for column, dtype in types_map.items():
        if column in combined_df.columns:
            if dtype in ("int64", "float64"):
                combined_df[column] = pd.to_numeric(
                    combined_df[column], errors="coerce"
                )
                combined_df[column] = combined_df[column].astype(dtype)
            else:
                combined_df[column] = combined_df[column].astype(dtype)

    return combined_df
