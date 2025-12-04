import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_datasets(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple:
    y = df[[target_col]]
    X = df.drop(columns=target_col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    return X_train, X_test, y_train, y_test
