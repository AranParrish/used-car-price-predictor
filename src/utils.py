import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.random import RandomState


def train_test_datasets(
    df: pd.DataFrame,
    target_col: str,
    test_size: int | float = 0.2,
    random_seed: RandomState | int = 42,
) -> tuple:
    """
    Function to split given dataset into test and training sets for machine learning models

    Args:
        df - Pandas dataframe containing full dataset (features and target)
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
            - input data does not contain any features
            - the random_seed is not an integer
        ValueError if:
            - input target column is not present in the dataframe
            - test_size is not a proportion between 0.0 and 1.0 or a valid number of samples
            - random_seed is outside the range [0, 2***32 - 1]
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input dataset must be a pandas dataframe")

    if len(df.columns) < 2:
        raise TypeError("Dataframe does not contain any features")

    if isinstance(test_size, float) and (test_size < 0.0 or test_size > 1.0):
        raise ValueError(
            "test_size must be a float between 0.0 and 1.0 or an integer number of samples"
        )

    if isinstance(test_size, int) and (test_size < 0 or test_size > len(df)):
        raise ValueError(
            "test_size must be a float between 0.0 and 1.0 or an integer number of samples"
        )

    if not isinstance(random_seed, (int, RandomState)):
        raise TypeError("random_seed must be an integer or numpy RandomState instance")

    if isinstance(random_seed, int) and (random_seed < 0 or random_seed > (2**32 - 1)):
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
