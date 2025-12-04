import pytest
import pandas as pd
from pathlib import Path
from src.utils import train_test_datasets
from src.data_loader import load_data


@pytest.fixture(scope="function")
def cleansed_df():
    return load_data(Path("data/valid_test_data/"))


@pytest.mark.describe("Train / Test Split tests")
class TestTrainTestSplit:

    @pytest.mark.it("Input is not mutated")
    def test_input_not_mutated(self, cleansed_df):
        copy_df = cleansed_df.copy(deep=True)
        train_test_datasets(
            cleansed_df, target_col="price", test_size=0.2, random_seed=42
        )
        pd.testing.assert_frame_equal(cleansed_df, copy_df)

    @pytest.mark.it("Returns expected output structure")
    def test_output_structure(self, cleansed_df):
        output = train_test_datasets(cleansed_df, target_col="price")
        assert isinstance(output, tuple)
        assert len(output) == 4
        X_train, X_test, y_train, y_test = output
        assert len(X_train.columns) == 9
        assert len(X_test.columns) == 9
        assert "price" in y_train.columns and "price" in y_test.columns
