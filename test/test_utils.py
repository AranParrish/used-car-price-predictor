import pytest
import pandas as pd
from pathlib import Path
from src.utils import train_test_datasets
from src.data_loader import load_data


@pytest.fixture(scope="function")
def cleansed_df():
    return load_data(Path("data/valid_test_data/"))


@pytest.mark.describe("Train / Test function tests")
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

    @pytest.mark.it("Returns expected train and test sample sizes")
    def test_train_test_sizes(self, cleansed_df):
        X_train, X_test, y_train, y_test = train_test_datasets(
            cleansed_df, target_col="price", test_size=0.2
        )
        expected_train_size = len(cleansed_df) * 0.8
        expected_test_size = len(cleansed_df) * 0.2
        assert len(X_train) == expected_train_size
        assert len(y_train) == expected_train_size
        assert len(X_test) == expected_test_size
        assert len(y_test) == expected_test_size


@pytest.mark.describe("Train / Test exception handling")
class TestTrainTestExceptions:

    @pytest.mark.it("Raises TypeError if input is not a dataframe")
    def test_input_not_a_dataframe(self):
        invalid_input = []
        with pytest.raises(TypeError) as excinfo:
            train_test_datasets(invalid_input, target_col="price")
        assert "Input dataset must be a pandas dataframe" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if target col does not exist")
    def test_target_col_does_not_exist(self, cleansed_df):
        with pytest.raises(ValueError) as excinfo:
            train_test_datasets(cleansed_df, target_col="invalid")
        assert "Target column not in input dataset" in str(excinfo.value)

    @pytest.mark.it(
        "Raises TypeError if dataframe does not contain at least one feature and one target"
    )
    def test_df_without_features(self, cleansed_df):
        invalid_df = cleansed_df[["price"]]
        with pytest.raises(TypeError) as excinfo:
            train_test_datasets(invalid_df, target_col="price")
        assert (
            "Dataframe must contain at least one feature column and one target column"
            in str(excinfo.value)
        )

    @pytest.mark.it("Raises ValueError if given invalid test size proportion")
    def test_invalid_test_size_float(self, cleansed_df):
        with pytest.raises(ValueError) as excinfo:
            train_test_datasets(cleansed_df, target_col="price", test_size=1.1)
        assert (
            "test_size must be a float between 0.0 and 1.0 or an integer number of samples"
            in str(excinfo.value)
        )

    @pytest.mark.it("Raises ValueError if given test number of samples")
    def test_invalid_test_size_int(self, cleansed_df):
        invalid_sample_size = len(cleansed_df) + 1
        with pytest.raises(ValueError) as excinfo:
            train_test_datasets(
                cleansed_df, target_col="price", test_size=invalid_sample_size
            )
        assert (
            "test_size must be a float between 0.0 and 1.0 or an integer number of samples"
            in str(excinfo.value)
        )

    @pytest.mark.it("Raises TypeError if random seed is not a valid type")
    def test_random_seed_invalid_type(self, cleansed_df):
        with pytest.raises(TypeError) as excinfo:
            train_test_datasets(cleansed_df, target_col="price", random_seed="42")
        assert "random_seed must be an integer" in str(excinfo.value)

    @pytest.mark.it(
        "Raises ValueError if random seed is outside the valid integer range"
    )
    def test_random_seed_outside_integer_limits(self, cleansed_df):
        with pytest.raises(ValueError) as excinfo:
            train_test_datasets(cleansed_df, target_col="price", random_seed=-1)
        assert "random_seed must be an integer in the range [0, 2**32 - 1]" in str(
            excinfo.value
        )
