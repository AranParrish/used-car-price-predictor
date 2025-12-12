import pytest, torch
import pandas as pd
from pathlib import Path
from src.utils import train_test_datasets, preprocessing, tensor_converter
from src.data_loader import load_data


@pytest.fixture(scope="function")
def cleansed_df():
    return load_data(Path("data/valid_test_data/"))


@pytest.fixture(scope="function")
def processed_df(cleansed_df):
    return preprocessing(cleansed_df)


@pytest.fixture(scope="function")
def processed_training_data(processed_df):
    return train_test_datasets(processed_df, target_col="price")


@pytest.mark.describe("Preprocessing function tests")
class TestPreprocessing:

    @pytest.mark.it("Input is not mutated")
    def test_input_not_mutated(self, cleansed_df):
        copy_df = cleansed_df.copy(deep=True)
        preprocessing(cleansed_df)
        pd.testing.assert_frame_equal(cleansed_df, copy_df)

    @pytest.mark.it("Returns a new dataframe")
    def test_returns_new_dataframe(self, cleansed_df):
        output = preprocessing(cleansed_df)
        assert isinstance(output, pd.DataFrame)
        assert output is not cleansed_df

    @pytest.mark.it("Removes categorical columns")
    def test_removes_categorical_columns(self, cleansed_df):
        output = preprocessing(cleansed_df)
        assert output.select_dtypes(include=["object", "string", "boolean"]).empty

    @pytest.mark.it("Dataset with no categorical columns returned unchanged")
    def test_no_categorical_returned(self, cleansed_df):
        numeric_only = cleansed_df[["price"]]
        output = preprocessing(numeric_only)
        pd.testing.assert_frame_equal(output, numeric_only)


@pytest.mark.describe("Preprocessing exception handling")
class TestPreprocessingExceptions:

    @pytest.mark.it("Raises TypeError if input is not a dataframe")
    def test_typeerror_not_a_dataframe(self):
        with pytest.raises(TypeError) as excinfo:
            preprocessing("not a dataframe")
        assert "Input must be a pandas dataframe" in str(excinfo.value)


@pytest.mark.describe("Train / Test function tests")
class TestTrainTestSplit:

    @pytest.mark.it("Input is not mutated")
    def test_input_not_mutated(self, processed_df):
        copy_df = processed_df.copy(deep=True)
        train_test_datasets(
            processed_df, target_col="price", test_size=0.2, random_seed=42
        )
        pd.testing.assert_frame_equal(processed_df, copy_df)

    @pytest.mark.it("Returns expected output structure")
    def test_output_structure(self, processed_df):
        output = train_test_datasets(processed_df, target_col="price")
        expected_numeric_features = ["year", "mileage", "tax", "mpg", "engineSize"]
        assert isinstance(output, tuple)
        assert len(output) == 4
        X_train, X_test, y_train, y_test = output
        assert all(feature in X_train.columns for feature in expected_numeric_features)
        assert all(feature in X_test.columns for feature in expected_numeric_features)
        assert "price" in y_train.columns and "price" in y_test.columns

    @pytest.mark.it("Returns expected train and test sample sizes")
    def test_train_test_sizes(self, processed_df):
        X_train, X_test, y_train, y_test = train_test_datasets(
            processed_df, target_col="price", test_size=0.2
        )
        expected_train_size = len(processed_df) * 0.8
        expected_test_size = len(processed_df) * 0.2
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

    @pytest.mark.it("Raises TypeError if input dataframe contains non-numeric columns")
    def test_input_non_numeric_cols(self, cleansed_df):
        with pytest.raises(TypeError) as excinfo:
            train_test_datasets(cleansed_df, target_col="price")
        assert "Input dataframe must not contain non-numeric columns" in str(
            excinfo.value
        )

    @pytest.mark.it("Raises ValueError if target col does not exist")
    def test_target_col_does_not_exist(self, processed_df):
        with pytest.raises(ValueError) as excinfo:
            train_test_datasets(processed_df, target_col="invalid")
        assert "Target column not in input dataset" in str(excinfo.value)

    @pytest.mark.it(
        "Raises TypeError if dataframe does not contain at least one feature and one target"
    )
    def test_df_without_features(self, processed_df):
        invalid_df = processed_df[["price"]]
        with pytest.raises(TypeError) as excinfo:
            train_test_datasets(invalid_df, target_col="price")
        assert (
            "Dataframe must contain at least one feature column and one target column"
            in str(excinfo.value)
        )

    @pytest.mark.it("Raises ValueError if given invalid test size proportion")
    def test_invalid_test_size_float(self, processed_df):
        with pytest.raises(ValueError) as excinfo:
            train_test_datasets(processed_df, target_col="price", test_size=1.1)
        assert (
            "test_size must be a float between 0.0 and 1.0 or an integer number of samples"
            in str(excinfo.value)
        )

    @pytest.mark.it("Raises ValueError if given test number of samples")
    def test_invalid_test_size_int(self, processed_df):
        invalid_sample_size = len(processed_df) + 1
        with pytest.raises(ValueError) as excinfo:
            train_test_datasets(
                processed_df, target_col="price", test_size=invalid_sample_size
            )
        assert (
            "test_size must be a float between 0.0 and 1.0 or an integer number of samples"
            in str(excinfo.value)
        )

    @pytest.mark.it("Raises TypeError if random seed is not a valid type")
    def test_random_seed_invalid_type(self, processed_df):
        with pytest.raises(TypeError) as excinfo:
            train_test_datasets(processed_df, target_col="price", random_seed="42")
        assert "random_seed must be an integer" in str(excinfo.value)

    @pytest.mark.it(
        "Raises ValueError if random seed is outside the valid integer range"
    )
    def test_random_seed_outside_integer_limits(self, processed_df):
        with pytest.raises(ValueError) as excinfo:
            train_test_datasets(processed_df, target_col="price", random_seed=-1)
        assert "random_seed must be an integer in the range [0, 2**32 - 1]" in str(
            excinfo.value
        )


@pytest.mark.describe("Tensor converter function tests")
class TestTensorConverter:

    @pytest.mark.it("Inputs are not mutated")
    def test_inputs_not_mutated(self, processed_training_data):
        X_train, _, y_train, _ = processed_training_data
        copy_X_train = X_train.copy(deep=True)
        copy_y_train = y_train.copy(deep=True)
        tensor_converter(X_train, y_train)
        pd.testing.assert_frame_equal(X_train, copy_X_train)
        pd.testing.assert_frame_equal(y_train, copy_y_train)

    @pytest.mark.it("Returns tensors")
    def test_returns_tensors(self, processed_training_data):
        X_train, _, y_train, _ = processed_training_data
        print(X_train.dtypes)
        X_tensor, y_tensor = tensor_converter(X_train, y_train)
        assert isinstance(X_tensor, torch.Tensor)
        assert isinstance(y_tensor, torch.Tensor)


@pytest.mark.describe("Tensor converter exception handling")
class TestTensorConverterExceptions:

    @pytest.mark.it("Raises TypeError if either input is not a dataframe")
    def test_typeerror_input_not_dataframe(self):
        with pytest.raises(TypeError) as excinfo:
            tensor_converter("not a dataframe", "not a dataframe")
        assert "Inputs must both be a pandas dataframe" in str(excinfo.value)

    @pytest.mark.it("Raises TypeError if either input contains non-numeric data")
    def test_typeerror_non_numeric_cols(self, cleansed_df):
        X = cleansed_df.drop(columns="price", axis=1)
        y = cleansed_df[["price"]]
        with pytest.raises(TypeError) as excinfo:
            tensor_converter(X, y)
        assert "Inputs must not contain non-numeric values" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if inputs differ in length")
    def test_valueerror_differing_length_inputs(self, processed_training_data):
        X, _, y, _ = processed_training_data
        y = y.head()
        with pytest.raises(ValueError) as excinfo:
            tensor_converter(X, y)
        assert "X and y lengths must match" in str(excinfo.value)
