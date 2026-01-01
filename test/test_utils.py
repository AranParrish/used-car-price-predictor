import pytest, torch
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import (
    linear_train_test_datasets,
    linear_preprocessing,
    embeddings_preprocessing,
    split_and_tensorise,
)
from src.data_loader import load_data


@pytest.fixture(scope="function")
def cleansed_df():
    return load_data(Path("data/valid_test_data/"))


@pytest.fixture(scope="function")
def linear_processed_df(cleansed_df):
    return linear_preprocessing(cleansed_df)


@pytest.fixture(scope="function")
def embeddings_preprocessing_data(cleansed_df):
    return embeddings_preprocessing(cleansed_df, target_col="price")


@pytest.mark.describe("Linear Preprocessing function tests")
class TestLinearPreprocessing:

    @pytest.mark.it("Input is not mutated")
    def test_input_not_mutated(self, cleansed_df):
        copy_df = cleansed_df.copy(deep=True)
        linear_preprocessing(cleansed_df)
        pd.testing.assert_frame_equal(cleansed_df, copy_df)

    @pytest.mark.it("Returns a new DataFrame")
    def test_returns_new_dataframe(self, cleansed_df):
        output = linear_preprocessing(cleansed_df)
        assert isinstance(output, pd.DataFrame)
        assert output is not cleansed_df

    @pytest.mark.it("Removes categorical columns")
    def test_removes_categorical_columns(self, cleansed_df):
        output = linear_preprocessing(cleansed_df)
        assert output.select_dtypes(include=["object", "string", "boolean"]).empty

    @pytest.mark.it("Dataset with no categorical columns returned unchanged")
    def test_no_categorical_returned(self, cleansed_df):
        numeric_only = cleansed_df[["price"]]
        output = linear_preprocessing(numeric_only)
        pd.testing.assert_frame_equal(output, numeric_only)


@pytest.mark.describe("Linear Preprocessing exception handling")
class TestLinearPreprocessingExceptions:

    @pytest.mark.it("Raises TypeError if input data is not a DataFrame")
    def test_typeerror_not_a_dataframe(self):
        with pytest.raises(TypeError) as excinfo:
            linear_preprocessing("not a DataFrame")
        assert "Input must be a pandas DataFrame" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if input data contains invalid rows")
    def test_valueerror_invalid_rows(self):
        invalid_data = Path("data/invalid_test_data/ford.csv")
        df = pd.read_csv(invalid_data)
        df["brand"] = "Ford"
        with pytest.raises(ValueError) as excinfo:
            linear_preprocessing(df)
        assert "Input data contains invalid rows" in str(excinfo.value)


@pytest.mark.describe("Linear Train / Test function tests")
class TestLinearTrainTestSplit:

    @pytest.mark.it("Input is not mutated")
    def test_input_not_mutated(self, linear_processed_df):
        copy_df = linear_processed_df.copy(deep=True)
        linear_train_test_datasets(
            linear_processed_df, target_col="price", test_size=0.2, random_seed=42
        )
        pd.testing.assert_frame_equal(linear_processed_df, copy_df)

    @pytest.mark.it("Returns expected output structure")
    def test_output_structure(self, linear_processed_df):
        outputs = linear_train_test_datasets(linear_processed_df, target_col="price")
        assert isinstance(outputs, list)
        assert len(outputs) == 4
        X_train, X_test, y_train, y_test = outputs
        assert all(
            isinstance(features_data, pd.DataFrame)
            for features_data in (X_train, X_test)
        )
        assert all(
            isinstance(target_data, pd.Series) for target_data in (y_train, y_test)
        )

    @pytest.mark.it("Returns expected train and test sample sizes")
    def test_train_test_sizes(self, linear_processed_df):
        X_train, X_test, y_train, y_test = linear_train_test_datasets(
            linear_processed_df, target_col="price", test_size=0.2
        )
        expected_train_size = len(linear_processed_df) * 0.8
        expected_test_size = len(linear_processed_df) * 0.2
        assert len(X_train) == expected_train_size
        assert len(y_train) == expected_train_size
        assert len(X_test) == expected_test_size
        assert len(y_test) == expected_test_size


@pytest.mark.describe("Linear Train / Test exception handling")
class TestLinearTrainTestExceptions:

    @pytest.mark.it("Raises TypeError if input is not a DataFrame")
    def test_input_not_a_dataframe(self):
        invalid_input = []
        with pytest.raises(TypeError) as excinfo:
            linear_train_test_datasets(invalid_input, target_col="price")
        assert "Input dataset must be a pandas DataFrame" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if input DataFrame contains non-numeric columns")
    def test_input_non_numeric_cols(self, cleansed_df):
        with pytest.raises(ValueError) as excinfo:
            linear_train_test_datasets(cleansed_df, target_col="price")
        assert "Input DataFrame must not contain non-numeric columns" in str(
            excinfo.value
        )

    @pytest.mark.it("Raises ValueError if target col does not exist")
    def test_target_col_does_not_exist(self, linear_processed_df):
        with pytest.raises(ValueError) as excinfo:
            linear_train_test_datasets(linear_processed_df, target_col="invalid")
        assert "Target column not in input dataset" in str(excinfo.value)

    @pytest.mark.it(
        "Raises ValueError if DataFrame does not contain at least one feature and one target"
    )
    def test_df_without_features(self, linear_processed_df):
        invalid_df = linear_processed_df[["price"]]
        with pytest.raises(ValueError) as excinfo:
            linear_train_test_datasets(invalid_df, target_col="price")
        assert (
            "DataFrame must contain at least one feature column and one target column"
            in str(excinfo.value)
        )


@pytest.mark.describe("Embeddings preprocessing function tests")
class TestEmbeddingsPreprocessing:

    @pytest.mark.it("Inputs are not mutated")
    def test_inputs_not_mutated(self, cleansed_df):
        copy_df = cleansed_df.copy(deep=True)
        embeddings_preprocessing(cleansed_df, target_col="price")
        pd.testing.assert_frame_equal(copy_df, cleansed_df)

    @pytest.mark.it("Output is expected structure")
    def test_output_expected_structure(self, cleansed_df):
        X_num, X_cat, y, mappings = embeddings_preprocessing(
            cleansed_df, target_col="price"
        )
        assert isinstance(X_num, pd.DataFrame)
        assert isinstance(X_cat, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(mappings, dict)

    @pytest.mark.it("Categorical columns are encoded")
    def test_cat_cols_encoded(self, cleansed_df):
        _, X_cat, _, _ = embeddings_preprocessing(cleansed_df, target_col="price")
        assert X_cat.select_dtypes(include=["object", "string", "boolean"]).empty

    @pytest.mark.it("Numeric columns are unchanged")
    def test_num_cols_unchanged(self, cleansed_df):
        features_df = cleansed_df.drop(columns="price")
        num_cols = features_df.select_dtypes(include=["number"])
        X_num, _, _, _ = embeddings_preprocessing(cleansed_df, target_col="price")
        pd.testing.assert_frame_equal(X_num, num_cols)

    @pytest.mark.it("Boolean columns converted to integer")
    def test_bool_becomes_int(self, cleansed_df):
        test_bool = np.random.default_rng(seed=42)
        cleansed_df["rand_bool"] = test_bool.choice(
            [True, False], size=len(cleansed_df)
        )
        X_num, X_cat, _, mappings = embeddings_preprocessing(
            cleansed_df, target_col="price"
        )
        assert "rand_bool" in X_num.columns
        assert X_num["rand_bool"].dtype == "int8"
        assert "rand_bool" not in X_cat.columns
        assert "rand_bool" not in mappings


@pytest.mark.describe("Embeddings Preprocessing Exception Handling")
class TestEmbeddingsPreprocessingExceptions:

    @pytest.mark.it("Raises TypeError if input data is not a DataFrame")
    def test_typeerror_input_data_not_a_dataframe(self):
        with pytest.raises(TypeError) as excinfo:
            embeddings_preprocessing("not a DataFrame", target_col="price")
        assert "Input data must be a pandas DataFrame" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if target column is not in data")
    def test_valueerror_target_col_not_in_data(self, cleansed_df):
        with pytest.raises(ValueError) as excinfo:
            embeddings_preprocessing(cleansed_df, target_col=2)
        assert "Target column not found in input data" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if input data contains invalid rows")
    def test_valueerror_for_invalid_rows(self):
        invalid_data = Path("data/invalid_test_data/ford.csv")
        df = pd.read_csv(invalid_data)
        df["brand"] = "Ford"
        with pytest.raises(ValueError) as excinfo:
            embeddings_preprocessing(df, target_col="price")
        assert "Input data contains invalid rows" in str(excinfo.value)


@pytest.mark.describe("Split and tensorise function tests")
class TestSplitTensorise:

    @pytest.mark.it("Inputs are not mutated")
    def test_inputs_not_mutated(self, embeddings_preprocessing_data):
        X_num, X_cat, y, _ = embeddings_preprocessing_data
        copy_X_num = X_num.copy(deep=True)
        copy_X_cat = X_cat.copy(deep=True)
        copy_y = y.copy(deep=True)
        split_and_tensorise(X_num, X_cat, y)
        pd.testing.assert_frame_equal(copy_X_num, X_num)
        pd.testing.assert_frame_equal(copy_X_cat, X_cat)
        pd.testing.assert_series_equal(copy_y, y)

    @pytest.mark.it("Returns tensors")
    def test_returns_tensors(self, embeddings_preprocessing_data):
        X_num, X_cat, y, _ = embeddings_preprocessing_data
        outputs = split_and_tensorise(X_num, X_cat, y)
        assert all(isinstance(output, torch.Tensor) for output in outputs)

    @pytest.mark.it("Returned tensor shapes match")
    def test_tensor_shapes_match(self, embeddings_preprocessing_data):
        X_num, X_cat, y, _ = embeddings_preprocessing_data
        X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = (
            split_and_tensorise(X_num, X_cat, y)
        )
        assert X_num_train.shape[0] == X_cat_train.shape[0] == y_train.shape[0]
        assert X_num_test.shape[0] == X_cat_test.shape[0] == y_test.shape[0]

    @pytest.mark.it("Returns expected output shapes")
    def test_returns_expected_shapes(self, embeddings_preprocessing_data):
        X_num, X_cat, y, _ = embeddings_preprocessing_data
        expected_train_size = X_num.shape[0] * 0.8
        expected_test_size = X_num.shape[0] * 0.2
        X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = (
            split_and_tensorise(X_num, X_cat, y)
        )
        assert (
            X_num_test.shape[0]
            == X_cat_test.shape[0]
            == y_test.shape[0]
            == expected_test_size
        )
        assert (
            X_num_train.shape[0]
            == X_cat_train.shape[0]
            == y_train.shape[0]
            == expected_train_size
        )
        assert X_num_test.shape[1] == X_num_train.shape[1] == X_num.shape[1]
        assert X_cat_test.shape[1] == X_cat_train.shape[1] == X_cat.shape[1]
        assert y_test.shape[1] == y_train.shape[1] == 1

    @pytest.mark.it("Train and test sizes returned as expected")
    def test_traintest_sizes(self):
        test_X_num = pd.DataFrame(np.random.rand(100, 3))
        test_X_cat = pd.DataFrame(np.random.rand(100, 2)).astype(dtype="int32")
        test_y = pd.Series(np.random.rand(100))
        X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = (
            split_and_tensorise(test_X_num, test_X_cat, test_y)
        )
        assert len(X_num_train) == len(X_cat_train) == len(y_train) == 80
        assert len(X_num_test) == len(X_cat_test) == len(y_test) == 20


@pytest.mark.describe("Split and tensorise exception handling")
class TestSplitTensoriseExceptions:

    @pytest.mark.it("Raises TypeError if any input data is not a DataFrame")
    def test_typeerror_any_input_data_not_dataframe(
        self, embeddings_preprocessing_data
    ):
        _, _, y, _ = embeddings_preprocessing_data
        with pytest.raises(TypeError) as excinfo:
            split_and_tensorise("not a DataFrame", "not a DataFrame", y)
        assert "Input features must be pandas DataFrame" in str(excinfo.value)

    @pytest.mark.it("Raises TypeError if target is not a Series")
    def test_target_not_a_series(self, embeddings_preprocessing_data):
        X_num, X_cat, _, _ = embeddings_preprocessing_data
        with pytest.raises(TypeError) as excinfo:
            split_and_tensorise(X_num, X_cat, "not a Series")
        assert "Target must be a pandas Series" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError for non-numeric values in numerical features")
    def test_valueerror_for_non_numeric_in_x_num(
        self, cleansed_df, embeddings_preprocessing_data
    ):
        _, X_cat, y, _ = embeddings_preprocessing_data
        X_num = cleansed_df.drop(columns="price")
        with pytest.raises(ValueError) as excinfo:
            split_and_tensorise(X_num, X_cat, y)
        assert "X_num must only contain numeric values" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError for non-integer values in categorical features")
    def test_valueerror_for_non_integer_in_x_cat(
        self, cleansed_df, embeddings_preprocessing_data
    ):
        X_num, _, y, _ = embeddings_preprocessing_data
        X_cat = cleansed_df.drop(columns="price")
        with pytest.raises(ValueError) as excinfo:
            split_and_tensorise(X_num, X_cat, y)
        assert "X_cat must only contain integer category code" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError for missing values")
    def test_missing_values(self, embeddings_preprocessing_data):
        invalid_data = Path("data/invalid_test_data/ford.csv")
        df = pd.read_csv(invalid_data)
        invalid_X_num = df[["engineSize"]]
        _, X_cat, y, _ = embeddings_preprocessing_data
        with pytest.raises(ValueError) as excinfo:
            split_and_tensorise(invalid_X_num, X_cat, y)
        assert "Input data must not contain missing values" in str(excinfo.value)
