import pytest, torch
import pandas as pd
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

    @pytest.mark.it("Returns a new dataframe")
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

    @pytest.mark.it("Raises TypeError if input data is not a dataframe")
    def test_typeerror_not_a_dataframe(self):
        with pytest.raises(TypeError) as excinfo:
            linear_preprocessing("not a dataframe")
        assert "Input must be a pandas dataframe" in str(excinfo.value)

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
        output = linear_train_test_datasets(linear_processed_df, target_col="price")
        expected_numeric_features = ["year", "mileage", "tax", "mpg", "engineSize"]
        assert isinstance(output, tuple)
        assert len(output) == 4
        X_train, X_test, y_train, y_test = output
        assert all(feature in X_train.columns for feature in expected_numeric_features)
        assert all(feature in X_test.columns for feature in expected_numeric_features)
        assert "price" in y_train.columns and "price" in y_test.columns

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

    @pytest.mark.it("Raises TypeError if input is not a dataframe")
    def test_input_not_a_dataframe(self):
        invalid_input = []
        with pytest.raises(TypeError) as excinfo:
            linear_train_test_datasets(invalid_input, target_col="price")
        assert "Input dataset must be a pandas dataframe" in str(excinfo.value)

    @pytest.mark.it("Raises TypeError if input dataframe contains non-numeric columns")
    def test_input_non_numeric_cols(self, cleansed_df):
        with pytest.raises(TypeError) as excinfo:
            linear_train_test_datasets(cleansed_df, target_col="price")
        assert "Input dataframe must not contain non-numeric columns" in str(
            excinfo.value
        )

    @pytest.mark.it("Raises ValueError if target col does not exist")
    def test_target_col_does_not_exist(self, linear_processed_df):
        with pytest.raises(ValueError) as excinfo:
            linear_train_test_datasets(linear_processed_df, target_col="invalid")
        assert "Target column not in input dataset" in str(excinfo.value)

    @pytest.mark.it(
        "Raises TypeError if dataframe does not contain at least one feature and one target"
    )
    def test_df_without_features(self, linear_processed_df):
        invalid_df = linear_processed_df[["price"]]
        with pytest.raises(TypeError) as excinfo:
            linear_train_test_datasets(invalid_df, target_col="price")
        assert (
            "Dataframe must contain at least one feature column and one target column"
            in str(excinfo.value)
        )

    @pytest.mark.it("Raises ValueError if given invalid test size proportion")
    def test_invalid_test_size_float(self, linear_processed_df):
        with pytest.raises(ValueError) as excinfo:
            linear_train_test_datasets(
                linear_processed_df, target_col="price", test_size=1.1
            )
        assert (
            "test_size must be a float between 0.0 and 1.0 or an integer number of samples"
            in str(excinfo.value)
        )

    @pytest.mark.it("Raises ValueError if given test number of samples")
    def test_invalid_test_size_int(self, linear_processed_df):
        invalid_sample_size = len(linear_processed_df) + 1
        with pytest.raises(ValueError) as excinfo:
            linear_train_test_datasets(
                linear_processed_df, target_col="price", test_size=invalid_sample_size
            )
        assert (
            "test_size must be a float between 0.0 and 1.0 or an integer number of samples"
            in str(excinfo.value)
        )

    @pytest.mark.it("Raises TypeError if random seed is not a valid type")
    def test_random_seed_invalid_type(self, linear_processed_df):
        with pytest.raises(TypeError) as excinfo:
            linear_train_test_datasets(
                linear_processed_df, target_col="price", random_seed="42"
            )
        assert "random_seed must be an integer" in str(excinfo.value)

    @pytest.mark.it(
        "Raises ValueError if random seed is outside the valid integer range"
    )
    def test_random_seed_outside_integer_limits(self, linear_processed_df):
        with pytest.raises(ValueError) as excinfo:
            linear_train_test_datasets(
                linear_processed_df, target_col="price", random_seed=-1
            )
        assert "random_seed must be an integer in the range [0, 2**32 - 1]" in str(
            excinfo.value
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
        assert isinstance(y, pd.DataFrame)
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


@pytest.mark.describe("Embeddings Preprocessing Exception Handling")
class TestEmbeddingsPreprocessingExceptions:

    @pytest.mark.it("Raises TypeError if input data is not a dataframe")
    def test_typeerror_input_data_not_a_df(self):
        with pytest.raises(TypeError) as excinfo:
            embeddings_preprocessing("not a dataframe", target_col="price")
        assert "Input data must be a pandas dataframe" in str(excinfo.value)

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
        pd.testing.assert_frame_equal(copy_y, y)

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


# @pytest.mark.describe("Tensor converter exception handling")
# class TestTensorConverterExceptions:

#     @pytest.mark.it("Raises TypeError if either input is not a dataframe")
#     def test_typeerror_input_not_dataframe(self):
#         with pytest.raises(TypeError) as excinfo:
#             tensor_converter("not a dataframe", "not a dataframe")
#         assert "Inputs must both be a pandas dataframe" in str(excinfo.value)

#     @pytest.mark.it("Raises TypeError if either input contains non-numeric data")
#     def test_typeerror_non_numeric_cols(self, cleansed_df):
#         X = cleansed_df.drop(columns="price", axis=1)
#         y = cleansed_df[["price"]]
#         with pytest.raises(TypeError) as excinfo:
#             tensor_converter(X, y)
#         assert "Inputs must not contain non-numeric values" in str(excinfo.value)

#     @pytest.mark.it("Raises ValueError if inputs differ in length")
#     def test_valueerror_differing_length_inputs(self, linear_processed_training_data):
#         X, _, y, _ = linear_processed_training_data
#         y = y.head()
#         with pytest.raises(ValueError) as excinfo:
#             tensor_converter(X, y)
#         assert "X and y lengths must match" in str(excinfo.value)
