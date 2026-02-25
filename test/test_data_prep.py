import pytest
from pathlib import Path
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from sklearn.preprocessing import StandardScaler

from src.data_prep import load_data, split_datasets, scale_num_data


@pytest.fixture(scope="function")
def valid_test_data():
    return Path("data/valid_test_data/")


@pytest.fixture(scope="function")
def invalid_test_data():
    return Path("data/invalid_test_data/")


@pytest.fixture(scope="function")
def expected_columns():
    return [
        "brand",
        "model",
        "year",
        "price",
        "transmission",
        "mileage",
        "fuelType",
        "tax",
        "mpg",
        "engineSize",
    ]


@pytest.fixture(scope="function")
def cleansed_df():
    return load_data(Path("data/valid_test_data/"))


@pytest.fixture(scope="function")
def split_data(cleansed_df):
    return split_datasets(cleansed_df, target_col="price")


@pytest.mark.describe("Load Data function tests")
class TestLoadData:

    @pytest.mark.it("Returns a dataframe")
    def test_load_data_returns_dataframe(self, valid_test_data):
        df = load_data(valid_test_data)
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.it("Dataframe contains expected columns")
    def test_load_data_returns_expected_columns(
        self, valid_test_data, expected_columns
    ):
        df = load_data(valid_test_data)
        assert all(column in df.columns for column in expected_columns)

    @pytest.mark.it("Loaded data is of expected type")
    def test_load_data_returns_expected_types(self, valid_test_data):
        string_types = ["brand", "model", "transmission", "fuelType"]
        int_types = ["year", "price", "mileage", "tax"]
        float_types = ["mpg", "engineSize"]
        df = load_data(valid_test_data)
        assert all(ptypes.is_string_dtype(df[str_col]) for str_col in string_types)
        assert all(ptypes.is_integer_dtype(df[int_col]) for int_col in int_types)
        assert all(ptypes.is_float_dtype(df[float_col]) for float_col in float_types)

    @pytest.mark.it("Combines tax and tax(£) columns if both present")
    def test_combines_tax_columns(self):
        input_data = Path("data/raw_data/")
        df = load_data(input_data)
        assert "tax(£)" not in df.columns
        assert len(df.columns) == 10


@pytest.mark.describe("Load Data exception handling")
class TestLoadDataExceptions:

    @pytest.mark.it("Raises exception for invalid data folder")
    def test_invalid_path(self):
        invalid_path = Path("data/invalid/")
        with pytest.raises(ValueError) as excinfo:
            df = load_data(invalid_path)
        assert f"No valid CSV files found at {invalid_path}" in str(excinfo.value)

    @pytest.mark.it("Removes invalid rows")
    def test_remove_invalid_data(self, invalid_test_data, caplog):
        with caplog.at_level("WARNING"):
            df = load_data(invalid_test_data)
        assert df.isna().sum().sum() == 0
        assert (
            "Dropped 9 rows from combined dataset due to invalid value(s)"
            in caplog.text
        )

    @pytest.mark.it("Strips extra columns")
    def test_extra_columns_removed(self, invalid_test_data, expected_columns):
        df = load_data(invalid_test_data)
        extra_cols = set(df.columns) - set(expected_columns)
        assert extra_cols == set()

    @pytest.mark.it("Excludes data with missing columns")
    def test_data_missing_cols(self, tmp_path, caplog):
        # Note: need to include some valid data as will otherwise raise ValueError and exit for having no valid data
        df_all_cols = pd.read_csv("data/invalid_test_data/ford.csv")
        df_missing_cols = df_all_cols.drop(columns=["year"])
        all_cols_file = tmp_path / "ford.csv"
        df_all_cols.to_csv(all_cols_file, index=False)
        missing_file = tmp_path / "ford_missing.csv"
        df_missing_cols.to_csv(missing_file, index=False)
        with caplog.at_level("WARNING"):
            load_data(tmp_path)
        assert missing_file.name in caplog.text
        assert "year" in caplog.text


@pytest.mark.describe("Split datasets function tests")
class TestSplitDatasets:

    @pytest.mark.it("Input is not mutated")
    def test_input_not_mutated(self, cleansed_df):
        copy_df = cleansed_df.copy(deep=True)
        split_datasets(cleansed_df, target_col="price", test_size=0.2, random_seed=42)
        pd.testing.assert_frame_equal(cleansed_df, copy_df)

    @pytest.mark.it("Returns expected output structure")
    def test_output_structure(self, cleansed_df):
        expected_level1_keys = {"train", "test"}
        expected_level2_keys = {"X_num", "X_cat", "y"}
        output = split_datasets(cleansed_df, target_col="price")
        assert isinstance(output, dict)
        assert all(key in expected_level1_keys for key in output.keys())
        assert all(key in expected_level2_keys for key in output["train"].keys())
        assert all(key in expected_level2_keys for key in output["test"].keys())

    @pytest.mark.it("Returns expected train and test sample sizes")
    def test_train_test_sizes(self, cleansed_df):
        output = split_datasets(cleansed_df, target_col="price", test_size=0.2)
        expected_train_size = len(cleansed_df) * 0.8
        expected_test_size = len(cleansed_df) * 0.2
        assert len(output["train"]["X_num"]) == expected_train_size
        assert len(output["train"]["X_cat"]) == expected_train_size
        assert len(output["train"]["y"]) == expected_train_size
        assert len(output["test"]["X_num"]) == expected_test_size
        assert len(output["test"]["X_cat"]) == expected_test_size
        assert len(output["test"]["y"]) == expected_test_size

    @pytest.mark.it("Returns expected numerical and categorical features")
    def test_num_cat_as_expected(self, cleansed_df):
        expected_num_cols = {"year", "mileage", "tax", "mpg", "engineSize"}
        expected_cat_cols = {"brand", "model", "transmission", "fuelType"}
        output = split_datasets(cleansed_df, target_col="price")
        X_num_train_cols = set(output["train"]["X_num"].columns)
        X_num_test_cols = set(output["test"]["X_num"].columns)
        X_cat_train_cols = set(output["train"]["X_cat"].columns)
        X_cat_test_cols = set(output["test"]["X_cat"].columns)
        assert X_num_train_cols == X_num_test_cols == expected_num_cols
        assert X_cat_train_cols == X_cat_test_cols == expected_cat_cols

    @pytest.mark.it("Converts bool features to numeric")
    def test_bool_to_numeric(self):
        test_df = pd.DataFrame(
            {
                "is_automatic": [True, False, True],
                "mileage": [100, 200, 300],
                "brand": ["Ford", "BMW", "Ford"],
                "price": [10, 20, 30],
            }
        )
        output = split_datasets(test_df, target_col="price", test_size=1)
        assert "is_automatic" in output["train"]["X_num"].columns
        assert output["train"]["X_num"]["is_automatic"].dtype == "int16"


@pytest.mark.describe("Linear Train / Test exception handling")
class TestSplitDatasetsExceptions:

    @pytest.mark.it("Raises TypeError if input is not a DataFrame")
    def test_input_not_a_dataframe(self):
        not_a_df = []
        with pytest.raises(TypeError) as excinfo:
            split_datasets(not_a_df, target_col="price")
        assert "df must be a pandas DataFrame" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if target col does not exist")
    def test_target_col_does_not_exist(self, cleansed_df):
        with pytest.raises(ValueError) as excinfo:
            split_datasets(cleansed_df, target_col="invalid")
        assert "Target column not in df" in str(excinfo.value)

    @pytest.mark.it(
        "Raises ValueError if DataFrame does not contain at least two feature columns and one target"
    )
    def test_df_without_features(self, cleansed_df):
        invalid_df = cleansed_df[["price"]]
        with pytest.raises(ValueError) as excinfo:
            split_datasets(invalid_df, target_col="price")
        assert (
            "df must contain at least two feature columns and one target column"
            in str(excinfo.value)
        )

    @pytest.mark.it("Raises ValueError if no numerical features")
    def test_no_num_features(self, cleansed_df):
        y = cleansed_df["price"]
        no_num_features = cleansed_df.select_dtypes(include=["object", "string"])
        test_df = no_num_features.join(y)
        with pytest.raises(ValueError) as excinfo:
            split_datasets(test_df, target_col="price")
        assert "df must contain both numerical and categorical features" in str(
            excinfo.value
        )

    @pytest.mark.it("Raises ValueError if no categorical features")
    def test_no_cat_features(self, cleansed_df):
        test_df = cleansed_df.select_dtypes(exclude=["object", "string"])
        with pytest.raises(ValueError) as excinfo:
            split_datasets(test_df, target_col="price")
        assert "df must contain both numerical and categorical features" in str(
            excinfo.value
        )

    @pytest.mark.it("Raises ValueError if df contains invalid rows")
    def test_df_contains_invalid_rows(self):
        invalid_df = pd.read_csv("data/invalid_test_data/ford.csv")
        with pytest.raises(ValueError) as excinfo:
            split_datasets(invalid_df, target_col="price")
        assert "df contains invalid rows" in str(excinfo.value)


@pytest.mark.describe("Scale num data function tests")
class TestScaleNumData:

    @pytest.mark.it("Inputs are not mutated")
    def test_inputs_not_mutated(self, split_data):
        copy_X_num_train = split_data["train"]["X_num"].copy()
        copy_X_num_test = split_data["test"]["X_num"].copy()
        scale_num_data(split_data["train"]["X_num"], split_data["test"]["X_num"])
        pd.testing.assert_frame_equal(copy_X_num_train, split_data["train"]["X_num"])
        pd.testing.assert_frame_equal(copy_X_num_test, split_data["test"]["X_num"])

    @pytest.mark.it("Returns expected output structure")
    def test_output_structure(self, split_data):
        output = scale_num_data(
            split_data["train"]["X_num"], split_data["test"]["X_num"]
        )
        X_num_train_scaled, X_num_test_scaled, scalar = output
        assert isinstance(output, tuple)
        assert isinstance(X_num_train_scaled, pd.DataFrame)
        assert isinstance(X_num_test_scaled, pd.DataFrame)
        assert isinstance(scalar, StandardScaler)

    @pytest.mark.it("Scales numerical train features")
    def test_scales_num_features(self, split_data):
        X_num_train_scaled, _, _ = scale_num_data(
            split_data["train"]["X_num"], split_data["test"]["X_num"]
        )
        for col in X_num_train_scaled.columns:
            # Skip any binary columns
            if X_num_train_scaled[col].nunique() > 2:
                assert np.isclose(
                    X_num_train_scaled[col].mean(), 0.0
                ), f"Column {col} was not centered."
                assert np.isclose(
                    X_num_train_scaled[col].std(ddof=0), 1.0
                ), f"Column {col} does not have unit variance."

    @pytest.mark.it("Binary columns unaltered")
    def test_binary_unaltered(self):
        test_df = pd.DataFrame(
            {
                "mileage": [100, 200, 300],
                "is_auto": [1.0, 0.0, 1.0],
            }
        )
        X_num_train_scaled, X_num_test_scaled, _ = scale_num_data(test_df, test_df)
        assert X_num_train_scaled["is_auto"].iloc[0] == 1.0
        assert X_num_train_scaled["is_auto"].nunique() == 2
        assert X_num_test_scaled["is_auto"].iloc[0] == 1.0
        assert X_num_test_scaled["is_auto"].nunique() == 2

    @pytest.mark.it("No leakage of test data")
    def test_no_test_data_leakage(self):
        # Train mean = 100.0
        example_X_num_train = pd.DataFrame({"mileage": [90.0, 100.0, 110.0]})
        # Test mean = 0.0
        example_X_num_test = pd.DataFrame({"mileage": [-10.0, 0.0, 10.0]})
        X_num_train_scaled, X_num_test_scaled, scaler = scale_num_data(
            example_X_num_train, example_X_num_test
        )
        assert np.isclose(X_num_train_scaled["mileage"].mean(), 0.0)
        assert not np.isclose(X_num_test_scaled["mileage"].mean(), 0.0)
        assert np.isclose(scaler.mean_[0], 100.0)


@pytest.mark.describe("Scale num data exception handling")
class TestScaleNumDataExceptions:

    @pytest.mark.it("Raises TypeError if inputs not DataFrames")
    def test_inputs_not_dataframes(self):
        invalid_X_num_train = {}
        invalid_X_num_test = {}
        with pytest.raises(TypeError) as excinfo:
            scale_num_data(invalid_X_num_train, invalid_X_num_test)
        assert "Inputs must both be pandas DataFrames" in str(excinfo.value)

    @pytest.mark.it("Raises TypeError for categorical features in any input")
    def test_cat_features_input(self):
        X_num_train_strings = pd.DataFrame({"brand": ["Ford", "BMW"]})
        X_num_test_strings = pd.DataFrame({"brand": ["Ford", "BMW"]})
        with pytest.raises(TypeError) as excinfo:
            scale_num_data(X_num_train_strings, X_num_test_strings)
        assert "Inputs must only contain numeric features" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError for invalid rows in train data")
    def test_invalid_rows_train_data(self, split_data):
        invalid_df = pd.read_csv("data/invalid_test_data/ford.csv")
        invalid_X_num_train = invalid_df.select_dtypes(include=[np.number])
        with pytest.raises(ValueError) as excinfo:
            scale_num_data(invalid_X_num_train, split_data["test"]["X_num"])
        assert "Inputs must not contain invalid rows" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError for invalid rows in test data")
    def test_invalid_rows_test_data(self, split_data):
        invalid_df = pd.read_csv("data/invalid_test_data/ford.csv")
        invalid_X_num_test = invalid_df.select_dtypes(include=[np.number])
        with pytest.raises(ValueError) as excinfo:
            scale_num_data(split_data["train"]["X_num"], invalid_X_num_test)
        assert "Inputs must not contain invalid rows" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if inputs contain differing number of columns")
    def test_differing_num_columns(self, split_data):
        invalid_X_num_test = split_data["test"]["X_num"].drop(columns="mileage")
        with pytest.raises(ValueError) as excinfo:
            scale_num_data(split_data["train"]["X_num"], invalid_X_num_test)
        assert "Inputs contain differing columns or column order" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if inputs contain differing columns names")
    def test_differing_columns_names(self, split_data):
        invalid_X_num_test = split_data["test"]["X_num"].rename(
            columns={"mileage": "odometer"}, inplace=False
        )
        with pytest.raises(ValueError) as excinfo:
            scale_num_data(split_data["train"]["X_num"], invalid_X_num_test)
        assert "Inputs contain differing columns or column order" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if column ordering differs for inputs")
    def test_differing_column_order(self, split_data):
        X_num_train = split_data["train"]["X_num"].copy()
        col_names = list(X_num_train.columns)
        col_names[0], col_names[1] = col_names[1], col_names[0]
        reordered_X_num_train = X_num_train.loc[:, col_names]
        with pytest.raises(ValueError) as excinfo:
            scale_num_data(reordered_X_num_train, split_data["test"]["X_num"])
        assert "Inputs contain differing columns or column order" in str(excinfo.value)
