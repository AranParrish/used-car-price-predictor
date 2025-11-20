import pytest
from pathlib import Path
import pandas as pd
import pandas.api.types as ptypes

from src.data_loader import load_data


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


@pytest.mark.describe("Valid data tests")
class TestValidData:

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


@pytest.mark.describe("Error handling")
class TestErrorHandling:

    @pytest.mark.it("Raises exception for invalid data folder")
    def test_invalid_path(self):
        invalid_path = Path("data/invalid/")
        with pytest.raises(ValueError) as excinfo:
            df = load_data(invalid_path)
        assert f"No valid CSV files found at {invalid_path}" in str(excinfo.value)

    @pytest.mark.it("Removes invalid rows")
    def test_remove_invalid_data(self, invalid_test_data):
        df = load_data(invalid_test_data)
        assert df.isna().sum().sum() == 0

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
