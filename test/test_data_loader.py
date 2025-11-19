import pytest
from pathlib import Path
import pandas as pd
import pandas.api.types as ptypes

from src.data_loader import load_data


@pytest.fixture(scope="function")
def source_data():
    return Path("data/raw_data/")


@pytest.mark.describe("Load data function tests")
class TestLoadData:

    @pytest.mark.it("Returns a dataframe")
    def test_load_data_returns_dataframe(self, source_data):
        df = load_data(source_data)
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.it("Dataframe contains expected columns")
    def test_load_data_returns_expected_columns(self, source_data):
        expected_columns = [
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
        df = load_data(source_data)
        assert all(column in expected_columns for column in df.columns)

    @pytest.mark.it("Loaded data is of expected type")
    def test_load_data_returns_expected_types(self, source_data):
        string_types = ["brand", "model", "transmission", "fuelType"]
        int_types = ["year", "price", "mileage", "tax"]
        float_types = ["mpg", "engineSize"]
        df = load_data(source_data)
        assert all(ptypes.is_string_dtype(df[str_col]) for str_col in string_types)
        assert all(ptypes.is_integer_dtype(df[int_col]) for int_col in int_types)
        assert all(ptypes.is_float_dtype(df[float_col]) for float_col in float_types)
