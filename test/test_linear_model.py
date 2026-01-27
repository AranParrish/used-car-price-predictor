import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
from dataclasses import dataclass
from src.linear_model import linear_reg_model, evaluate_linear_model
from src.data_loader import load_data
from src.utils import linear_train_test_datasets, linear_preprocessing


@dataclass(frozen=True)
class DataInputs:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@pytest.fixture(scope="function")
def sample_data():
    df = load_data(Path("data/valid_test_data/"))
    processed_df = linear_preprocessing(df)
    X_train, X_test, y_train, y_test = linear_train_test_datasets(
        processed_df, target_col="price"
    )
    return DataInputs(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


@pytest.fixture(scope="function")
def sample_model(sample_data):
    return linear_reg_model(sample_data.X_train, sample_data.y_train)


@pytest.mark.describe("Linear regression model function tests")
class TestLinearRegFunction:

    @pytest.mark.it("Inputs are not mutated")
    def test_inputs_not_mutated(self, sample_data):
        copy_X_train = sample_data.X_train.copy(deep=True)
        copy_y_train = sample_data.y_train.copy(deep=True)
        linear_reg_model(sample_data.X_train, sample_data.y_train)
        pd.testing.assert_frame_equal(copy_X_train, sample_data.X_train)
        pd.testing.assert_series_equal(copy_y_train, sample_data.y_train)

    @pytest.mark.it("Returns a linear regression model")
    def test_returns_linear_regression_model(self, sample_data):
        output = linear_reg_model(sample_data.X_train, sample_data.y_train)
        assert isinstance(output, LinearRegression)

    @pytest.mark.it("Returns a trained linear regression model")
    def test_returns_trained_model(self, sample_data):
        output = linear_reg_model(sample_data.X_train, sample_data.y_train)
        assert hasattr(output, "coef_")

    @pytest.mark.it("Trained model produces predictions with expected shape")
    def test_trained_model_predictions_shape(self, sample_data):
        output = linear_reg_model(sample_data.X_train, sample_data.y_train)
        preds = output.predict(sample_data.X_test)
        assert preds.shape == sample_data.y_test.shape
        assert len(preds) == len(sample_data.X_test)


@pytest.mark.describe("Linear regression model exception handling")
class TestLinearRegExceptions:

    @pytest.mark.it("Raises TypeError if features data is not DataFrames")
    def test_typeerror_features_not_a_dataframe(self):
        test_y_train = pd.Series()
        with pytest.raises(TypeError) as excinfo:
            linear_reg_model("not a DataFrame", test_y_train)
        assert "X_train must be a pandas DataFrame" in str(excinfo.value)

    @pytest.mark.it("Raises TypeError if target data is not a series")
    def test_typeerror_target_not_a_series(self):
        test_X_train = pd.DataFrame()
        with pytest.raises(TypeError) as excinfo:
            linear_reg_model(test_X_train, "not a series")
        assert "y_train must be a pandas Series" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if an features data contains non-numeric values")
    def test_valueerror_non_numeric_in_features(self):
        df = load_data(Path("data/valid_test_data/"))
        y = df["price"]
        X = df.drop(columns="price")
        invalid_X_train, _, invalid_y_train, _ = train_test_split(X, y, test_size=0.2)
        with pytest.raises(ValueError) as excinfo:
            linear_reg_model(invalid_X_train, invalid_y_train)
        assert "X_train must only contain numeric values" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if target is non-numeric")
    def test_valueerror_non_numeric_target(self, sample_data):
        df = load_data(Path("data/valid_test_data/"))
        invalid_y_train = df["model"]
        with pytest.raises(ValueError) as excinfo:
            linear_reg_model(sample_data.X_train, invalid_y_train)
        assert "y_train must only contain numeric values" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if data contains missing values")
    def test_missing_values(self, sample_data):
        invalid_data = Path("data/invalid_test_data/ford.csv")
        df = pd.read_csv(invalid_data)
        invalid_X_train = df[["year", "engineSize"]]
        with pytest.raises(ValueError) as excinfo:
            linear_reg_model(invalid_X_train, sample_data.y_train)
        assert "Input data must not contain missing values" in str(excinfo.value)


@pytest.mark.describe("Evaluate Linear Model function tests")
class TestEvaluateLinearModel:

    @pytest.mark.it("Inputs not mutated")
    def test_inputs_not_mutated(self, sample_model, sample_data):
        copy_X_test = sample_data.X_test.copy(deep=True)
        copy_y_test = sample_data.y_test.copy(deep=True)
        copy_model_coef = sample_model.coef_.copy()
        copy_model_intercept = sample_model.intercept_
        evaluate_linear_model(sample_model, sample_data.X_test, sample_data.y_test)
        pd.testing.assert_frame_equal(sample_data.X_test, copy_X_test)
        pd.testing.assert_series_equal(sample_data.y_test, copy_y_test)
        assert np.allclose(sample_model.coef_, copy_model_coef)
        assert sample_model.intercept_ == copy_model_intercept

    @pytest.mark.it("Returns expected format")
    def test_returns_expected_format(self, sample_model, sample_data):
        output = evaluate_linear_model(
            sample_model, sample_data.X_test, sample_data.y_test
        )
        assert isinstance(output, dict)
        assert all(
            isinstance(key, str) and isinstance(value, float)
            for key, value in output.items()
        )

    @pytest.mark.it("Outputs expected metrics")
    def test_returns_expected_metrics(self, sample_model, sample_data):
        expected_metrics = {"mse", "rmse", "mae", "r2"}
        output = evaluate_linear_model(
            sample_model, sample_data.X_test, sample_data.y_test
        )
        assert all(key in output.keys() for key in expected_metrics)
