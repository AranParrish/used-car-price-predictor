import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.linear_model import linear_reg_model
from src.data_loader import load_data
from src.utils import linear_train_test_datasets, linear_preprocessing


@pytest.fixture(scope="function")
def valid_training_data():
    df = load_data(Path("data/valid_test_data/"))
    processed_df = linear_preprocessing(df)
    return linear_train_test_datasets(processed_df, target_col="price")


@pytest.mark.describe("Linear regression model function tests")
class TestLinearRegFunction:

    @pytest.mark.it("Inputs are not mutated")
    def test_inputs_not_mutated(self, valid_training_data):
        X_train, _, y_train, _ = valid_training_data
        copy_X_train = X_train.copy(deep=True)
        copy_y_train = y_train.copy(deep=True)
        linear_reg_model(X_train, y_train)
        pd.testing.assert_frame_equal(copy_X_train, X_train)
        pd.testing.assert_series_equal(copy_y_train, y_train)

    @pytest.mark.it("Returns a linear regression model")
    def test_returns_linear_regression_model(self, valid_training_data):
        X_train, _, y_train, _ = valid_training_data
        output = linear_reg_model(X_train, y_train)
        assert isinstance(output, LinearRegression)

    @pytest.mark.it("Returns a trained linear regression model")
    def test_returns_trained_model(self, valid_training_data):
        X_train, _, y_train, _ = valid_training_data
        output = linear_reg_model(X_train, y_train)
        assert hasattr(output, "coef_")

    @pytest.mark.it("Trained model produces predictions with expected shape")
    def test_trained_model_predictions_shape(self, valid_training_data):
        X_train, X_test, y_train, y_test = valid_training_data
        output = linear_reg_model(X_train, y_train)
        preds = output.predict(X_test)
        assert preds.shape == y_test.shape
        assert len(preds) == len(X_test)


@pytest.mark.describe("Linear regression model exception handling")
class TestLinearRegExceptions:

    @pytest.mark.it("Raises TypeError if features data is not dataframes")
    def test_typeerror_features_not_a_dataframe(self):
        test_y_train = pd.Series()
        with pytest.raises(TypeError) as excinfo:
            linear_reg_model("not a dataframe", test_y_train)
        assert "X_train must be a pandas Dataframe" in str(excinfo.value)

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
        X = df.drop(columns="price", axis=1)
        invalid_X_train, _, invalid_y_train, _ = train_test_split(X, y, test_size=0.2)
        with pytest.raises(ValueError) as excinfo:
            linear_reg_model(invalid_X_train, invalid_y_train)
        assert "X_train must only contain numeric values" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if target is non-numeric")
    def test_valueerror_non_numeric_target(self, valid_training_data):
        df = load_data(Path("data/valid_test_data/"))
        (
            X_train,
            _,
            _,
            _,
        ) = valid_training_data
        invalid_y_train = df["model"]
        with pytest.raises(ValueError) as excinfo:
            linear_reg_model(X_train, invalid_y_train)
        assert "y_train must only contain numeric values" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if features contains invalid data")
    def test_invalid_data_in_features(self, valid_training_data):
        _, _, valid_y_train, _ = valid_training_data
        invalid_data = Path("data/invalid_test_data/ford.csv")
        df = pd.read_csv(invalid_data)
        invalid_X_train = df[["year", "engineSize"]]
        with pytest.raises(ValueError) as excinfo:
            linear_reg_model(invalid_X_train, valid_y_train)
        assert "X_train must not contain missing values" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if target contains invalid data")
    def test_invalid_data_in_target(self, valid_training_data):
        valid_X_train, _, _, _ = valid_training_data
        invalid_data = Path("data/invalid_test_data/ford.csv")
        df = pd.read_csv(invalid_data)
        invalid_y_train = df["engineSize"]
        with pytest.raises(ValueError) as excinfo:
            linear_reg_model(valid_X_train, invalid_y_train)
        assert "y_train must not contain missing values" in str(excinfo.value)
