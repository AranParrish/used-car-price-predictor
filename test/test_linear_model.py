import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.linear_model import linear_reg_model
from src.data_loader import load_data
from src.utils import train_test_datasets, preprocessing


@pytest.fixture(scope="function")
def valid_training_data():
    df = load_data(Path("data/valid_test_data/"))
    processed_df = preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_datasets(
        processed_df, target_col="price"
    )
    return X_train, X_test, y_train, y_test


@pytest.mark.describe("Linear regression model function tests")
class TestLinearRegFunction:

    @pytest.mark.it("Inputs are not mutated")
    def test_inputs_not_mutated(self, valid_training_data):
        X_train, _, y_train, _ = valid_training_data
        copy_X_train = X_train.copy(deep=True)
        copy_y_train = y_train.copy(deep=True)
        linear_reg_model(X_train, y_train)
        pd.testing.assert_frame_equal(copy_X_train, X_train)
        pd.testing.assert_frame_equal(copy_y_train, y_train)

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

    @pytest.mark.it("Raises TypeError if inputs are not dataframes")
    def test_typeerror_input_not_a_dataframe(self):
        with pytest.raises(TypeError) as excinfo:
            linear_reg_model("not a dataframe", "not a dataframe")
        assert "Inputs must both be a pandas dataframe" in str(excinfo.value)

    @pytest.mark.it("Raises TypeError if an input contains non-numeric values")
    def test_typeerror_non_numeric(self):
        df = load_data(Path("data/valid_test_data/"))
        y = df[["price"]]
        X = df.drop(columns="price", axis=1)
        invalid_X_train, _, invalid_y_train, _ = train_test_split(X, y, test_size=0.2)
        with pytest.raises(TypeError) as excinfo:
            linear_reg_model(invalid_X_train, invalid_y_train)
        assert "Inputs must not contain non-numeric values" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if input lengths do not match")
    def test_input_lengths_match(self, valid_training_data):
        X_train, _, y_train, _ = valid_training_data
        shortened_y_train = y_train.head()
        with pytest.raises(ValueError) as excinfo:
            linear_reg_model(X_train, shortened_y_train)
        assert "X_train and y_train lengths must match" in str(excinfo.value)
