import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from pathlib import Path
from copy import deepcopy
from unittest.mock import MagicMock

from src.linear_model import (
    linear_preprocessing,
    linear_reg_model,
    evaluate_linear_model,
)
from src.data_prep import (
    load_data,
    split_datasets,
    scale_num_data,
)


@pytest.fixture(scope="function")
def preproc_test_data():
    cleansed_df = load_data(Path("data/valid_test_data/"))
    split_data = split_datasets(cleansed_df, target_col="price")
    X_num_train_scaled, X_num_test_scaled, _ = scale_num_data(
        split_data["train"]["X_num"], split_data["test"]["X_num"]
    )
    return {
        "X_num_train_scaled": X_num_train_scaled,
        "X_num_test_scaled": X_num_test_scaled,
        "X_cat_train": split_data["train"]["X_cat"],
        "X_cat_test": split_data["test"]["X_cat"],
        "y_train": split_data["train"]["y"],
        "y_test": split_data["test"]["y"],
    }


@pytest.fixture(scope="function")
def model_data(preproc_test_data):
    y_train = preproc_test_data.pop("y_train")
    y_test = preproc_test_data.pop("y_test")
    X_train, X_test, _ = linear_preprocessing(**preproc_test_data)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture(scope="function")
def sample_model(model_data):
    return linear_reg_model(model_data["X_train"], model_data["y_train"])


@pytest.mark.describe("Linear Preprocessing function tests")
class TestLinearPreprocessing:

    @pytest.mark.it("Inputs are not mutated")
    def test_inputs_not_mutated(self, preproc_test_data):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        copy_X_num_train_scaled = preproc_test_data["X_num_train_scaled"].copy(
            deep=True
        )
        copy_X_num_test_scaled = preproc_test_data["X_num_test_scaled"].copy(deep=True)
        copy_X_cat_train = preproc_test_data["X_cat_train"].copy(deep=True)
        copy_X_cat_test = preproc_test_data["X_cat_test"].copy(deep=True)
        linear_preprocessing(**preproc_test_data)
        pd.testing.assert_frame_equal(
            copy_X_num_train_scaled, preproc_test_data["X_num_train_scaled"]
        )
        pd.testing.assert_frame_equal(
            copy_X_num_test_scaled, preproc_test_data["X_num_test_scaled"]
        )
        pd.testing.assert_frame_equal(
            copy_X_cat_train, preproc_test_data["X_cat_train"]
        )
        pd.testing.assert_frame_equal(copy_X_cat_test, preproc_test_data["X_cat_test"])

    @pytest.mark.it("Returns expected output structure")
    def test_output_structure(self, preproc_test_data):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        X_train, X_test, ohe = linear_preprocessing(**preproc_test_data)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(ohe, OneHotEncoder)
        assert X_train.columns.equals(X_test.columns)

    @pytest.mark.it("One-hot encodes categories")
    def test_ohe_cat(self):
        sample_X_num = pd.DataFrame({"age": [0.1, 0.2, 0.3]}, index=[0, 1, 2])
        sample_X_cat = pd.DataFrame(
            {"colour": ["Red", "Blue", "Green"]}, index=[0, 1, 2]
        ).astype("string")
        X_train_final, _, _ = linear_preprocessing(
            sample_X_num, sample_X_num, sample_X_cat, sample_X_cat
        )
        # Expect 3 columns total as using drop='first' to simplify encoding so "Blue" would be when both "Red" and "Green" are 1
        assert X_train_final.shape[1] == 3
        expected_cols = ["age", "colour_Red", "colour_Green"]
        assert all(col in X_train_final.columns for col in expected_cols)
        assert "colour_Blue" not in X_train_final.columns

    @pytest.mark.it("X_cat_test has same encoding schema as X_cat_train")
    def test_consistent_encoding_schema(self):
        sample_X_num_train = pd.DataFrame({"m": [0.1, 0.2]}, index=[0, 1])
        sample_X_cat_train = pd.DataFrame(
            {"brand": ["Ford", "BMW"]}, index=[0, 1]
        ).astype("string")
        sample_X_num_test = pd.DataFrame({"m": [0.3]}, index=[0])
        sample_X_cat_test = pd.DataFrame({"brand": ["BMW"]}, index=[0]).astype("string")
        X_train, X_test, _ = linear_preprocessing(
            sample_X_num_train, sample_X_num_test, sample_X_cat_train, sample_X_cat_test
        )
        assert list(X_train.columns) == list(X_test.columns)

    @pytest.mark.it("Excludes categorical features with a single value")
    def test_exclude_single_val_cat_cols(self):
        sample_X_num = pd.DataFrame({"mileage": [100000, 20000]}, index=[0, 1])
        sample_X_cat = pd.DataFrame({"brand": ["Ford", "Ford"]}, index=[0, 1]).astype(
            "string"
        )
        X_train, _, _ = linear_preprocessing(
            sample_X_num, sample_X_num, sample_X_cat, sample_X_cat
        )
        assert list(X_train.columns) == ["mileage"]
        assert "brand_Ford" not in X_train.columns

    @pytest.mark.it("Skips OHE if no categorical columns with variance")
    def test_skip_ohe_no_cat_col_var(self):
        sample_X_num = pd.DataFrame({"mileage": [0.1, 0.2]}, index=[0, 1])
        sample_X_cat = pd.DataFrame({"brand": ["Ford", "Ford"]}, index=[0, 1]).astype(
            "string"
        )
        X_train, _, ohe = linear_preprocessing(
            sample_X_num, sample_X_num, sample_X_cat, sample_X_cat
        )
        assert X_train.equals(sample_X_num)
        assert ohe is None

    @pytest.mark.it("Only encodes categorical columns with variance")
    def test_only_encode_cat_with_var(self):
        sample_X_num = pd.DataFrame({"mileage": [10000, 20000]}, index=[0, 1])
        sample_X_cat = pd.DataFrame(
            {
                "brand": ["Ford", "Ford"],
                "transmission": ["Manual", "Automatic"],
            },
            index=[0, 1],
        ).astype("string")
        X_train, _, ohe = linear_preprocessing(
            sample_X_num, sample_X_num, sample_X_cat, sample_X_cat
        )
        assert "transmission_Manual" in X_train.columns
        assert "brand" not in X_train.columns
        assert "brand" not in ohe.feature_names_in_


@pytest.mark.describe("Linear Preprocessing exception handling")
class TestLinearPreprocessingExceptions:

    @pytest.mark.parametrize(
        "preproc_params",
        [
            "X_num_train_scaled",
            "X_num_test_scaled",
            "X_cat_train",
            "X_cat_test",
        ],
    )
    @pytest.mark.it("Raises TypeError if input is not a DataFrame")
    def test_input_not_dataframe(self, preproc_test_data, preproc_params):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        model_data = deepcopy(preproc_test_data)
        model_data[preproc_params] = "not a DataFrame"
        with pytest.raises(TypeError) as excinfo:
            linear_preprocessing(**model_data)
        assert "must be a pandas DataFrame" in str(excinfo.value)

    @pytest.mark.parametrize(
        "num_params",
        [
            "X_num_train_scaled",
            "X_num_test_scaled",
        ],
    )
    @pytest.mark.it("Raises TypeError for non-numeric values in numerical features")
    def test_non_numeric_in_num_features(self, preproc_test_data, num_params):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        model_data = deepcopy(preproc_test_data)
        model_data[num_params]["year"] = model_data[num_params]["year"].astype("string")
        with pytest.raises(TypeError, match="must not contain non-numeric values"):
            linear_preprocessing(**model_data)

    @pytest.mark.parametrize(
        "cat_params",
        [
            "X_cat_train",
            "X_cat_test",
        ],
    )
    @pytest.mark.it("Raises TypeError for numeric values in categorical features")
    def test_x_cat_train_numeric(self, preproc_test_data, cat_params):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        model_data = deepcopy(preproc_test_data)
        model_data[cat_params]["year"] = model_data["X_num_train_scaled"]["year"]
        with pytest.raises(TypeError, match="must not contain numeric values"):
            linear_preprocessing(**model_data)

    @pytest.mark.parametrize(
        "preproc_params",
        [
            "X_num_train_scaled",
            "X_num_test_scaled",
            "X_cat_train",
            "X_cat_test",
        ],
    )
    @pytest.mark.it("Raises ValueError if input is an empty DataFrame")
    def test_input_empty_dataframe(self, preproc_test_data, preproc_params):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        model_data = deepcopy(preproc_test_data)
        model_data[preproc_params] = pd.DataFrame()
        with pytest.raises(ValueError, match="is an empty DataFrame"):
            linear_preprocessing(**model_data)

    @pytest.mark.parametrize(
        "num_params",
        [
            "X_num_train_scaled",
            "X_num_test_scaled",
        ],
    )
    @pytest.mark.it("Raises ValueError for invalid values in numeric features")
    def test_invalid_values_num_features(self, preproc_test_data, num_params):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        model_data = deepcopy(preproc_test_data)
        model_data[num_params].iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="contains invalid values"):
            linear_preprocessing(**model_data)

    @pytest.mark.parametrize(
        "cat_params",
        [
            "X_cat_train",
            "X_cat_test",
        ],
    )
    @pytest.mark.it("Raises ValueError for invalid values in categorical features")
    def test_invalid_values_cat_features(self, preproc_test_data, cat_params):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        model_data = deepcopy(preproc_test_data)
        model_data[cat_params].iloc[0, 0] = " "
        with pytest.raises(ValueError, match="contains empty strings or whitespace"):
            linear_preprocessing(**model_data)

    @pytest.mark.parametrize(
        "cat_params",
        [
            "X_cat_train",
            "X_cat_test",
        ],
    )
    @pytest.mark.it("Raises TypeError for object dtype in categorical features")
    def test_input_contains_object_dtype(self, preproc_test_data, cat_params):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        model_data = deepcopy(preproc_test_data)
        model_data[cat_params]["model"] = model_data[cat_params]["model"].astype(object)
        with pytest.raises(
            TypeError,
            match="contains generic 'object' dtypes: cast to an explicit type first",
        ):
            linear_preprocessing(**model_data)

    @pytest.mark.it("Raises ValueError for mismatched columns in numerical data")
    def test_mismatched_num_columns(self, preproc_test_data):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        model_data = deepcopy(preproc_test_data)
        model_data["X_num_train_scaled"].drop(
            model_data["X_num_train_scaled"].columns[0], axis=1, inplace=True
        )
        with pytest.raises(ValueError, match="columns do not match"):
            linear_preprocessing(**model_data)

    @pytest.mark.it("Raises ValueError for mismatched columns in categorical data")
    def test_mismatched_cat_columns(self, preproc_test_data):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        model_data = deepcopy(preproc_test_data)
        model_data["X_cat_train"].drop(
            model_data["X_cat_train"].columns[0], axis=1, inplace=True
        )
        with pytest.raises(ValueError, match="columns do not match"):
            linear_preprocessing(**model_data)

    @pytest.mark.it("Raises ValueError for mismatched training data indices")
    def test_mismatched_train_indices(self, preproc_test_data):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        model_data = deepcopy(preproc_test_data)
        model_data["X_num_train_scaled"].rename(index={0: 50000}, inplace=True)
        with pytest.raises(ValueError, match="indices do not match"):
            linear_preprocessing(**model_data)

    @pytest.mark.it("Raises ValueError for mismatched testing data indices")
    def test_mismatched_test_indices(self, preproc_test_data):
        preproc_test_data.pop("y_train")
        preproc_test_data.pop("y_test")
        model_data = deepcopy(preproc_test_data)
        model_data["X_num_test_scaled"].rename(index={0: 50000}, inplace=True)
        with pytest.raises(ValueError, match="indices do not match"):
            linear_preprocessing(**model_data)


@pytest.mark.describe("Linear regression model function tests")
class TestLinearRegFunction:

    @pytest.mark.it("Inputs are not mutated")
    def test_inputs_not_mutated(self, model_data):
        copy_X_train = model_data["X_train"].copy(deep=True)
        copy_y_train = model_data["y_train"].copy(deep=True)
        linear_reg_model(model_data["X_train"], model_data["y_train"])
        pd.testing.assert_frame_equal(copy_X_train, model_data["X_train"])
        pd.testing.assert_series_equal(copy_y_train, model_data["y_train"])

    @pytest.mark.it("Returns a linear regression model")
    def test_returns_linear_regression_model(self, model_data):
        output = linear_reg_model(model_data["X_train"], model_data["y_train"])
        assert isinstance(output, LinearRegression)

    @pytest.mark.it("Returns a trained linear regression model")
    def test_returns_trained_model(self, model_data):
        output = linear_reg_model(model_data["X_train"], model_data["y_train"])
        assert hasattr(output, "coef_")

    @pytest.mark.it("Trained model produces predictions with expected shape")
    def test_trained_model_predictions_shape(self, model_data):
        output = linear_reg_model(model_data["X_train"], model_data["y_train"])
        preds = output.predict(model_data["X_test"])
        assert len(preds) == len(model_data["y_test"])
        assert len(preds) == len(model_data["X_test"])


@pytest.mark.describe("Linear regression model exception handling")
class TestLinearRegExceptions:

    @pytest.mark.it("Raises TypeError if features data is not DataFrames")
    def test_features_not_a_dataframe(self):
        test_y_train = pd.Series()
        with pytest.raises(TypeError) as excinfo:
            linear_reg_model("not a DataFrame", test_y_train)
        assert "X_train must be a pandas DataFrame" in str(excinfo.value)

    @pytest.mark.it("Raises TypeError if target data is not a series")
    def test_target_not_a_series(self):
        test_X_train = pd.DataFrame()
        with pytest.raises(TypeError) as excinfo:
            linear_reg_model(test_X_train, "not a series")
        assert "y_train must be a pandas Series" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if features data contains non-numeric values")
    def test_non_numeric_in_features(self):
        invalid_X_train = pd.DataFrame({"mileage": ["10000", "20000", "30000"]})
        y_train = pd.Series([10, 20, 30])
        with pytest.raises(
            ValueError, match="X_train must only contain numeric values"
        ):
            linear_reg_model(invalid_X_train, y_train)

    @pytest.mark.it("Raises ValueError if target is non-numeric")
    def test_non_numeric_target(self):
        X_train = pd.DataFrame({"mileage": [10000, 20000, 30000]})
        invalid_y_train = pd.Series(["10", "20", "30"])
        with pytest.raises(
            ValueError, match="y_train must only contain numeric values"
        ):
            linear_reg_model(X_train, invalid_y_train)

    @pytest.mark.it("Raises ValueError if X_train contains missing values")
    def test_x_missing_values(self):
        invalid_X_train = pd.DataFrame({"mileage": [10000, np.nan, 30000]})
        y_train = pd.Series([10, 20, 30])
        with pytest.raises(ValueError, match="X_train contains missing values"):
            linear_reg_model(invalid_X_train, y_train)

    @pytest.mark.it("Raises ValueError if y_train contains missing values")
    def test_y_missing_values(self):
        X_train = pd.DataFrame({"mileage": [10000, 20000, 30000]})
        invalid_y_train = pd.Series([10, np.nan, 30])
        with pytest.raises(ValueError, match="y_train contains missing values"):
            linear_reg_model(X_train, invalid_y_train)

    @pytest.mark.it("Raises ValueError for mismatched input lengths")
    def test_mismatched_input_lengths(self, model_data):
        X_train = model_data["X_train"]
        shortened_y_train = model_data["y_train"].iloc[:-1]
        with pytest.raises(
            ValueError, match="contain differing number of observations"
        ):
            linear_reg_model(X_train, shortened_y_train)


@pytest.mark.describe("Evaluate Linear Model function tests")
class TestEvaluateLinearModel:

    @pytest.mark.it("Inputs not mutated")
    def test_inputs_not_mutated(self, sample_model, model_data):
        copy_X_test = model_data["X_test"].copy(deep=True)
        copy_y_test = model_data["y_test"].copy(deep=True)
        copy_model_coef = sample_model.coef_.copy()
        copy_model_intercept = sample_model.intercept_
        evaluate_linear_model(sample_model, model_data["X_test"], model_data["y_test"])
        pd.testing.assert_frame_equal(model_data["X_test"], copy_X_test)
        pd.testing.assert_series_equal(model_data["y_test"], copy_y_test)
        assert np.allclose(sample_model.coef_, copy_model_coef)
        assert sample_model.intercept_ == copy_model_intercept

    @pytest.mark.it("Returns expected format")
    def test_returns_expected_format(self, sample_model, model_data):
        output = evaluate_linear_model(
            sample_model, model_data["X_test"], model_data["y_test"]
        )
        assert isinstance(output, dict)
        assert all(
            isinstance(key, str) and isinstance(value, float)
            for key, value in output.items()
        )

    @pytest.mark.it("Outputs expected metrics")
    def test_returns_expected_metrics(self, sample_model, model_data):
        expected_metrics = {"mse", "rmse", "mae", "r2"}
        output = evaluate_linear_model(
            sample_model, model_data["X_test"], model_data["y_test"]
        )
        assert all(key in output.keys() for key in expected_metrics)

    @pytest.mark.it("Predict only called once")
    def test_predict_called_once(self, model_data):
        model = MagicMock()
        model.predict.return_value = np.zeros(len(model_data["y_test"]))
        evaluate_linear_model(model, model_data["X_test"], model_data["y_test"])
        model.predict.assert_called_once_with(model_data["X_test"])

    @pytest.mark.it("Works with column vector predictions")
    def test_column_vector_predictions(self):
        model = MagicMock()
        model.predict.return_value = np.array([[1.0], [2.0], [3.0]])
        dummy_X_test = pd.DataFrame({"x": [1, 2, 3]})
        dummy_y_test = pd.Series([1.0, 2.0, 3.0])
        output = evaluate_linear_model(model, dummy_X_test, dummy_y_test)
        assert output["rmse"] == 0.0


@pytest.mark.describe("Evaluate Linear Model exception handling")
class TestEvaluateLinearModelExceptions:

    @pytest.mark.it("Raises TypeError model does not have a predict method")
    def test_model_not_sklearn_linear(self, model_data):
        test_model = "not a model"
        with pytest.raises(TypeError) as excinfo:
            evaluate_linear_model(
                test_model, model_data["X_test"], model_data["y_test"]
            )
        assert "Model must be an sklearn regressor" in str(excinfo.value)

    @pytest.mark.it("Raises TypeError if X_test is not a pandas DataFrame")
    def test_xtest_not_a_dataframe(self, sample_model, model_data):
        invalid_X_test = "not a dataframe"
        with pytest.raises(TypeError) as excinfo:
            evaluate_linear_model(sample_model, invalid_X_test, model_data["y_test"])
        assert "X_test must be a pandas DataFrame" in str(excinfo.value)

    @pytest.mark.it(
        "Raises TypeError if y_test is not a pandas Series or 1D numpy array"
    )
    def test_ytest_not_1d_array(self, sample_model, model_data):
        invalid_y_test = "not array-like"
        with pytest.raises(TypeError) as excinfo:
            evaluate_linear_model(sample_model, model_data["X_test"], invalid_y_test)
        assert "y_test must be a pandas Series or 1D numpy array" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if X_test and y_test lengths differ")
    def test_differing_test_data_lengths(self, sample_model, model_data):
        shortened_X_test = model_data["X_test"].head(5)
        with pytest.raises(ValueError) as excinfo:
            evaluate_linear_model(sample_model, shortened_X_test, model_data["y_test"])
        assert "X_test and y_test must have the same number of observations" in str(
            excinfo.value
        )

    @pytest.mark.it("Raises ValueError if y_test is not one dimensional")
    def test_ytest_not_1D(self, sample_model):
        valid_X_test = pd.DataFrame({"x": [1, 2, 3]})
        invalid_y_test = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError) as excinfo:
            evaluate_linear_model(sample_model, valid_X_test, invalid_y_test)
        assert "y_test must be 1-dimensional" in str(excinfo.value)
