import pytest, warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
from copy import deepcopy
from unittest.mock import MagicMock

from src.linear_model import (
    linear_preprocessing,
    # linear_reg_model,
    # evaluate_linear_model,
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
    }


# @pytest.fixture(scope="function")
# def linear_processed_df(cleansed_df):
#     return linear_preprocessing(cleansed_df)


# @pytest.fixture(scope="function")
# def sample_model(sample_data):
#     return linear_reg_model(sample_data.X_train, sample_data.y_train)


@pytest.mark.describe("Linear Preprocessing function tests")
class TestLinearPreprocessing:

    @pytest.mark.it("Inputs are not mutated")
    def test_inputs_not_mutated(self, preproc_test_data):
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
        sample_data = deepcopy(preproc_test_data)
        sample_data[preproc_params] = "not a DataFrame"
        with pytest.raises(TypeError) as excinfo:
            linear_preprocessing(**sample_data)
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
        sample_data = deepcopy(preproc_test_data)
        sample_data[num_params]["year"] = sample_data[num_params]["year"].astype(
            "string"
        )
        with pytest.raises(TypeError, match="must not contain non-numeric values"):
            linear_preprocessing(**sample_data)

    @pytest.mark.parametrize(
        "cat_params",
        [
            "X_cat_train",
            "X_cat_test",
        ],
    )
    @pytest.mark.it("Raises TypeError for numeric values in categorical features")
    def test_x_cat_train_numeric(self, preproc_test_data, cat_params):
        sample_data = deepcopy(preproc_test_data)
        sample_data[cat_params]["year"] = sample_data["X_num_train_scaled"]["year"]
        with pytest.raises(TypeError, match="must not contain numeric values"):
            linear_preprocessing(**sample_data)

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
        sample_data = deepcopy(preproc_test_data)
        sample_data[preproc_params] = pd.DataFrame()
        with pytest.raises(ValueError, match="is an empty DataFrame"):
            linear_preprocessing(**sample_data)

    @pytest.mark.parametrize(
        "num_params",
        [
            "X_num_train_scaled",
            "X_num_test_scaled",
        ],
    )
    @pytest.mark.it("Raises ValueError for invalid values in numeric features")
    def test_invalid_values_num_features(self, preproc_test_data, num_params):
        sample_data = deepcopy(preproc_test_data)
        sample_data[num_params].iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="contains invalid values"):
            linear_preprocessing(**sample_data)

    @pytest.mark.parametrize(
        "cat_params",
        [
            "X_cat_train",
            "X_cat_test",
        ],
    )
    @pytest.mark.it("Raises ValueError for invalid values in categorical features")
    def test_invalid_values_cat_features(self, preproc_test_data, cat_params):
        sample_data = deepcopy(preproc_test_data)
        sample_data[cat_params].iloc[0, 0] = " "
        with pytest.raises(ValueError, match="contains empty strings or whitespace"):
            linear_preprocessing(**sample_data)

    @pytest.mark.parametrize(
        "cat_params",
        [
            "X_cat_train",
            "X_cat_test",
        ],
    )
    @pytest.mark.it("Raises TypeError for object dtype in categorical features")
    def test_input_contains_object_dtype(self, preproc_test_data, cat_params):
        sample_data = deepcopy(preproc_test_data)
        sample_data[cat_params]["model"] = sample_data[cat_params]["model"].astype(
            object
        )
        with pytest.raises(
            TypeError,
            match="contains generic 'object' dtypes: cast to an explicit type first",
        ):
            linear_preprocessing(**sample_data)

    @pytest.mark.it("Raises ValueError for mismatched columns in numerical data")
    def test_mismatched_num_columns(self, preproc_test_data):
        sample_data = deepcopy(preproc_test_data)
        sample_data["X_num_train_scaled"].drop(
            sample_data["X_num_train_scaled"].columns[0], axis=1, inplace=True
        )
        with pytest.raises(ValueError, match="columns do not match"):
            linear_preprocessing(**sample_data)

    @pytest.mark.it("Raises ValueError for mismatched columns in categorical data")
    def test_mismatched_cat_columns(self, preproc_test_data):
        sample_data = deepcopy(preproc_test_data)
        sample_data["X_cat_train"].drop(
            sample_data["X_cat_train"].columns[0], axis=1, inplace=True
        )
        with pytest.raises(ValueError, match="columns do not match"):
            linear_preprocessing(**sample_data)

    @pytest.mark.it("Raises ValueError for mismatched training data indices")
    def test_mismatched_train_indices(self, preproc_test_data):
        sample_data = deepcopy(preproc_test_data)
        sample_data["X_num_train_scaled"].rename(index={0: 50000}, inplace=True)
        with pytest.raises(ValueError, match="indices do not match"):
            linear_preprocessing(**sample_data)

    @pytest.mark.it("Raises ValueError for mismatched testing data indices")
    def test_mismatched_test_indices(self, preproc_test_data):
        sample_data = deepcopy(preproc_test_data)
        sample_data["X_num_test_scaled"].rename(index={0: 50000}, inplace=True)
        with pytest.raises(ValueError, match="indices do not match"):
            linear_preprocessing(**sample_data)


# @pytest.mark.describe("Linear Preprocessing function tests")
# class TestLinearPreprocessing:

#     @pytest.mark.it("Input is not mutated")
#     def test_input_not_mutated(self, cleansed_df):
#         copy_df = cleansed_df.copy(deep=True)
#         linear_preprocessing(cleansed_df)
#         pd.testing.assert_frame_equal(cleansed_df, copy_df)

#     @pytest.mark.it("Returns a new DataFrame")
#     def test_returns_new_dataframe(self, cleansed_df):
#         output = linear_preprocessing(cleansed_df)
#         assert isinstance(output, pd.DataFrame)
#         assert output is not cleansed_df

#     @pytest.mark.it("Removes categorical columns")
#     def test_removes_categorical_columns(self, cleansed_df):
#         output = linear_preprocessing(cleansed_df)
#         assert output.select_dtypes(include=["object", "string", "boolean"]).empty

#     @pytest.mark.it("Dataset with no categorical columns returned unchanged")
#     def test_no_categorical_returned(self, cleansed_df):
#         numeric_only = cleansed_df[["price"]]
#         output = linear_preprocessing(numeric_only)
#         pd.testing.assert_frame_equal(output, numeric_only)


# @pytest.mark.describe("Linear Preprocessing exception handling")
# class TestLinearPreprocessingExceptions:

#     @pytest.mark.it("Raises TypeError if input data is not a DataFrame")
#     def test_typeerror_not_a_dataframe(self):
#         with pytest.raises(TypeError) as excinfo:
#             linear_preprocessing("not a DataFrame")
#         assert "Input must be a pandas DataFrame" in str(excinfo.value)

#     @pytest.mark.it("Raises ValueError if input data contains invalid rows")
#     def test_valueerror_invalid_rows(self):
#         invalid_data = Path("data/invalid_test_data/ford.csv")
#         df = pd.read_csv(invalid_data)
#         df["brand"] = "Ford"
#         with pytest.raises(ValueError) as excinfo:
#             linear_preprocessing(df)
#         assert "Input data contains invalid rows" in str(excinfo.value)


# @pytest.mark.describe("Linear Train / Test function tests")
# class TestLinearTrainTestSplit:

#     @pytest.mark.it("Input is not mutated")
#     def test_input_not_mutated(self, linear_processed_df):
#         copy_df = linear_processed_df.copy(deep=True)
#         linear_train_test_datasets(
#             linear_processed_df, target_col="price", test_size=0.2, random_seed=42
#         )
#         pd.testing.assert_frame_equal(linear_processed_df, copy_df)

#     @pytest.mark.it("Returns expected output structure")
#     def test_output_structure(self, linear_processed_df):
#         outputs = linear_train_test_datasets(linear_processed_df, target_col="price")
#         assert isinstance(outputs, list)
#         assert len(outputs) == 4
#         X_train, X_test, y_train, y_test = outputs
#         assert all(
#             isinstance(features_data, pd.DataFrame)
#             for features_data in (X_train, X_test)
#         )
#         assert all(
#             isinstance(target_data, pd.Series) for target_data in (y_train, y_test)
#         )

#     @pytest.mark.it("Returns expected train and test sample sizes")
#     def test_train_test_sizes(self, linear_processed_df):
#         X_train, X_test, y_train, y_test = linear_train_test_datasets(
#             linear_processed_df, target_col="price", test_size=0.2
#         )
#         expected_train_size = len(linear_processed_df) * 0.8
#         expected_test_size = len(linear_processed_df) * 0.2
#         assert len(X_train) == expected_train_size
#         assert len(y_train) == expected_train_size
#         assert len(X_test) == expected_test_size
#         assert len(y_test) == expected_test_size


# @pytest.mark.describe("Linear Train / Test exception handling")
# class TestLinearTrainTestExceptions:

#     @pytest.mark.it("Raises TypeError if input is not a DataFrame")
#     def test_input_not_a_dataframe(self):
#         invalid_input = []
#         with pytest.raises(TypeError) as excinfo:
#             linear_train_test_datasets(invalid_input, target_col="price")
#         assert "Input dataset must be a pandas DataFrame" in str(excinfo.value)

#     @pytest.mark.it("Raises ValueError if input DataFrame contains non-numeric columns")
#     def test_input_non_numeric_cols(self, cleansed_df):
#         with pytest.raises(ValueError) as excinfo:
#             linear_train_test_datasets(cleansed_df, target_col="price")
#         assert "Input DataFrame must not contain non-numeric columns" in str(
#             excinfo.value
#         )

#     @pytest.mark.it("Raises ValueError if target col does not exist")
#     def test_target_col_does_not_exist(self, linear_processed_df):
#         with pytest.raises(ValueError) as excinfo:
#             linear_train_test_datasets(linear_processed_df, target_col="invalid")
#         assert "Target column not in input dataset" in str(excinfo.value)

#     @pytest.mark.it(
#         "Raises ValueError if DataFrame does not contain at least one feature and one target"
#     )
#     def test_df_without_features(self, linear_processed_df):
#         invalid_df = linear_processed_df[["price"]]
#         with pytest.raises(ValueError) as excinfo:
#             linear_train_test_datasets(invalid_df, target_col="price")
#         assert (
#             "DataFrame must contain at least one feature column and one target column"
#             in str(excinfo.value)
#         )


# @pytest.mark.describe("Linear regression model function tests")
# class TestLinearRegFunction:

#     @pytest.mark.it("Inputs are not mutated")
#     def test_inputs_not_mutated(self, sample_data):
#         copy_X_train = sample_data.X_train.copy(deep=True)
#         copy_y_train = sample_data.y_train.copy(deep=True)
#         linear_reg_model(sample_data.X_train, sample_data.y_train)
#         pd.testing.assert_frame_equal(copy_X_train, sample_data.X_train)
#         pd.testing.assert_series_equal(copy_y_train, sample_data.y_train)

#     @pytest.mark.it("Returns a linear regression model")
#     def test_returns_linear_regression_model(self, sample_data):
#         output = linear_reg_model(sample_data.X_train, sample_data.y_train)
#         assert isinstance(output, LinearRegression)

#     @pytest.mark.it("Returns a trained linear regression model")
#     def test_returns_trained_model(self, sample_data):
#         output = linear_reg_model(sample_data.X_train, sample_data.y_train)
#         assert hasattr(output, "coef_")

#     @pytest.mark.it("Trained model produces predictions with expected shape")
#     def test_trained_model_predictions_shape(self, sample_data):
#         output = linear_reg_model(sample_data.X_train, sample_data.y_train)
#         preds = output.predict(sample_data.X_test)
#         assert preds.shape == sample_data.y_test.shape
#         assert len(preds) == len(sample_data.X_test)


# @pytest.mark.describe("Linear regression model exception handling")
# class TestLinearRegExceptions:

#     @pytest.mark.it("Raises TypeError if features data is not DataFrames")
#     def test_typeerror_features_not_a_dataframe(self):
#         test_y_train = pd.Series()
#         with pytest.raises(TypeError) as excinfo:
#             linear_reg_model("not a DataFrame", test_y_train)
#         assert "X_train must be a pandas DataFrame" in str(excinfo.value)

#     @pytest.mark.it("Raises TypeError if target data is not a series")
#     def test_typeerror_target_not_a_series(self):
#         test_X_train = pd.DataFrame()
#         with pytest.raises(TypeError) as excinfo:
#             linear_reg_model(test_X_train, "not a series")
#         assert "y_train must be a pandas Series" in str(excinfo.value)

#     @pytest.mark.it("Raises ValueError if an features data contains non-numeric values")
#     def test_valueerror_non_numeric_in_features(self):
#         df = load_data(Path("data/valid_test_data/"))
#         y = df["price"]
#         X = df.drop(columns="price")
#         invalid_X_train, _, invalid_y_train, _ = train_test_split(X, y, test_size=0.2)
#         with pytest.raises(ValueError) as excinfo:
#             linear_reg_model(invalid_X_train, invalid_y_train)
#         assert "X_train must only contain numeric values" in str(excinfo.value)

#     @pytest.mark.it("Raises ValueError if target is non-numeric")
#     def test_valueerror_non_numeric_target(self, sample_data):
#         df = load_data(Path("data/valid_test_data/"))
#         invalid_y_train = df["model"]
#         with pytest.raises(ValueError) as excinfo:
#             linear_reg_model(sample_data.X_train, invalid_y_train)
#         assert "y_train must only contain numeric values" in str(excinfo.value)

#     @pytest.mark.it("Raises ValueError if data contains missing values")
#     def test_missing_values(self, sample_data):
#         invalid_data = Path("data/invalid_test_data/ford.csv")
#         df = pd.read_csv(invalid_data)
#         invalid_X_train = df[["year", "engineSize"]]
#         with pytest.raises(ValueError) as excinfo:
#             linear_reg_model(invalid_X_train, sample_data.y_train)
#         assert "Input data must not contain missing values" in str(excinfo.value)


# @pytest.mark.describe("Evaluate Linear Model function tests")
# class TestEvaluateLinearModel:

#     @pytest.mark.it("Inputs not mutated")
#     def test_inputs_not_mutated(self, sample_model, sample_data):
#         copy_X_test = sample_data.X_test.copy(deep=True)
#         copy_y_test = sample_data.y_test.copy(deep=True)
#         copy_model_coef = sample_model.coef_.copy()
#         copy_model_intercept = sample_model.intercept_
#         evaluate_linear_model(sample_model, sample_data.X_test, sample_data.y_test)
#         pd.testing.assert_frame_equal(sample_data.X_test, copy_X_test)
#         pd.testing.assert_series_equal(sample_data.y_test, copy_y_test)
#         assert np.allclose(sample_model.coef_, copy_model_coef)
#         assert sample_model.intercept_ == copy_model_intercept

#     @pytest.mark.it("Returns expected format")
#     def test_returns_expected_format(self, sample_model, sample_data):
#         output = evaluate_linear_model(
#             sample_model, sample_data.X_test, sample_data.y_test
#         )
#         assert isinstance(output, dict)
#         assert all(
#             isinstance(key, str) and isinstance(value, float)
#             for key, value in output.items()
#         )

#     @pytest.mark.it("Outputs expected metrics")
#     def test_returns_expected_metrics(self, sample_model, sample_data):
#         expected_metrics = {"mse", "rmse", "mae", "r2"}
#         output = evaluate_linear_model(
#             sample_model, sample_data.X_test, sample_data.y_test
#         )
#         assert all(key in output.keys() for key in expected_metrics)

#     @pytest.mark.it("Predict only called once")
#     def test_predict_called_once(self, sample_data):
#         model = MagicMock()
#         model.predict.return_value = np.zeros(len(sample_data.y_test))
#         evaluate_linear_model(model, sample_data.X_test, sample_data.y_test)
#         model.predict.assert_called_once_with(sample_data.X_test)

#     @pytest.mark.it("Works with column vector predictions")
#     def test_column_vector_predictions(self, sample_data):
#         class DummyModel:
#             def predict(self, X):
#                 return np.array([[1.0], [2.0], [3.0]])

#         model = DummyModel()
#         dummy_X_test = pd.DataFrame({"x": [1, 2, 3]})
#         dummy_y_test = pd.Series([1.0, 2.0, 3.0])
#         output = evaluate_linear_model(model, dummy_X_test, dummy_y_test)
#         assert output["rmse"] == 0.0


# @pytest.mark.describe("Evaluate Linear Model exception handling")
# class TestEvaluateLinearModelExceptions:

#     @pytest.mark.it("Raises TypeError model does not have a predict method")
#     def test_model_not_sklearn_linear(self, sample_data):
#         test_model = "not a model"
#         with pytest.raises(TypeError) as excinfo:
#             evaluate_linear_model(test_model, sample_data.X_test, sample_data.y_test)
#         assert "Model must be an sklearn regressor" in str(excinfo.value)

#     @pytest.mark.it("Raises TypeError if X_test is not a pandas DataFrame")
#     def test_xtest_not_a_dataframe(self, sample_model, sample_data):
#         invalid_X_test = "not a dataframe"
#         with pytest.raises(TypeError) as excinfo:
#             evaluate_linear_model(sample_model, invalid_X_test, sample_data.y_test)
#         assert "X_test must be a pandas DataFrame" in str(excinfo.value)

#     @pytest.mark.it(
#         "Raises TypeError if y_test is not a pandas Series or 1D numpy array"
#     )
#     def test_ytest_not_1d_array(self, sample_model, sample_data):
#         invalid_y_test = "not array-like"
#         with pytest.raises(TypeError) as excinfo:
#             evaluate_linear_model(sample_model, sample_data.X_test, invalid_y_test)
#         assert "y_test must be a pandas Series or 1D numpy array" in str(excinfo.value)

#     @pytest.mark.it("Raises ValueError if X_test and y_test lengths differ")
#     def test_differing_test_data_lengths(self, sample_model, sample_data):
#         shortened_X_test = sample_data.X_test.head(5)
#         with pytest.raises(ValueError) as excinfo:
#             evaluate_linear_model(sample_model, shortened_X_test, sample_data.y_test)
#         assert "X_test and y_test must have the same number of observations" in str(
#             excinfo.value
#         )

#     @pytest.mark.it("Raises ValueError if y_test is not one dimensional")
#     def test_ytest_not_1D(self, sample_model):
#         valid_X_test = pd.DataFrame({"x": [1, 2, 3]})
#         invalid_y_test = np.array([[1.0], [2.0], [3.0]])
#         with pytest.raises(ValueError) as excinfo:
#             evaluate_linear_model(sample_model, valid_X_test, invalid_y_test)
#         assert "y_test must be 1-dimensional" in str(excinfo.value)
