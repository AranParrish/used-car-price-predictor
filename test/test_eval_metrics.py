import numpy as np
import pandas as pd
import torch, pytest
from copy import deepcopy
from dataclasses import dataclass
from src.eval_metrics import evaluate_model


@dataclass(frozen=True)
class EvalInputs:
    y_true: np.ndarray
    y_pred: np.ndarray


@pytest.fixture(scope="function")
def example_targets():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 3, 2, 5])
    return EvalInputs(y_true=y_true, y_pred=y_pred)


@pytest.mark.describe("Evaluate Model function tests")
class TestEvaluateModel:

    @pytest.mark.it("Inputs are not mutated")
    def test_inputs_not_mutated(self, example_targets):
        copy_y_true = deepcopy(example_targets.y_true)
        copy_y_pred = deepcopy(example_targets.y_pred)
        evaluate_model(example_targets.y_true, example_targets.y_pred)
        np.testing.assert_array_equal(example_targets.y_true, copy_y_true)
        np.testing.assert_array_equal(example_targets.y_pred, copy_y_pred)

    @pytest.mark.it("Returns expected output format")
    def test_returns_expected_output_format(self, example_targets):
        output = evaluate_model(example_targets.y_true, example_targets.y_pred)
        assert isinstance(output, dict)
        assert all(isinstance(key, str) for key in output.keys())
        assert all(isinstance(value, float) for value in output.values())

    @pytest.mark.it("Contains expected metrics")
    def test_outputs_expected_metrics(self, example_targets):
        expected_keys = {"mse", "rmse", "mae", "r2"}
        output = evaluate_model(example_targets.y_true, example_targets.y_pred)
        assert all(key in output.keys() for key in expected_keys)

    @pytest.mark.it("Results as expected")
    def test_expected_results(self, example_targets):
        expected_results = {
            "mse": 0.75,
            "rmse": 0.8660254,
            "mae": 0.75,
            "r2": 0.4,
        }
        output = evaluate_model(example_targets.y_true, example_targets.y_pred)
        assert all(
            output[key] == pytest.approx(value)
            for key, value in expected_results.items()
        )

    @pytest.mark.it("Can evaluate Torch Tensors")
    def test_evaluate_torch_tensors(self, example_targets):
        y_true_torch = torch.from_numpy(example_targets.y_true)
        y_pred_torch = torch.from_numpy(example_targets.y_pred)
        expected_keys = {"mse", "rmse", "mae", "r2"}
        output = evaluate_model(y_true_torch, y_pred_torch)
        assert all(key in output.keys() for key in expected_keys)

    @pytest.mark.it("Can evaluate Pandas Series")
    def test_evaluate_pandas_series(self, example_targets):
        y_true_series = pd.Series(example_targets.y_true)
        y_pred_series = pd.Series(example_targets.y_pred)
        expected_keys = {"mse", "rmse", "mae", "r2"}
        output = evaluate_model(y_true_series, y_pred_series)
        assert all(key in output.keys() for key in expected_keys)


@pytest.mark.describe("Evaluate Model exception handling")
class TestEvaluateExceptions:

    @pytest.mark.it("Raises TypeError for invalid input type")
    def test_invalid_input_type(self, example_targets):
        y_pred_invalid = [1, 2, 3, 4]
        with pytest.raises(TypeError) as excinfo:
            evaluate_model(example_targets.y_true, y_pred_invalid)
        assert "Inputs must be a NumPy Array, Pandas Series, or Torch Tensor" in str(
            excinfo.value
        )

    @pytest.mark.it("Raises ValueError for length mismatch")
    def test_length_mismatch(self, example_targets):
        short_y_pred = np.array([1, 3, 2])
        with pytest.raises(ValueError) as excinfo:
            evaluate_model(example_targets.y_true, short_y_pred)
        assert "Inputs must have the same shape" in str(excinfo.value)
