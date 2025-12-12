import pytest, torch
import pandas as pd
import torch.nn as nn
from pathlib import Path
from src.nn_model import create_nn_model, train_nn_model
from src.utils import preprocessing, train_test_datasets, tensor_converter
from src.data_loader import load_data


@pytest.fixture(scope="function")
def training_data():
    cleansed_df = load_data(Path("data/valid_test_data/"))
    preprocessed_df = preprocessing(cleansed_df)
    X_train, _, y_train, _ = train_test_datasets(preprocessed_df, target_col="price")
    X_tensor, y_tensor = tensor_converter(X_train, y_train)
    return X_tensor, y_tensor


@pytest.mark.describe("Create NN model function tests")
class TestCreateNNModel:

    @pytest.mark.it("Input not mutated")
    def test_input_not_mutated(self):
        test_input_dim = 10
        copy_test_input_dim = 10
        create_nn_model(test_input_dim)
        assert test_input_dim == copy_test_input_dim

    @pytest.mark.it("Returns a torch module")
    def test_returns_torch_module(self):
        model = create_nn_model(input_dim=10)
        assert isinstance(model, nn.Module)

    @pytest.mark.it("Output shape is correct")
    def test_returns_expected_output_shape(self):
        model = create_nn_model(input_dim=10)
        test_input = torch.randn(1, 10)
        output = model(test_input)
        assert output.shape == (1, 1)

    @pytest.mark.it("Model contains expected number of layers")
    def test_no_layers(self):
        model = create_nn_model(input_dim=10)
        layer_count = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layer_count += 1
        assert layer_count == 3


@pytest.mark.describe("Train NN model function tests")
class TestTrainNNModel:

    @pytest.mark.it("Training loop runs without error")
    def test_training_loop_runs(self, training_data):
        X_tensor, y_tensor = training_data
        model = create_nn_model(input_dim=X_tensor.shape[1])
        try:
            train_nn_model(model, X_tensor, y_tensor, epochs=3)
        except Exception as e:
            pytest.fail(f"Training loop failed: {e}")
