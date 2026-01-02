# import pytest, torch
# import pandas as pd
# import torch.nn as nn
# from pathlib import Path
# from src.nn_model import create_nn_model, train_nn_model
# from src.utils import linear_preprocessing, linear_train_test_datasets, tensor_converter
# from src.data_loader import load_data


# @pytest.fixture(scope="function")
# def training_data():
#     cleansed_df = load_data(Path("data/valid_test_data/"))
#     preprocessed_df = linear_preprocessing(cleansed_df)
#     X_train, _, y_train, _ = linear_train_test_datasets(
#         preprocessed_df, target_col="price"
#     )
#     X_train_tensor, y_train_tensor = tensor_converter(X_train, y_train)
#     return X_train_tensor, y_train_tensor


# @pytest.mark.describe("Create NN model function tests")
# class TestCreateNNModel:

#     @pytest.mark.it("Input not mutated")
#     def test_input_not_mutated(self):
#         test_input_dim = 10
#         copy_test_input_dim = 10
#         create_nn_model(test_input_dim)
#         assert test_input_dim == copy_test_input_dim

#     @pytest.mark.it("Returns a torch module")
#     def test_returns_torch_module(self):
#         model = create_nn_model(input_dim=10)
#         assert isinstance(model, nn.Module)

#     @pytest.mark.it("Output shape is correct")
#     def test_returns_expected_output_shape(self):
#         model = create_nn_model(input_dim=10)
#         test_input = torch.randn(1, 10)
#         output = model(test_input)
#         assert output.shape == (1, 1)

#     @pytest.mark.it("Model contains expected number of layers")
#     def test_no_layers(self):
#         model = create_nn_model(input_dim=10)
#         layer_count = 0
#         for module in model.modules():
#             if isinstance(module, nn.Linear):
#                 layer_count += 1
#         assert layer_count == 3


# @pytest.mark.describe("Train NN model function tests")
# class TestTrainNNModel:

#     @pytest.mark.it("Input tensors are not mutated")
#     def test_input_tensors_not_mutated(self, training_data):
#         X_train_tensor, y_train_tensor = training_data
#         copy_X_train_tensor = X_train_tensor.detach().clone()
#         copy_y_train_tensor = y_train_tensor.detach().clone()
#         model = create_nn_model(input_dim=X_train_tensor.shape[1])
#         train_nn_model(model, X_train_tensor, y_train_tensor)
#         assert torch.equal(X_train_tensor, copy_X_train_tensor)
#         assert torch.equal(y_train_tensor, copy_y_train_tensor)

#     @pytest.mark.it("Returns expected format of losses")
#     def test_return_experted_format(self, training_data):
#         X_train_tensor, y_train_tensor = training_data
#         model = create_nn_model(input_dim=X_train_tensor.shape[1])
#         losses_df = train_nn_model(model, X_train_tensor, y_train_tensor, epochs=2)
#         assert isinstance(losses_df, pd.DataFrame)
#         assert list(losses_df.columns) == ["epoch", "MSE"]
#         assert losses_df["MSE"].dtype == float

#     @pytest.mark.it("Training loop runs without error")
#     def test_training_loop_runs(self, training_data):
#         X_train_tensor, y_train_tensor = training_data
#         model = create_nn_model(input_dim=X_train_tensor.shape[1])
#         try:
#             train_nn_model(model, X_train_tensor, y_train_tensor, epochs=3)
#         except Exception as e:
#             pytest.fail(f"Training loop failed: {e}")

#     @pytest.mark.it("Loss decreases with training")
#     def test_training_decreases_loss(self, training_data):
#         X_train_tensor, y_train_tensor = training_data
#         model = create_nn_model(input_dim=X_train_tensor.shape[1])
#         losses_df = train_nn_model(model, X_train_tensor, y_train_tensor, epochs=2)
#         list_MSE = list(losses_df["MSE"].values)
#         assert list_MSE[1] < list_MSE[0]

#     @pytest.mark.it("Returns losses in expected interval")
#     def test_loss_interval(self, training_data):
#         X_train_tensor, y_train_tensor = training_data
#         model = create_nn_model(input_dim=X_train_tensor.shape[1])
#         losses_df = train_nn_model(model, X_train_tensor, y_train_tensor, epochs=20)
#         assert len(losses_df) == 20 / 2


# @pytest.mark.describe("Train NN model exception handling")
# class TestTrainNNModelExceptions:

#     @pytest.mark.it("Raises TypeError if model is not a PyTorch Module instance")
#     def test_model_not_pytorch_nn(self, training_data):
#         X_train_tensor, y_train_tensor = training_data
#         with pytest.raises(TypeError) as excinfo:
#             train_nn_model("not a PyTorch nn model", X_train_tensor, y_train_tensor)
#         assert "Input model must be a PyTorch neural network" in str(excinfo.value)

#     @pytest.mark.it("Raises TypeError if input data is not PyTorch Tensors")
#     def test_input_data_pytorch_tensors(self):
#         model = create_nn_model(input_dim=10)
#         with pytest.raises(TypeError) as excinfo:
#             train_nn_model(model, "not a Tensor", "not a Tensor")
#         assert "Input training datasets must both be a PyTorch Tensor" in str(
#             excinfo.value
#         )

#     @pytest.mark.it("Raises TyperError if epochs is not an integer")
#     def test_epochs_not_int(self, training_data):
#         X_train_tensor, y_train_tensor = training_data
#         model = create_nn_model(input_dim=X_train_tensor.shape[1])
#         with pytest.raises(TypeError) as excinfo:
#             train_nn_model(model, X_train_tensor, y_train_tensor, epochs="ten")
#         assert "Number of epochs must be an integer" in str(excinfo.value)

#     @pytest.mark.it("Raises TypeError if lr is not a float")
#     def test_lr_not_float(self, training_data):
#         X_train_tensor, y_train_tensor = training_data
#         model = create_nn_model(input_dim=X_train_tensor.shape[1])
#         with pytest.raises(TypeError) as excinfo:
#             train_nn_model(model, X_train_tensor, y_train_tensor, lr="1")
#         assert "Learning rate (lr) must be a float" in str(excinfo.value)

#     @pytest.mark.it("Raises ValueError for mismatched input tensors")
#     def test_mismatched_input_tensors(self, training_data):
#         X_train_tensor, y_train_tensor = training_data
#         model = create_nn_model(input_dim=X_train_tensor.shape[1])
#         X_train_tensor = X_train_tensor[:10]
#         with pytest.raises(ValueError) as excinfo:
#             train_nn_model(model, X_train_tensor, y_train_tensor)
#         assert "Number of rows in input tensors must match" in str(excinfo.value)

#     @pytest.mark.it("Raises ValueError if epochs is negative")
#     def test_negative_epochs(self, training_data):
#         X_train_tensor, y_train_tensor = training_data
#         model = create_nn_model(input_dim=X_train_tensor.shape[1])
#         with pytest.raises(ValueError) as excinfo:
#             train_nn_model(model, X_train_tensor, y_train_tensor, epochs=-10)
#         assert "Number of epochs cannot be negative" in str(excinfo.value)

#     @pytest.mark.it("Raises ValueError if lr is negative")
#     def test_negative_lr(self, training_data):
#         X_train_tensor, y_train_tensor = training_data
#         model = create_nn_model(input_dim=X_train_tensor.shape[1])
#         with pytest.raises(ValueError) as excinfo:
#             train_nn_model(model, X_train_tensor, y_train_tensor, lr=-0.001)
#         assert "Learning rate (lr) cannot be negative" in str(excinfo.value)
