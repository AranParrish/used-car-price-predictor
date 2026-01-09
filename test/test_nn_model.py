import pytest, torch
import pandas as pd
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from src.nn_model import EmbeddingNN
from src.utils import (
    embeddings_preprocessing,
    split_and_tensorise,
    fastai_embedding_dims,
    embedding_specs,
)
from src.data_loader import load_data


@pytest.fixture(scope="function")
def cleansed_df():
    return load_data(Path("data/valid_test_data/"))


@pytest.fixture(scope="function")
def preprocessed_data(cleansed_df):
    return embeddings_preprocessing(cleansed_df, target_col="price")


@pytest.fixture(scope="function")
def training_data(preprocessed_data):
    X_num, X_cat, y, _ = preprocessed_data
    X_num_train, _, X_cat_train, _, y_train, _ = split_and_tensorise(X_num, X_cat, y)
    return (X_num_train, X_cat_train, y_train)


@pytest.fixture(scope="function")
def example_embedding_specs(preprocessed_data):
    _, _, _, metadata = preprocessed_data
    mappings = metadata["mappings"]
    sample_embedding_dims = fastai_embedding_dims(mappings)
    sample_embedding_specs = embedding_specs(mappings, sample_embedding_dims)
    return sample_embedding_specs


@pytest.mark.describe("Embedding NN custom class tests")
class TestEmbeddingNN:

    @pytest.mark.it("Inputs not mutated")
    def test_inputs_not_mutated(self, example_embedding_specs):
        copy_embedding_specs = deepcopy(example_embedding_specs)
        test_num_numeric = 10
        copy_test_num_numeric = 10
        EmbeddingNN(test_num_numeric, example_embedding_specs)
        assert copy_embedding_specs == example_embedding_specs
        assert copy_test_num_numeric == test_num_numeric

    @pytest.mark.it("Returns a torch NN module")
    def test_returns_torch_nn_module(self, example_embedding_specs):
        num_numeric = 3
        model = EmbeddingNN(num_numeric, example_embedding_specs)
        assert isinstance(model, nn.Module)

    @pytest.mark.it("Model has one embedding layer per categorical feature")
    def test_one_embedding_layer_per_categorical_feature(self, example_embedding_specs):
        num_numeric = 3
        model = EmbeddingNN(num_numeric, example_embedding_specs)
        assert len(model.embeddings) == len(example_embedding_specs)

    @pytest.mark.it("Output shape is correct")
    def test_model_output_shape(self, training_data, example_embedding_specs):
        X_num_train, X_cat_train, _ = training_data
        num_numeric = X_num_train.shape[1]
        model = EmbeddingNN(num_numeric, example_embedding_specs)
        output = model(X_num_train, X_cat_train)
        assert output.shape == (X_num_train.shape[0], 1)

    @pytest.mark.it("Model contains expected number of layers")
    def test_expected_number_of_layers(self, example_embedding_specs):
        num_numeric = 3
        model = EmbeddingNN(num_numeric, example_embedding_specs)
        layer_count = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layer_count += 1
        assert layer_count == 3


@pytest.mark.describe("Embedding NN exception handling")
class TestEmbeddingNNExceptions:

    @pytest.mark.it("Raises TypeError for non-integer number of numeric features")
    def test_non_integer_numeric_features(self, example_embedding_specs):
        invalid_num_numeric = "3"
        with pytest.raises(TypeError) as excinfo:
            EmbeddingNN(invalid_num_numeric, example_embedding_specs)
        assert "num_numeric_features must be an integer" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError for negative number of numeric features")
    def test_negative_numeric_features(self, example_embedding_specs):
        invalid_num_numeric = -1
        with pytest.raises(ValueError) as excinfo:
            EmbeddingNN(invalid_num_numeric, example_embedding_specs)
        assert "num_numeric_features must be non-negative" in str(excinfo.value)

    @pytest.mark.it("Raises TypeError for non-list embedding specs")
    def test_embedding_specs_not_a_list(self):
        num_numeric = 3
        invalid_embedding_specs = {}
        with pytest.raises(TypeError) as excinfo:
            EmbeddingNN(num_numeric, invalid_embedding_specs)
        assert "embedding_specs must be a list" in str(excinfo.value)

    @pytest.mark.it("Raises TypeError if any embedding spec is not a tuple")
    def test_embedding_spec_not_a_tuple(self):
        num_numeric = 3
        invalid_embedding_specs = [(2, 1), [1, 1]]
        with pytest.raises(TypeError) as excinfo:
            EmbeddingNN(num_numeric, invalid_embedding_specs)
        assert "All embedding specs must be tuples" in str(excinfo.value)

    @pytest.mark.it(
        "Raises TypeError if any embedding spec contains non-integer values"
    )
    def test_embedding_spec_contains_non_integers(self):
        num_numeric = 3
        invalid_embedding_specs = [(2, 1), (1, "1")]
        with pytest.raises(TypeError) as excinfo:
            EmbeddingNN(num_numeric, invalid_embedding_specs)
        assert "Embedding spec values must be integers" in str(excinfo.value)

    @pytest.mark.it(
        "Raises ValueError if any embedding spec is not a tuple of length 2"
    )
    def test_embedding_spec_tuple_not_length_2(self):
        num_numeric = 3
        invalid_embedding_specs = [(2, 1), (1, 1, 1)]
        with pytest.raises(ValueError) as excinfo:
            EmbeddingNN(num_numeric, invalid_embedding_specs)
        assert "Each embedding spec must have exactly two values" in str(excinfo.value)

    @pytest.mark.it(
        "Raises ValueError if any embedding spec value is not greater than zero"
    )
    def test_embedding_spec_not_greater_than_zero(self):
        num_numeric = 3
        invalid_embedding_specs = [(2, 1), (1, -1)]
        with pytest.raises(ValueError) as excinfo:
            EmbeddingNN(num_numeric, invalid_embedding_specs)
        assert "Embedding spec values must be > 0" in str(excinfo.value)

    @pytest.mark.it("Raises ValueError if there are no categorical features")
    def test_no_input_features(self):
        num_numeric = 3
        embedding_specs = []
        with pytest.raises(ValueError) as excinfo:
            EmbeddingNN(num_numeric, embedding_specs)
        assert (
            "EmbeddingNN requires at least one categorical feature (embedding_specs must not be empty)"
            in str(excinfo.value)
        )


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
