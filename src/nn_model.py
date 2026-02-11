import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from src.eval_metrics import evaluate_model


class EmbeddingNN(nn.Module):
    """
    Custom class creating a PyTorch neural network model with embedding inputs.
    Model includes three linear layers and two hidden layers both using the ReLU activation function.

    Args:
        num_numeric_features - the total number of numeric features
        embedding_specs - a list of tuples detailing the cardinality and embedding dimension for each categorical feature

    Returns:
        A PyTorch neural network model with three linear layers and two hidden layers.
        Input nodes is computed from number of numerical features and embedding specs for categorical features.

    Raises:
        TypeError if:
            - num_numeric_features is not an integer
            - embedding_specs is not a list of tuples containing integer pairs (of cardinality and embedding dimensions)
        ValueError if:
            - num_numeric_features is negative
            - there are tuples without a length of 2 in the embedding_specs
            - any embedding specs contain values <= 0
            - no categorical features are provided (i.e. embedding_specs is an empty list)
    """

    def __init__(
        self, num_numeric_features: int, embedding_specs: list[tuple[int, int]]
    ):
        super().__init__()

        if not isinstance(num_numeric_features, int):
            raise TypeError("num_numeric_features must be an integer")

        if num_numeric_features <= 0:
            raise ValueError("EmbeddingNN requires at least one numerical feature")

        if not isinstance(embedding_specs, list):
            raise TypeError("embedding_specs must be a list")

        if not embedding_specs:
            raise ValueError(
                "EmbeddingNN requires at least one categorical feature (embedding_specs must not be empty)"
            )

        for embedding_spec in embedding_specs:

            if not isinstance(embedding_spec, tuple):
                raise TypeError("All embedding specs must be tuples")

            if len(embedding_spec) != 2:
                raise ValueError("Each embedding spec must have exactly two values")

            num_categories, embedding_dim = embedding_spec
            if not (isinstance(num_categories, int)) or not isinstance(
                embedding_dim, int
            ):
                raise TypeError("Embedding spec values must be integers")

            if num_categories <= 0 or embedding_dim <= 0:
                raise ValueError("Embedding spec values must be > 0")

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, embedding_dim)
                for num_categories, embedding_dim in embedding_specs
            ]
        )

        total_embedding_dims = sum(dim for _, dim in embedding_specs)
        total_input_dims = num_numeric_features + total_embedding_dims

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, X_num: torch.Tensor, X_cat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass method for making predictions through the model.

        Args:
            X_num - Tensor of numerical features data
            X_cat - Tensor of encoded categorical features data

        Returns:
            Tensor of predicted output values for each row.

        Raises:
            ValueError if:
                - No categorical features data is provided
                - No numerical features data is provided
                - The number of categorical features does not match the number of embeddings
                - The number of rows differs between the input datasets
        """

        if X_cat.numel() == 0:
            raise ValueError("X_cat must contain at least one categorical feature")

        if X_num.numel() == 0:
            raise ValueError("X_num must contain at least one numerical feature")

        if X_num.shape[0] != X_cat.shape[0]:
            raise ValueError("X_num and X_cat must have the same number of rows")

        if X_cat.shape[1] != len(self.embeddings):
            raise ValueError(
                f"Expected {len(self.embeddings)} categorical features, got {X_cat.shape[1]}"
            )

        embedded = [emb(X_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        X_emb = torch.cat(embedded, dim=1)
        X_combined = torch.cat([X_num, X_emb], dim=1)
        return self.mlp(X_combined)


def train_embedding_model(
    model: EmbeddingNN,
    X_num_train: torch.Tensor,
    X_cat_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 10,
    lr: float = 0.001,
) -> pd.DataFrame:
    """
    Function to train a PyTorch neural network model using MSE and Adam optimizer.

    Args:
        model - PyTorch Embedding NN model to be trained
        X_num_train - PyTorch tensor of numerical input features training data
        X_cat_train - PyTorch tensor of categorical input features training data
        y_train - PyTorch tensor of target training data
        epochs(Optional) - number of epochs for training, default value of 10
        lr(Optional) - learning rate for optimiser model, default value of 0.001

    Returns:
        A pandas Dataframe mapping the MSE loss against the number of epochs.
        The loss is recorded at every epoch for epochs <=10, otherwise every (epochs // 10) epochs.

    Raises:
        TypeError if:
            - input model is not a PyTorch Embedding neural network Module
            - any input Tensor is not a PyTorch Tensor type
            - epochs is not an integer
            - lr is not a float
        ValueError if:
            - the input Tensor lengths do not match
            - epochs is negative
            - lr is negative
    """
    if not isinstance(model, EmbeddingNN):
        raise TypeError("Input model must be a PyTorch Embedding neural network")
    if not all(
        isinstance(training_data, torch.Tensor)
        for training_data in (X_num_train, X_cat_train, y_train)
    ):
        raise TypeError("Input training datasets must all be torch Tensors")
    if not isinstance(epochs, int):
        raise TypeError("Number of epochs must be an integer")
    if not isinstance(lr, float):
        raise TypeError("Learning rate (lr) must be a float")

    target_no_rows = y_train.shape[0]
    if X_num_train.shape[0] != target_no_rows or X_cat_train.shape[0] != target_no_rows:
        raise ValueError("All input tensors must have the same number of rows")
    if epochs < 0:
        raise ValueError("Number of epochs cannot be negative")
    if lr < 0:
        raise ValueError("Learning rate (lr) cannot be negative")

    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses_df = pd.DataFrame(columns=["epoch", "MSE"])
    interval = 1 if epochs <= 10 else max(1, epochs // 10)

    model.train()

    for epoch in range(epochs):
        predictions = model.forward(X_num_train, X_cat_train)
        MSE = loss(predictions, y_train)
        MSE.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % interval == 0:
            losses_df.loc[len(losses_df)] = [epoch + 1, MSE.item()]

    return losses_df


def evaluate_embedding_model(
    model: EmbeddingNN,
    X_num_test: torch.Tensor,
    X_cat_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict[str, float]:
    """
    Evaluate a trained PyTorch embedding neural network on called out test data.

    Args:
        model - a trained PyTorch embedding neural network model.
        X_num_test - numerical features test data.
        X_cat_test - categorical features test data.
        y_test - target test data.

    Returns:
        Evaluation metrics for the model (MSE, RMSE, MAE, R2).

    Raises:
        TypeError if:
            - model is not a PyTorch Embedding neural network model
            - any test data is not a torch Tensor
        ValueError if:
            - all test data does not have the same number of observations
            - X_cat_test does not have the same number of categorical features as the trained model
            - y_test is not one dimensional
    """
    if not isinstance(model, EmbeddingNN):
        raise TypeError("Input model must be a PyTorch Embedding neural network")
    if not all(
        isinstance(testing_data, torch.Tensor)
        for testing_data in (X_num_test, X_cat_test, y_test)
    ):
        raise TypeError("Input testing datasets must all be torch Tensors")

    target_no_rows = y_test.shape[0]
    if X_num_test.shape[0] != target_no_rows or X_cat_test.shape[0] != target_no_rows:
        raise ValueError("All input tensors must have the same number of rows")
    if X_cat_test.shape[1] != len(model.embeddings):
        raise ValueError(
            f"Expected {len(model.embeddings)} categorical features, got {X_cat_test.shape[1]}"
        )

    model_training_state = model.training
    model.eval()
    with torch.no_grad():
        y_pred = model(X_num_test, X_cat_test)
    if model_training_state:
        model.train()
    return evaluate_model(y_test, y_pred)
