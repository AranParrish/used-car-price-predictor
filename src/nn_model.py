import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd


def create_nn_model(input_dim: int) -> nn.Module:
    """
    Function to create a PyTorch neural network model with three linear layers.

    Args:
        input_dim - the input dimension (i.e. the number of input features)

    Returns:
        A PyTorch neural network model with three linear layers and two hidden layers using the ReLU activation function.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    return model


def train_nn_model(
    model: nn.Module,
    X_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    epochs: int = 10,
    lr: float = 0.001,
) -> pd.DataFrame:
    """
    Function to train a PyTorch neural network model using MSE and Adam optimizer.

    Args:
        model - PyTorch NN model to be trained
        X_train_tensor - PyTorch tensor of input features training data
        y_train_tensor - PyTorch tensor of target training data
        epochs(Optional) - number of epochs for training, default value of 10
        lr(Optional) - learning rate for optimiser model, default value of 0.001

    Returns:
        A pandas Dataframe mapping the MSE loss against the number of epochs.
        The loss is recorded at every epoch for epochs <=10, otherwise every (epochs // 10) epochs.

    Raises:
        TypeError if:
            - input model is not a PyTorch neural network Module
            - either input Tensor is not a PyTorch Tensor type
            - epochs is not an integer
            - lr is not a float
        ValueError if:
            - the input Tensor lengths do not match
            - epochs is negative
            - lr is negative
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Input model must be a PyTorch neural network")

    if not all(
        isinstance(training_data, torch.Tensor)
        for training_data in (X_train_tensor, y_train_tensor)
    ):
        raise TypeError("Input training datasets must both be a PyTorch Tensor")

    if not isinstance(epochs, int):
        raise TypeError("Number of epochs must be an integer")

    if not isinstance(lr, float):
        raise TypeError("Learning rate (lr) must be a float")

    if X_train_tensor.shape[0] != y_train_tensor.shape[0]:
        raise ValueError("Number of rows in input tensors must match")

    if epochs < 0:
        raise ValueError("Number of epochs cannot be negative")

    if lr < 0:
        raise ValueError("Learning rate (lr) cannot be negative")

    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses_df = pd.DataFrame(columns=["epoch", "MSE"])
    interval = 1 if epochs <= 10 else max(1, epochs // 10)

    for epoch in range(epochs):
        predictions = model(X_train_tensor)
        MSE = loss(predictions, y_train_tensor)
        MSE.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % interval == 0:
            losses_df.loc[len(losses_df)] = [epoch + 1, MSE.item()]

    return losses_df
