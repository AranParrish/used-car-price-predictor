import torch.nn as nn
import torch


def create_nn_model(input_dim: int) -> nn.Module:
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
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    epochs: int = 10,
    lr: float = 0.001,
):
    pass
