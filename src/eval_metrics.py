import numpy as np
import pandas as pd
import torch


def _to_numpy(y: np.ndarray | pd.Series | torch.Tensor) -> np.ndarray:
    """
    Converts array-like inputs to a NumPy array.

    Raises:
        TypeError if input type is unsupported.
    """
    if isinstance(y, pd.Series):
        return y.to_numpy()
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy()
    if isinstance(y, np.ndarray):
        return y
    else:
        raise TypeError("Inputs must be a NumPy Array, Pandas Series, or Torch Tensor")


def evaluate_model(
    y_true: np.ndarray | pd.Series | torch.Tensor,
    y_pred: np.ndarray | pd.Series | torch.Tensor,
) -> dict[str, float]:
    """
    Function to evaluate model predictions against true values.
    Calculates and returns MSE, RMSE, MAE, and R squared values.

    Args:
        y_true - true target values
        y_pred - model predicted target values

    Returns:
        Dictionary of evaluation metrics (MSE, RMSE, MAE, R squared)

    Raises:
        TypeError if either input is not an array-like type.
        ValueError if the inputs differ in length.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape")

    diff = y_true - y_pred

    mse = float((diff**2).mean())
    rmse = float(np.sqrt(mse))
    mae = float(np.abs(diff).mean())

    sse = float((diff**2).sum())
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = float(1 - (sse / sst))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
