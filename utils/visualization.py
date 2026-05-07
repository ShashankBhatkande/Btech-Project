import numpy as np
import matplotlib.pyplot as plt
import torch


def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)


def plot_actual_vs_predicted(y_true, y_pred, node_idx=10, horizon_step=0, save_path=None):
    """Plot actual vs predicted values for a single node and horizon step."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.ndim != 3 or y_pred.ndim != 3:
        raise ValueError("y_true and y_pred must have shape (B, N, H)")
    if node_idx < 0 or node_idx >= y_true.shape[1]:
        raise IndexError(f"node_idx must be between 0 and {y_true.shape[1] - 1}")
    if horizon_step < 0 or horizon_step >= y_true.shape[2]:
        raise IndexError(f"horizon_step must be between 0 and {y_true.shape[2] - 1}")

    series_true = y_true[:, node_idx, horizon_step]
    series_pred = y_pred[:, node_idx, horizon_step]

    plt.figure()
    plt.plot(series_true, label="Actual")
    plt.plot(series_pred, label="Predicted")
    plt.xlabel("Sample index")
    plt.ylabel("Traffic speed")
    plt.title(f"Actual vs Predicted for node {node_idx} at horizon step {horizon_step + 1}")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_multi_step_prediction(y_true, y_pred, sample_idx=0, node_idx=10):
    """Plot multi-step forecast for a single sample and node."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.ndim != 3 or y_pred.ndim != 3:
        raise ValueError("y_true and y_pred must have shape (B, N, H)")
    if sample_idx < 0 or sample_idx >= y_true.shape[0]:
        raise IndexError(f"sample_idx must be between 0 and {y_true.shape[0] - 1}")
    if node_idx < 0 or node_idx >= y_true.shape[1]:
        raise IndexError(f"node_idx must be between 0 and {y_true.shape[1] - 1}")

    horizon = y_true.shape[2]
    steps = np.arange(1, horizon + 1)

    plt.figure()
    plt.plot(steps, y_true[sample_idx, node_idx, :], marker="o", label="Actual")
    plt.plot(steps, y_pred[sample_idx, node_idx, :], marker="o", label="Predicted")
    plt.xlabel("Horizon step")
    plt.ylabel("Traffic speed")
    plt.title(f"Multi-step prediction for sample {sample_idx}, node {node_idx}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss_curve(train_losses, val_losses):
    """Plot training and validation loss over epochs."""
    train_losses = _to_numpy(train_losses)
    val_losses = _to_numpy(val_losses)

    if train_losses.ndim != 1 or val_losses.ndim != 1:
        raise ValueError("train_losses and val_losses must be one-dimensional lists or arrays")
    if train_losses.shape[0] != val_losses.shape[0]:
        raise ValueError("train_losses and val_losses must have the same length")

    epochs = np.arange(1, train_losses.shape[0] + 1)

    plt.figure()
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_vs_horizon(y_true, y_pred):
    """Plot mean absolute error for each prediction horizon step."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.ndim != 3 or y_pred.ndim != 3:
        raise ValueError("y_true and y_pred must have shape (B, N, H)")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    mae_per_step = np.mean(np.abs(y_pred - y_true), axis=(0, 1))
    steps = np.arange(1, mae_per_step.shape[0] + 1)

    plt.figure()
    plt.plot(steps, mae_per_step, marker="o")
    plt.xlabel("Horizon step")
    plt.ylabel("MAE")
    plt.title("Mean Absolute Error vs Horizon")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(model_names, mae_values):
    """Plot a bar chart comparing multiple models using MAE values."""
    model_names = list(model_names)
    mae_values = _to_numpy(mae_values)

    if mae_values.ndim != 1 or len(model_names) != mae_values.shape[0]:
        raise ValueError("model_names and mae_values must have the same length")

    x = np.arange(len(model_names))

    plt.figure()
    plt.bar(x, mae_values)
    plt.xticks(x, model_names)
    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.show()


def plot_scatter_actual_vs_predicted(y_true, y_pred):
    """Plot a scatter plot of flattened actual vs predicted values."""
    y_true = _to_numpy(y_true).ravel()
    y_pred = _to_numpy(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Scatter")
    plt.tight_layout()
    plt.show()
