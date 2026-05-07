import torch
import torch.nn as nn


def mse_with_regularizer_loss(inputs, targets, model, lamda=1.5e-3):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    mse_loss = torch.sum((inputs - targets) ** 2) / 2
    return mse_loss + reg_loss


def weighted_mse_loss(inputs, targets, temporal_weights=None, threshold_low=10, threshold_high=60):
    """
    Weighted MSE loss that emphasizes certain conditions
    
    Args:
        inputs: predicted values
        targets: ground truth values
        temporal_weights: weights for different timesteps (default: None)
        threshold_low: lower threshold for congestion weight
        threshold_high: upper threshold for congestion weight
    
    Returns:
        weighted MSE loss
    """
    mse = (inputs - targets) ** 2
    
    # Weight by traffic condition (congestion gets higher weight)
    # Free flow (speed > threshold_high): weight = 1.0
    # Congestion (threshold_low < speed < threshold_high): weight = 2.0
    # Severe congestion (speed < threshold_low): weight = 3.0
    condition_weights = torch.ones_like(targets)
    condition_weights[(targets > threshold_low) & (targets < threshold_high)] = 2.0
    condition_weights[targets <= threshold_low] = 3.0
    
    # Apply condition weights
    weighted_mse = mse * condition_weights
    
    # Apply temporal weights if provided
    if temporal_weights is not None:
        weighted_mse = weighted_mse * temporal_weights
    
    return weighted_mse.mean()


def smooth_l1_loss(inputs, targets, beta=1.0):
    """
    Smooth L1 loss (Huber loss) - robust to outliers
    
    Args:
        inputs: predicted values
        targets: ground truth values
        beta: transition point between L1 and L2
    
    Returns:
        smooth L1 loss
    """
    loss_fn = nn.SmoothL1Loss(beta=beta, reduction='mean')
    return loss_fn(inputs, targets)


def temporal_smoothing_loss(predictions, lambda_smooth=0.1):
    """
    Penalize abrupt changes between consecutive predictions
    
    Args:
        predictions: (batch, num_nodes, horizon) or (batch*num_nodes, horizon)
        lambda_smooth: weight for smoothing loss
    
    Returns:
        temporal smoothing loss
    """
    # Calculate differences between consecutive timesteps
    if len(predictions.shape) == 3:
        # (batch, num_nodes, horizon)
        diffs = torch.diff(predictions, dim=2)  # (batch, num_nodes, horizon-1)
    else:
        # (batch*num_nodes, horizon)
        diffs = torch.diff(predictions, dim=1)  # (batch*num_nodes, horizon-1)
    
    # L1 norm of differences
    smoothing_loss = torch.mean(torch.abs(diffs))
    
    return lambda_smooth * smoothing_loss
