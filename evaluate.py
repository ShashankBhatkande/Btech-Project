"""
Evaluation script to generate visualizations from trained model predictions.
Collects predictions from validation set and creates graphs.
"""


import torch
import numpy as np
from tasks.supervised import SupervisedForecastTask
from utils.data import SpatioTemporalCSVDataModule
from utils import (
    plot_actual_vs_predicted,
    plot_multi_step_prediction,
    plot_loss_curve,
    plot_error_vs_horizon,
    plot_model_comparison,
    plot_scatter_actual_vs_predicted,
)
import pandas as pd
import argparse

# Allow loading custom model classes
torch.serialization.add_safe_globals([
    "models.tgcn.TGCN",
])



def collect_predictions(model, datamodule, device="cpu"):
    """Collect predictions and ground truth from validation set."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    val_loader = datamodule.val_dataloader()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            predictions = model(x)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())
    
    y_pred = torch.cat(all_predictions, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    
    # Transpose y_true from (B, H, N) to (B, N, H) to match y_pred shape
    y_true = y_true.transpose(1, 2)
    
    return y_true, y_pred


def main():
    """Example usage: load checkpoint and visualize predictions."""
    parser = argparse.ArgumentParser(description="Evaluate T-GCN on PeMSD7 dataset.")
    parser.add_argument('--data', type=str, default='pemsd7_228', choices=['pemsd7_228', 'pemsd7_1026'], help='Dataset to use (pemsd7_228 or pemsd7_1026)')
    parser.add_argument('--checkpoint', type=str, default='lightning_logs/T-GCN/version_33/checkpoints/epoch=45-step=18354.ckpt', help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--seq_len', type=int, default=12, help='Input sequence length')
    parser.add_argument('--pre_len', type=int, default=6, help='Prediction length')
    args = parser.parse_args()

    DATA_PATHS = {
        "pemsd7_228": {"feat": "data/PeMSD7_V_228.csv", "adj": "data/PeMSD7_W_228.csv"},
        "pemsd7_1026": {"feat": "data/PeMSD7_V_1026.csv", "adj": "data/PeMSD7_W_1026.csv"},
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data module
    dm = SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"],
        adj_path=DATA_PATHS[args.data]["adj"],
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
    )
    dm.setup()

    checkpoint_path = args.checkpoint

    try:
        # Patch torch.load to allow loading custom classes
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        task = SupervisedForecastTask.load_from_checkpoint(
            checkpoint_path,
            feat_max_val=dm.feat_max_val
        )
        model = task.to(device)
        print(f"Loaded model from {checkpoint_path}")
        
        # Restore original torch.load
        torch.load = original_load
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        torch.load = original_load
        print("Skipping predictions collection.")
        print("\nTo use this script:")
        print("1. Train your model with: python main.py --max_epochs 50 ...")
        print("2. Find checkpoint in: lightning_logs/T-GCN/versionXX/checkpoints/")
        print("3. Update checkpoint_path in evaluate.py")
        print("4. Run: python evaluate.py")
        return
    
    # Collect predictions
    print("Collecting predictions from validation set...")
    y_true, y_pred = collect_predictions(model, dm, device)
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Ground truth shape: {y_true.shape}")
    
    # Denormalize
    y_true_denorm = y_true * dm.feat_max_val
    y_pred_denorm = y_pred * dm.feat_max_val
    
    # Save predictions and actuals to CSV
    records = []
    B, N, H = y_true_denorm.shape
    for b in range(B):
        for n in range(N):
            for h in range(H):
                records.append({
                    'sample_index': b,
                    'node_index': n,
                    'horizon_step': h+1,
                    'actual_speed': float(y_true_denorm[b, n, h]),
                    'predicted_speed': float(y_pred_denorm[b, n, h]),
                })
    df = pd.DataFrame(records)
    df.to_csv('predictions_vs_actual.csv', index=False)
    print("Saved predictions and actuals to predictions_vs_actual.csv")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("1. Actual vs Predicted (node 10, all horizon steps)")
    for horizon_step in range(6):
        plot_actual_vs_predicted(y_true_denorm, y_pred_denorm, node_idx=10, horizon_step=horizon_step)
    
    print("2. Multi-step prediction (sample 0, node 10)")
    plot_multi_step_prediction(y_true_denorm, y_pred_denorm, sample_idx=0, node_idx=10)
    
    print("3. Error vs Horizon")
    plot_error_vs_horizon(y_true_denorm, y_pred_denorm)
    
    print("4. Scatter plot (actual vs predicted)")
    plot_scatter_actual_vs_predicted(y_true_denorm, y_pred_denorm)
    
   
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
