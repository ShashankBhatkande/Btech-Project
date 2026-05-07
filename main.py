import argparse
import inspect
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import torch.serialization
import numpy
import models
import models.gcn
import models.gru
import models.tgcn
import tasks
import utils.callbacks
import utils.data
import utils.email
import utils.logging


DATA_PATHS = {
    "pemsd7_228": {"feat": "data/PeMSD7_V_228.csv", "adj": "data/PeMSD7_W_228.csv"},
    "pemsd7_1028": {"feat": "data/PeMSD7_V_1028.csv", "adj": "data/PeMSD7_W_1028.csv"},
}


def add_trainer_arguments(parser):
    if hasattr(pl.Trainer, "add_argparse_args"):
        return pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of training epochs")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use")
    parser.add_argument("--accelerator", type=str, default=None, help="Accelerator type")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices to use")
    parser.add_argument("--precision", type=int, default=32, help="Precision to use")
    parser.add_argument("--enable_checkpointing", action="store_true", help="Enable checkpointing")
    return parser


def build_trainer(args, callbacks):
    if hasattr(pl.Trainer, "from_argparse_args"):
        trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
        # Force TensorBoard logger if available
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
            trainer.logger = TensorBoardLogger("lightning_logs", name="T-GCN")
        except ImportError:
            pass
        return trainer

    trainer_kwargs = {}
    trainer_params = set(inspect.signature(pl.Trainer.__init__).parameters) - {"self"}
    for name in ("max_epochs", "gpus", "accelerator", "devices", "precision", "enable_checkpointing"):
        if name in trainer_params and hasattr(args, name):
            trainer_kwargs[name] = getattr(args, name)
    return pl.Trainer(callbacks=callbacks, **trainer_kwargs)


def get_model(args, dm):
    model = None
    if args.model_name == "GCN":
        model = models.GCN(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
    if args.model_name == "GRU":
        model = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim)
    if args.model_name == "TGCN":
        model = models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim)
    return model


def get_task(args, model, dm):
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm.feat_max_val, **vars(args)
    )
    return task


def get_callbacks(args):
    monitor_metric = getattr(args, "monitor_metric", "val_loss")
    monitor_mode = "max" if monitor_metric in ("accuracy", "R2", "ExplainedVar") else "min"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=3,
        verbose=True
    )
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    
    # Early stopping: stop if monitored metric doesn't improve for 15 epochs
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=monitor_metric,
        mode=monitor_mode,
        patience=15,
        verbose=True,
        check_finite=True
    )
    
    callbacks = [
        checkpoint_callback,
        plot_validation_predictions_callback,
        early_stopping_callback,
    ]
    return callbacks


def main_supervised(args):
    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)
    trainer = build_trainer(args, callbacks)
    trainer.fit(task, dm)
    # Temporarily allow all globals for checkpoint loading (trusted local checkpoint)
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    torch.load = patched_load
    try:
        results = trainer.validate(datamodule=dm)
    finally:
        torch.load = original_load
    return results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_trainer_arguments(parser)

    parser.add_argument(
        "--data", type=str, help="The name of the dataset", choices=("pemsd7_228", "pemsd7_1028"), default="pemsd7_228"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN"),
        default="GCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="val_loss",
        choices=("val_loss", "accuracy", "RMSE", "MAE", "R2", "ExplainedVar"),
        help="Metric to monitor for checkpointing and early stopping",
    )

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)

    try:
        results = main(args)
    except:  # noqa: E722
        traceback.print_exc()
        if args.send_email:
            tb = traceback.format_exc()
            subject = "[Email Bot][❌] " + "-".join([args.settings, args.model_name, args.data])
            utils.email.send_email(tb, subject)
        exit(-1)

    if args.send_email:
        subject = "[Email Bot][✅] " + "-".join([args.settings, args.model_name, args.data])
        utils.email.send_experiment_results_email(args, results, subject=subject)

