from __future__ import annotations

import pytorch_lightning as pl
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks import EarlyStopping, BatchSizeFinder
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import F1Score, ConfusionMatrix
from typing import Optional, TYPE_CHECKING
import os
import torchview  # noqa: F401  # optional, used only for model graph plotting
import matplotlib.pyplot as plt
import seaborn as sns

from ..dataset import split_datasets
from .lightning import LitModel, LitDataModule, DatasetEpochCallback

from .utils import get_model_class, get_loss_class
from .train_bpe import TextBPETokenizerTrainer
from .checkpointing import EvaluationLauncher, ThreadedModelCheckpoint
from .performance import PerformanceMonitor
from ..dataset import TextDataLoader  # noqa: F401

if TYPE_CHECKING:
    from .vidtok import VidtokLightning  # noqa: F401


class ExperimentTokenizer:
    def __init__(self, cfg: dict) -> None:
        """Args:

        cfg: Configuration dictionary
        """
        datasets = split_datasets(**cfg["datasplitter"])

        # get all training data
        train_data = []
        for i in range(len(datasets.train)):
            x, _ = datasets.train[i]
            train_data.append(x[0])

        train_data = torch.stack(train_data).permute(0, 2, 1)  # (B, T, C)
        print(train_data.shape)

        tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        tokenizer = tokenizer.fit(action_data=train_data, vocab_size=cfg["vocab_size"])

        tokenizer.save_pretrained(cfg["save_dir"])


class ExperimentTokenizerText:
    def __init__(self, cfg: dict) -> None:
        """Args:

        cfg: Configuration dictionary
        """
        tokenizer = TextBPETokenizerTrainer(**cfg["tokenizer"])
        tokenizer.train()


class ExperimentDL:
    def __init__(self, cfg: dict, lit_model: Optional[LitModel] = None) -> None:
        """Args:

        cfg: Configuration dictionary lit_model: Optional[LitModel] = None
        """
        # print pytorch version
        print("--------------------------------")
        print(f"PyTorch Lightning version: {pl.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print("--------------------------------")

        self.cfg = cfg
        self.trainer_args = cfg["trainer"]
        self.dataloader_args = cfg["dataloader"]
        self.resume_from = cfg["resume_from"]
        self.dataset_name = cfg.get("dataset_name", "omega")
        self.save_dir = cfg["save_dir"]
        self.free_run_cfg = cfg.get("k_step_free_run")
        # Remove to avoid passing unknown kwargs to Lightning Trainer
        self.early_stopping_cfg = self.trainer_args.pop("early_stopping", None)
        # Optional full validation pass before resuming training.
        self.validate_before_resume = bool(
            self.trainer_args.pop("validate_before_resume", False)
        )

        self.trainer_args["default_root_dir"] = cfg["save_dir"]

        # load model config
        with open(cfg["model_config"]) as f:
            model_cfg = yaml.safe_load(f)

        # TensorBoard logger
        self.logger = TensorBoardLogger(
            save_dir=self.save_dir, name=cfg.get("experiment_name", "logs")
        )
        log_every = self.trainer_args.get("log_every_n_steps", 1)

        self.datasets = split_datasets(**cfg["datasplitter"])

        # Get model and loss classes dynamically
        model_class = None
        loss_class = None
        if cfg.get("model_name"):
            model_class = get_model_class(cfg["model_name"])
        if cfg.get("loss_name"):
            loss_class = get_loss_class(cfg["loss_name"])

        postprocessor = getattr(self.datasets.train, "postprocessor", None)

        lit_model_args = {
            "model_class": model_class,
            "loss_class": loss_class,
            "model_cfg": model_cfg,
            "loss_cfg": cfg.get("loss", {}),
            "trainer_cfg": cfg["lightning"],
            "postprocessor": postprocessor,
            "free_run_cfg": self.free_run_cfg,
        }
        if lit_model is not None:
            self.lit_model = lit_model(**lit_model_args)
        else:
            self.lit_model = LitModel(**lit_model_args)

        self.eval_launcher = None
        if cfg.get("eval_runner", False):
            if cfg["eval_runner"].get("enabled", False):
                self.eval_launcher = EvaluationLauncher(cfg, Path(self.save_dir))

        ckpt_cadence = self.trainer_args.pop("checkpoint_cadence_epochs", None)

        best_ckpt = ThreadedModelCheckpoint(
            monitor="val/loss",  # metric to monitor
            mode="min",  # 'min' for loss, 'max' for accuracy or similar
            save_top_k=1,  # save only the best model
            filename="best-checkpoint",  # optional: custom filename
        )
        epoch_ckpt = ThreadedModelCheckpoint(
            filename="last-checkpoint",
            save_top_k=-1,  # keep every epoch
            save_on_train_epoch_end=True,  # trigger right after each epoch
            epoch_cadence=ckpt_cadence,
            after_save=self.eval_launcher,
        )

        callbacks = self.trainer_args.get("callbacks", []) or []
        callbacks.extend([best_ckpt, epoch_ckpt])

        perf_callback = PerformanceMonitor(log_every_n_steps=log_every)
        callbacks.append(perf_callback)

        if self.trainer_args.pop("tune_batch_size", True):
            batch_size_finder = BatchSizeFinder(
                mode="power",  # or "binsearch"
                init_val=self.dataloader_args["batch_size"],
                steps_per_trial=3,
                max_trials=25,
                batch_arg_name="batch_size",
            )
            callbacks.append(batch_size_finder)

        # Optional early stopping based on validation loss stagnation
        early_stopping_cfg = (
            self.early_stopping_cfg
            if self.early_stopping_cfg is not None
            else cfg.get("early_stopping")
        )
        if isinstance(early_stopping_cfg, bool):
            if early_stopping_cfg:
                callbacks.append(self._build_early_stopping({}))
        elif early_stopping_cfg is not None:
            callbacks.append(self._build_early_stopping(early_stopping_cfg))

        # If the training dataset exposes an epoch hook, add a callback
        if hasattr(self.datasets, "train") and (
            hasattr(self.datasets.train, "set_epoch")
            or hasattr(self.datasets.train, "on_epoch_start")
        ):
            callbacks.append(DatasetEpochCallback(self.datasets.train))
            callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2))

        self.trainer_args["callbacks"] = callbacks
        self.trainer_args["logger"] = self.logger

    def _build_early_stopping(self, cfg) -> EarlyStopping:
        """Normalize user config and build an EarlyStopping callback."""
        if cfg is True:
            cfg = {}
        if isinstance(cfg, int):
            cfg = {"patience": int(cfg)}
        if not isinstance(cfg, dict):
            raise ValueError("early_stopping must be a bool, int, or dict.")

        patience = int(cfg.get("patience", 10))
        if patience < 0:
            raise ValueError("early_stopping.patience must be >= 0")

        kwargs = {
            "monitor": cfg.get("monitor", "val/loss"),
            "mode": cfg.get("mode", "min"),
            "patience": patience,
            "min_delta": float(cfg.get("min_delta", 0.0)),
            "check_on_train_epoch_end": bool(
                cfg.get("check_on_train_epoch_end", False)
            ),
        }
        for key in (
            "stopping_threshold",
            "divergence_threshold",
            "strict",
            "verbose",
            "check_finite",
            "log_rank_zero_only",
        ):
            if key in cfg:
                kwargs[key] = cfg[key]
        return EarlyStopping(**kwargs)

    def _visualize_model(self, cfg: dict) -> None:
        # Use torchview to visualize the model
        x = torch.randn(1, 2, 100)
        sensor_pos_ori = torch.randn(1, 2, 2)
        sensor_type = torch.randint(0, 2, (1, 2))
        self.model_graph = torchview.draw_graph(
            self.lit_model.model,
            input_data=[(x, sensor_pos_ori, sensor_type)],
            save_graph=True,
            directory=cfg["save_dir"],
            filename="model_graph.pdf",
        )

    def train(self) -> None:
        args = self.dataloader_args

        dataloader_cls_name = self.cfg.get("dataloader_class", "DataLoader")
        dataloader_cls = globals()[dataloader_cls_name]

        dm = LitDataModule(self.datasets.train, self.datasets.val, dataloader_cls, args)
        trainer = pl.Trainer(**self.trainer_args)

        if self.validate_before_resume and self.resume_from:
            # Log a full validation pass before resuming training.
            trainer.validate(self.lit_model, datamodule=dm, ckpt_path=self.resume_from)

        trainer.fit(self.lit_model, datamodule=dm, ckpt_path=self.resume_from)

    def test(self) -> None:
        args = self.dataloader_args
        test_loader = DataLoader(self.datasets.test, shuffle=False, **args)
        trainer = pl.Trainer(**self.trainer_args)

        trainer.test(self.lit_model, test_loader, ckpt_path=self.resume_from)

        if self.lit_model.test_targets:
            # compute F1 score and confusion matrix
            targets = torch.cat(self.lit_model.test_targets)
            preds = torch.cat(self.lit_model.test_predictions)
            preds_classes = preds.argmax(dim=-1)
            num_classes = preds.size(-1)

            f1_macro = F1Score(
                task="multiclass", average="macro", num_classes=num_classes
            )
            f1_score = f1_macro(preds, targets)
            print(f"F1 score: {f1_score}")

            # Compute confusion matrix
            cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)
            confusion = cm(preds_classes, targets)
            confusion_np = confusion.cpu().numpy()

            # Save confusion matrix to file
            os.makedirs(self.save_dir, exist_ok=True)
            cm_pdf_path = os.path.join(self.save_dir, "confusion_matrix.pdf")

            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_np, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(cm_pdf_path)
            plt.close()
            print(f"Confusion matrix saved to {cm_pdf_path}")
            return

        # save test predictions for datasets that expose the helper
        if hasattr(self.datasets.test, "generate_submission_in_csv"):
            preds = torch.cat(self.lit_model.test_predictions)
            preds = [p for p in preds]
            self.datasets.test.generate_submission_in_csv(
                preds, f"{self.save_dir}/holdout_phoneme_predictions.csv"
            )


class ExperimentVidtok(ExperimentDL):
    """Training harness for the VidTok model on MEG interpolated images."""

    def __init__(self, cfg: dict) -> None:
        # Run base init to set up logging, callbacks, trainer args, and dataset splits.
        from .vidtok import VidtokLightning, ImageSaverCallback

        super().__init__(cfg, lit_model=VidtokLightning)

        # Add image saver callback for reconstruction visualization
        image_saver = ImageSaverCallback(
            save_dir=self.save_dir,
            num_samples=cfg.get("image_saver_num_samples", 10),
        )
        self.trainer_args["callbacks"].append(image_saver)
