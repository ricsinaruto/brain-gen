from typing import Any, Optional
from pathlib import Path
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from brain_gen.training.lightning import LitModel


class VidtokLightning(LitModel):
    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)

        outputs, targets = self.postprocessor(
            outputs[0].detach(), targets.detach(), gaussian=True
        )

        metrics_for_stage: dict[str, torch.Tensor] = {}
        self.log(f"{stage}/loss", loss.detach(), prog_bar=True)
        for name, metric in self.loss.metrics.items():
            metric_val = metric(outputs, targets)
            metrics_for_stage[name] = metric_val
            self.log(f"{stage}/{name}", metric_val, prog_bar=True)

        # log learning rate
        lr = None
        if stage == "train":
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr", lr)

        return loss


class ImageSaverCallback(pl.Callback):
    """Saves input and reconstructed images to disk at the end of each epoch."""

    def __init__(self, save_dir: str, num_samples: int = 4) -> None:
        super().__init__()
        self.save_dir = Path(save_dir) / "reconstructions"
        self.num_samples = num_samples
        self._cached_batch: Optional[torch.Tensor] = None

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Cache first batch of validation data for reconstruction visualization
        if batch_idx == 0 and self._cached_batch is None:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            # Store a few samples (detached, on CPU to save GPU memory)
            self._cached_batch = x[: self.num_samples].detach().clone()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: "VidtokLightning"
    ) -> None:
        if self._cached_batch is None:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)
        epoch = trainer.current_epoch

        # Move batch to model device and run reconstruction
        x = self._cached_batch.to(pl_module.device)
        if x.ndim == 4:
            x = x.unsqueeze(2)

        with torch.no_grad():
            xrec, _ = pl_module.model(x, global_step=pl_module.global_step)

        if x.ndim == 5 and xrec.ndim == 4:
            xrec = xrec.unsqueeze(2)

        # Handle temporal dim: take middle frame if 5D
        if x.ndim == 5:
            t_mid = x.shape[2] // 2
            x_img = x[:, :, t_mid, :, :]  # (B, C, H, W)
            xrec_img = xrec[:, :, t_mid, :, :]
        else:
            x_img = x
            xrec_img = xrec

        # Move to CPU and convert to numpy
        x_np = x_img.cpu().numpy()
        xrec_np = xrec_img.cpu().numpy()

        # Create figure with input/reconstruction pairs
        n = min(self.num_samples, x_np.shape[0])
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)

        for i in range(n):
            # Input image - handle single or multi-channel
            img_in = x_np[i]
            img_rec = xrec_np[i]

            # If single channel, squeeze; else take first channel or average
            if img_in.shape[0] == 1:
                img_in = img_in[0]
                img_rec = img_rec[0]
            else:
                # Average over channels for visualization
                img_in = img_in.mean(axis=0)
                img_rec = img_rec.mean(axis=0)

            axes[0, i].imshow(img_in, cmap="RdBu_r", aspect="auto")
            axes[0, i].set_title(f"Input {i}")
            axes[0, i].axis("off")

            axes[1, i].imshow(img_rec, cmap="RdBu_r", aspect="auto")
            axes[1, i].set_title(f"Reconstruction {i}")
            axes[1, i].axis("off")

        fig.suptitle(f"Epoch {epoch}")
        plt.tight_layout()

        save_path = self.save_dir / f"epoch_{epoch:04d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Clear cached batch so we get fresh samples next epoch
        self._cached_batch = None
