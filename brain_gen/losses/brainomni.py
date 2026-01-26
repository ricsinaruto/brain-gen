# ADAPTED FROM: https://github.com/OpenTSLab/BrainOmni/

import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainOmniCausalTokenizerLoss(nn.Module):
    """Pass-through loss for the causal tokenizer (uses its internal objective)."""

    def __init__(self):
        super().__init__()
        self.metrics: dict[str, torch.Tensor] = {
            "mae": self._time_loss,
            "pcc": self._pcc,
            "commit": self._commitment_loss,
            "amp": self._amp_loss,
            "phi": self._phase_loss,
            "ppl": self._entropy,
        }

    def _time_loss(self, outputs, targets, **kwargs):
        return outputs["l1"]

    def _entropy(self, outputs, targets, **kwargs):
        return outputs["ppl"]

    def _pcc(self, outputs, targets, **kwargs):
        return outputs["pcc"]

    def _amp_loss(self, outputs, targets, **kwargs):
        return outputs["amp"]

    def _phase_loss(self, outputs, targets, **kwargs):
        return outputs["phi"]

    def _commitment_loss(self, outputs, targets, **kwargs):
        return outputs["commit"]

    def forward(self, outputs, targets, **kwargs):
        if not isinstance(outputs, dict) or "loss" not in outputs:
            raise ValueError("Tokenizer loss expects model outputs to include 'loss'.")
        return outputs["loss"]


class BrainOmniCausalForecastLoss(nn.Module):
    """Cross-entropy over stage-wise latent logits for causal forecasting."""

    def __init__(self):
        super().__init__()
        self.metrics = {"accuracy": self._accuracy}

    def forward(self, outputs, targets, reduction: str = "mean", **kwargs):
        if not isinstance(outputs, dict):
            raise ValueError("Forecast loss expects a dict of model outputs.")

        logits = outputs.get("logits")
        target_tokens = outputs.get("targets", targets)
        if logits is None or target_tokens is None:
            raise ValueError("Missing logits/targets for BrainOmni forecast loss.")

        B, C, T, Nq, K = logits.shape
        logits = logits.reshape(B * C * T * Nq, K)
        target_tokens = target_tokens.reshape(B * C * T * Nq)
        loss = F.cross_entropy(logits.float(), target_tokens, reduction=reduction)
        if reduction == "none":
            loss = loss.view(B, C, T, Nq)

        # Optionally incorporate tokenizer commitment loss when jointly training.
        commitment = outputs.get("commitment_loss", None)
        if commitment is not None and reduction != "none":
            loss = loss + commitment
        return loss

    @staticmethod
    def _accuracy(outputs, targets=None):
        logits = outputs["logits"]
        target_tokens = outputs.get("targets", targets)
        if target_tokens is None:
            return torch.tensor(float("nan"), device=logits.device)
        pred = logits.argmax(dim=-1)
        return (pred == target_tokens).float().mean()
