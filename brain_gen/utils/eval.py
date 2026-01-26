import torch
from torch import Tensor

from torchmetrics import F1Score
import torch.nn.functional as F

from ..utils.quantizers import mulaw_inv_torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    # Support soft targets by converting to hard indices via argmax
    if isinstance(targets, tuple) or isinstance(targets, list):
        targets = targets[0]

    if targets.dim() == logits.dim():
        targets = targets.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean()


def top_k_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 5
) -> torch.Tensor:
    """Compute top-k accuracy.

    Supports soft targets.
    """
    topk = logits.topk(k, dim=-1).indices
    if isinstance(targets, tuple) or isinstance(targets, list):
        targets = targets[0]

    if targets.dim() == logits.dim():
        targets = targets.argmax(dim=-1)
    correct = topk.eq(targets.unsqueeze(-1)).any(dim=-1).float()
    return correct.mean()


def f1_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if isinstance(targets, tuple) or isinstance(targets, list):
        targets = targets[0]

    # reshape to 2D
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)

    f1_macro = F1Score(
        task="multiclass", average="macro", num_classes=logits.size(-1)
    ).to(logits.device)

    return f1_macro(logits, targets)


def mse_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    # Support soft targets by converting to hard indices via argmax
    if isinstance(targets, tuple) or isinstance(targets, list):
        targets = targets[0]

    if targets.dim() == logits.dim():
        targets = targets.argmax(dim=-1)

    # convert to continuous with inverse mulaw
    preds = mulaw_inv_torch(preds)
    targets = mulaw_inv_torch(targets)

    return F.mse_loss(preds, targets)


@torch.inference_mode()
def sample(
    logits: Tensor,
    strategy: str = "argmax",
    temperature: float | Tensor = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> Tensor:
    """Sample from *logits* according to *strategy* (last dim = vocab).

    Args:     logits: Tensor of shape (B, C, Q) containing the logits for the next token

    Returns:     Tensor of shape (B, C) containing the sampled tokens
    """

    if strategy == "argmax":
        return logits.argmax(dim=-1)

    if isinstance(temperature, (int, float)):
        temperature = float(temperature)
        temperature = temperature if temperature > 0.0 else 1.0
    else:
        temperature = torch.as_tensor(
            temperature, device=logits.device, dtype=logits.dtype
        )
        if temperature.numel() == 1:
            temp_val = float(temperature.item())
            temperature = temp_val if temp_val > 0.0 else 1.0
        else:
            temperature = torch.clamp(temperature, min=1e-6)

    probs = torch.softmax(logits / temperature, dim=-1)

    if strategy == "roulette":
        flat = probs.view(-1, probs.size(-1))
        return torch.multinomial(flat, 1).view(logits.shape[:-1])

    if strategy == "top_k":
        k = min(top_k, probs.size(-1))
        vals, idx = torch.topk(probs, k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True)
        samp = torch.multinomial(vals.view(-1, k), 1).view(*logits.shape[:-1])
        return torch.gather(idx, -1, samp.unsqueeze(-1)).squeeze(-1)

    if strategy == "top_p":
        sorted_p, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        mask = cum > top_p
        mask[..., 0] = False  # keep at least one
        sorted_p[mask] = 0.0
        sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
        samp = torch.multinomial(sorted_p.view(-1, sorted_p.size(-1)), 1).view(
            *logits.shape[:-1]
        )
        return torch.gather(sorted_idx, -1, samp.unsqueeze(-1)).squeeze(-1)

    raise ValueError(f"Unknown sampling strategy '{strategy}'")
