# ADAPTED FROM: https://github.com/OpenTSLab/BrainOmni/

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from ..training.lightning import LitModel
from ..layers.brainomni.attn import (
    RMSNorm,
    SpatialTemporalAttentionBlock,
)
from ..utils.eval import sample as sample_logits

from .tokenizers.brainomni import BrainOmniCausalTokenizer, CausalTokenSequence


class BrainOmniCausalForecast(nn.Module):
    """Autoregressive forecaster over BrainOmni latent tokens with causal attention."""

    def __init__(
        self,
        overlap_ratio: float,
        lm_dim: int,
        lm_head: int,
        lm_depth: int,
        lm_dropout: float,
        num_quantizers_used: int | None = None,
        freeze_tokenizer: bool = False,
        tokenizer_kwargs: dict = None,
        tokenizer_path: str = None,
    ):
        super().__init__()
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        self.lm_dim = lm_dim
        self.overlap_ratio = overlap_ratio
        self.num_quantizers_used = tokenizer_kwargs.get("num_quantizers")
        self.freeze_tokenizer = freeze_tokenizer

        self.tokenizer = BrainOmniCausalTokenizer(**tokenizer_kwargs)
        n_dim = tokenizer_kwargs.get("n_dim")

        if tokenizer_path is not None:
            lit = LitModel.load_from_checkpoint(tokenizer_path, strict=False)
            self.tokenizer = lit.model

            # check if model is compiled
            if hasattr(self.tokenizer, "_orig_mod"):
                self.tokenizer = self.tokenizer._orig_mod

        else:
            self.tokenizer = BrainOmniCausalTokenizer(**tokenizer_kwargs)

        # freeze tokenizer during autoregressive training (optional)
        if self.freeze_tokenizer:
            for p in self.tokenizer.parameters():
                p.requires_grad_(False)

        self.projection = nn.Linear(n_dim, lm_dim) if n_dim != lm_dim else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                SpatialTemporalAttentionBlock(lm_dim, lm_head, lm_dropout, causal=True)
                for _ in range(lm_depth)
            ]
        )
        self.predict_head = nn.Linear(
            lm_dim, self.num_quantizers_used * tokenizer_kwargs.get("codebook_size")
        )
        # self.apply(self._init_weights)

    # ----------------------------- helpers ----------------------------- #
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            if isinstance(m.weight, nn.Parameter):
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)

    def _channel_bias(self, embeddings: torch.Tensor) -> torch.Tensor:
        neuro = (
            self.tokenizer.encoder.neuros.type_as(embeddings)
            .detach()
            .view(1, -1, 1, embeddings.shape[-1])
        )
        return embeddings + neuro

    def _logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Args:

        embeddings: B C W D Returns:     logits: B C W Nq K
        """
        x = self.projection(self._channel_bias(embeddings))
        for block in self.blocks:
            x = block(x)

        logits = rearrange(
            self.predict_head(x),
            "B C W (N D) -> B C W N D",
            N=self.num_quantizers_used,
        )
        return logits

    def forward_token_sequence(
        self,
        token_seq: CausalTokenSequence,
        compute_targets: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run the AR head on a provided causal token sequence."""
        embeddings = token_seq.embeddings
        if self.freeze_tokenizer:
            embeddings = embeddings.detach()

        if compute_targets:
            context = embeddings[:, :, :-1]
        else:
            context = embeddings

        logits = self._logits_from_embeddings(context)
        output: Dict[str, torch.Tensor] = {"logits_full": logits}

        if compute_targets:
            target_idx = token_seq.indices[..., : self.num_quantizers_used]
            output["logits"] = logits
            output["targets"] = target_idx[:, :, 1:]

        return output

    # ----------------------------- forward ----------------------------- #
    def forward(
        self,
        x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pos: torch.Tensor | None = None,
        sensor_type: torch.Tensor | None = None,
        **kwargs,
    ):
        """Args:

        x: raw MEG (B, C, T) or tuple (x, pos, sensor_type)
        """
        if self.freeze_tokenizer:
            self.tokenizer.eval()

        overlap = float(kwargs.get("overlap_ratio", self.overlap_ratio))
        if isinstance(x, (tuple, list)):
            x, pos, sensor_type = x  # type: ignore[misc]
        if pos is None or sensor_type is None:
            raise ValueError("pos and sensor_type must be provided for BrainOmni.")

        if self.freeze_tokenizer:
            with torch.no_grad():
                token_seq, commitment = self.tokenizer.tokenize(
                    x, pos, sensor_type, overlap_ratio=overlap
                )
        else:
            token_seq, commitment = self.tokenizer.tokenize(
                x, pos, sensor_type, overlap_ratio=overlap
            )

        outputs = self.forward_token_sequence(token_seq, compute_targets=True)
        outputs["token_seq"] = token_seq

        if not self.freeze_tokenizer:
            outputs["commitment_loss"] = commitment
        return outputs

    # ----------------------------- generation ----------------------------- #
    @torch.inference_mode()
    def forecast(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        steps: int,
        *,
        overlap_ratio: float | None = None,
        sampling: str = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressively extend the sequence by *steps* latent tokens and decode back
        to MEG space using the tokenizer decoder."""
        self.eval()
        overlap = float(
            overlap_ratio if overlap_ratio is not None else self.overlap_ratio
        )

        token_seq, _ = self.tokenizer.tokenize(
            x, pos, sensor_type, overlap_ratio=overlap
        )
        seq_indices = token_seq.indices[..., : self.num_quantizers_used]

        for _ in range(int(steps)):
            emb = self.tokenizer.indices_to_embeddings(
                seq_indices, token_seq.tokens_per_window
            )
            emb_flat = rearrange(emb, "B C N T D -> B C (N T) D")
            tmp_seq = CausalTokenSequence(
                embeddings=emb_flat,
                indices=seq_indices,
                tokens_per_window=token_seq.tokens_per_window,
                num_windows=math.ceil(emb_flat.shape[2] / token_seq.tokens_per_window),
                overlap_ratio=overlap,
            )
            logits = self._logits_from_embeddings(tmp_seq.embeddings)
            next_logits = logits[:, :, -1]

            next_indices = []
            for q in range(self.num_quantizers_used):
                next_indices.append(
                    sample_logits(
                        next_logits[..., q, :],
                        strategy=sampling,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                )
            next_indices = torch.stack(next_indices, dim=-1)
            seq_indices = torch.cat([seq_indices, next_indices.unsqueeze(2)], dim=2)

        decode_indices = self._restore_full_indices(seq_indices, token_seq)
        windows = self.tokenizer.decode_windows(
            decode_indices,
            pos,
            sensor_type,
            tokens_per_window=token_seq.tokens_per_window,
        )
        stride = self.tokenizer._stride(overlap)
        return self.tokenizer.overlap_add(windows, stride=stride)

    def _restore_full_indices(
        self, seq_indices: torch.Tensor, token_seq: CausalTokenSequence
    ) -> torch.Tensor:
        """Bring truncated quantizer predictions back to full RVQ depth for decoding."""
        total_q = self.tokenizer.quantizer.rvq.num_quantizers
        if seq_indices.shape[-1] == total_q:
            return seq_indices

        used_q = seq_indices.shape[-1]
        base = token_seq.indices
        if base.shape[2] < seq_indices.shape[2]:
            pad_len = seq_indices.shape[2] - base.shape[2]
            pad = base[..., -1:, :].expand(
                base.shape[0], base.shape[1], pad_len, base.shape[-1]
            )
            base = torch.cat([base, pad], dim=2)

        base = base[..., : seq_indices.shape[2], :].clone()
        base[..., :used_q] = seq_indices
        return base
