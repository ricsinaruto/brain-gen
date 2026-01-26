import torch
from torch import nn
import numpy as np
from ..layers.embeddings import Embeddings
from ..layers.st_blocks import (  # noqa: F401
    STGPTBlock,
    STGPTBlockParallel,
    STBlock,
    STConvBlock,
)


class STGPT2MEG(nn.Module):
    def __init__(
        self,
        num_channels: int,
        vocab_size: int,
        d_model: int,
        layers: int,
        trf_args: dict,
        embedding_args: dict = None,
        trf_block: str = "STGPTBlock",
    ):
        super().__init__()
        embedding_args = embedding_args or {}
        self.num_channels = num_channels
        self.quant_levels = vocab_size

        trf_class = globals()[trf_block]

        blocks = [trf_class(**trf_args) for _ in range(layers)]
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.RMSNorm(d_model)
        self.embeddings = Embeddings(
            num_channels,
            quant_levels=vocab_size,
            quant_emb=d_model,
            channel_emb=d_model,
            **embedding_args,
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        chid: np.ndarray = None,
        cond: torch.Tensor = None,
        sid: torch.Tensor = None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        x = self.embeddings(x, chid, cond, sid)

        _, T, E = x.shape
        x = x.reshape(-1, self.num_channels, T, E)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if not return_logits:
            return x

        x = self.head(x)
        return x  # (B, C, T, Q)
