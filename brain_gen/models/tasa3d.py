import torch
import torch.nn as nn

from ..layers.st_blocks import TASA3DBlock
from ..layers.conv import time_group_norm


class TASA3D(nn.Module):
    """Stacks TASA-3D blocks; outputs categorical(256) logits per pixel.

    Causality: only temporal attention mixes in T; all spatial ops use k_t=1.
    """

    def __init__(
        self,
        input_hw: tuple[int, int],
        emb_dim: int = 16,
        quant_levels: int = 256,
        depth: int = 4,
        num_down: int = 3,
        channel_grow: int = 2,
        heads: int = 8,
        drop: float = 0.0,
        rope: bool = True,
        n_cond_tok: int = 0,
        spatial_emb: bool = False,
    ):
        super().__init__()
        self.token = nn.Embedding(quant_levels, emb_dim)
        self.blocks = nn.ModuleList(
            [
                TASA3DBlock(
                    C_in=emb_dim,
                    input_hw=input_hw,
                    num_down=num_down,
                    channel_grow=channel_grow,
                    heads=heads,
                    rope=rope,
                    drop=drop,
                    use_spatial_emb=spatial_emb,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.GroupNorm(1, emb_dim)
        self.head = nn.Conv3d(emb_dim, quant_levels, 1)  # categorical logits

        self.n_cond_tok = n_cond_tok
        if n_cond_tok > 0:
            self.emb_tok = nn.Embedding(n_cond_tok, emb_dim)

    def _apply_conditioning(self, x: torch.Tensor, cond: torch.Tensor):
        """Args:

        x: [B,H,W,T,D]     cond: [B,1,1,T] Returns:     [B,H,W,T,D]
        """
        B, H, W, T, D = x.shape

        # expand cond to [B,H,W,T]
        cond = cond.expand(-1, H, W, -1)  # B x H x W x T
        cond = self.emb_tok(cond)
        return x + cond

    def forward(
        self, x_tokens: torch.Tensor, embeds: torch.Tensor = None
    ) -> torch.Tensor:  # [B,H,W,T] ints
        cond = None
        if isinstance(x_tokens, tuple) or isinstance(x_tokens, list):
            x_tokens, cond = x_tokens

        if embeds is None:
            x = self.token(x_tokens)  # [B,H,W,T,D]
        else:
            x = embeds

        if cond is not None and self.n_cond_tok > 0:
            x = self._apply_conditioning(x, cond)

        x = x.permute(0, 4, 3, 1, 2).contiguous()  # [B,D,T,H,W]

        for blk in self.blocks:
            x = blk(x)  # temporal-only attention inside
        x = time_group_norm(self.norm, x)
        logits = self.head(x)  # [B,256,T,H,W]
        return logits.permute(0, 3, 4, 2, 1).contiguous()  # [B,H,W,T,256]
