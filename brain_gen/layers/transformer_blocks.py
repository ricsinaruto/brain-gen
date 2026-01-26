import torch
import torch.nn as nn


from .attention import AttentionBlock
from .mlp import MLP, MLPMoE


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "standard",  # must be either "standard" or "gpt_oss"
        mlp_type: str = "standard",  # must be either "standard" or "moe"
    ):
        super().__init__()
        self.attn = AttentionBlock(attn_type=attn_type, attn_args=attn_args)

        if mlp_type == "standard":
            self.mlp = MLP(**mlp_args)
        elif mlp_type == "moe":
            self.mlp = MLPMoE(**mlp_args)
        else:
            raise ValueError(f"Invalid MLP type: {mlp_type}")

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        x = self.attn(x, causal=causal)
        x = self.mlp(x)
        return x


class TransformerBlockCond(torch.nn.Module):
    def __init__(
        self,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "conditioned",
        mlp_type: str = "standard",
        n_cond_tok: int = 0,
        n_cond_global: int = 0,
        d_tok_emb: int = 0,
        d_glob_emb: int = 0,
    ):
        super().__init__()
        self.n_cond_tok = n_cond_tok
        self.n_cond_global = n_cond_global

        if n_cond_tok > 0:
            self.emb_tok = nn.Embedding(n_cond_tok, d_tok_emb)
            # map to FiLM params (per-token & global)
            self.mod_tok = nn.Sequential(
                nn.SiLU(), nn.Linear(d_tok_emb, 4 * attn_args["d_model"])
            )  # γ1t,β1t,γ2t,β2t  (per token)

        if n_cond_global > 0:
            self.emb_glob = nn.Embedding(n_cond_global, d_glob_emb)
            self.mod_glb = nn.Sequential(
                nn.SiLU(), nn.Linear(d_glob_emb, 4 * attn_args["d_model"])
            )  # γ1g,β1g,γ2g,β2g  (per sequence)

        attn_args["d_tok_emb"] = d_tok_emb
        attn_args["n_cond_tok"] = n_cond_tok
        self.attn = AttentionBlock(attn_type=attn_type, attn_args=attn_args)

        if mlp_type == "standard":
            self.mlp = MLP(**mlp_args)
        elif mlp_type == "moe":
            self.mlp = MLPMoE(**mlp_args)
        else:
            raise ValueError(f"Invalid MLP type: {mlp_type}")

    def forward(
        self,
        x: torch.Tensor,
        c_tok_ids: torch.Tensor = None,
        c_global_ids: torch.Tensor = None,
        causal: bool = False,
    ) -> torch.Tensor:
        # initialize FiLM params
        g1t, b1t, g2t, b2t = 0, 0, 0, 0
        g1g, b1g, g2g, b2g = 0, 0, 0, 0

        # ---- FiLM params (sum of per-token + broadcasted global) ----
        if self.n_cond_tok > 0:
            mt = self.mod_tok(self.emb_tok(c_tok_ids))  # [B,S,4D]
            g1t, b1t, g2t, b2t = mt.chunk(4, dim=-1)  # each [B,S,D]

        if self.n_cond_global > 0:
            mg = self.mod_glb(self.emb_glob(c_global_ids))  # [B,4D]
            g1g, b1g, g2g, b2g = mg.chunk(4, dim=-1)  # each [B,D]
            g1g = g1g.unsqueeze(1)
            b1g = b1g.unsqueeze(1)
            g2g = g2g.unsqueeze(1)
            b2g = b2g.unsqueeze(1)

        # broadcast global to sequence and sum
        g1 = g1t + g1g
        b1 = b1t + b1g
        g2 = g2t + g2g
        b2 = b2t + b2g

        h = self.attn(x, causal=causal, residual=False, c_tok_ids=c_tok_ids)
        x = x + h * (1 + g1) + b1

        h = self.mlp(x, residual=False)
        x = x + h * (1 + g2) + b2

        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        attn_args: dict,
        mlp_args: dict,
        vocab_size: int,
        num_layers: int,
        attn_type: str = "standard",  # must be either "standard" or "gpt_oss"
        mlp_type: str = "standard",  # must be either "standard" or "moe"
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            vocab_size,
            attn_args["d_model"],
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(attn_args, mlp_args, attn_type, mlp_type)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(attn_args["d_model"])
        self.unembedding = torch.nn.Linear(
            attn_args["d_model"],
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        embeds: torch.Tensor = None,
        causal: bool = False,
        extra_embs: list[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        if embeds is None:
            x = self.embedding(x)
        else:
            x = embeds

        if extra_embs:
            x = x + sum(extra_embs)  # sum any extra embeddings (e.g., spatial, cond)

        for block in self.block:
            x = block(x, causal=causal)
        x = self.norm(x)
        x = self.unembedding(x)
        return x
