import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from .attention import AttentionBlock, MultiHeadAttention, MultiHeadAttentionGPTOSS
from .transformer_blocks import MLP, MLPMoE
from .conv import Conv3dBlock, DownStage3D, UpStage3D, time_group_norm


class STBlock(nn.Module):
    def __init__(
        self,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "standard",
        mlp_type: str = "standard",
    ):
        super().__init__()
        dim = attn_args["d_model"]

        if attn_type == "standard":
            attn_class = MultiHeadAttention
        elif attn_type == "gpt_oss":
            attn_class = MultiHeadAttentionGPTOSS
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        self.norm1 = nn.RMSNorm(dim)
        self.spatial_attn = attn_class(**attn_args)

        self.temporal_norm1 = nn.RMSNorm(dim)
        self.temporal_attn = attn_class(**attn_args)
        self.temporal_fc = nn.Linear(dim, dim)

        self.norm2 = nn.RMSNorm(dim)

        if mlp_type == "standard":
            self.mlp = MLP(**mlp_args)
        elif mlp_type == "moe":
            self.mlp = MLPMoE(**mlp_args)
        else:
            raise ValueError(f"Invalid MLP type: {mlp_type}")

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, _ = x.shape

        # Temporal
        xt = x
        xt = rearrange(xt, "b c t m -> (b c) t m")
        xt = self.temporal_norm1(xt)
        res_temporal = self.temporal_attn(xt, xt, xt, causal=True)
        res_temporal = rearrange(res_temporal, "(b c) t m -> b c t m", b=B, c=C, t=T)
        res_temporal = self.temporal_fc(res_temporal)
        xt = x + res_temporal

        # Spatial
        xs = xt
        xs = rearrange(xs, "b c t m -> (b t) c m", b=B, c=C, t=T)
        xs = self.norm1(xs)
        res_spatial = self.spatial_attn(xs, xs, xs, causal=False)
        res_spatial = rearrange(res_spatial, "(b t) c m -> b c t m", b=B, c=C, t=T)
        x = xt + res_spatial

        # Mlp
        x = x + self.mlp(self.norm2(x))
        return x


class STGPTBlock(nn.Module):
    def __init__(
        self,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "standard",
        mlp_type: str = "moe",
    ):
        super().__init__()

        self.temporal_attn = AttentionBlock(attn_type=attn_type, attn_args=attn_args)
        self.spatial_attn = AttentionBlock(attn_type=attn_type, attn_args=attn_args)
        if mlp_type == "standard":
            self.spatial_mlp = MLP(**mlp_args)
            self.temporal_mlp = MLP(**mlp_args)
        elif mlp_type == "moe":
            self.spatial_mlp = MLPMoE(**mlp_args)
            self.temporal_mlp = MLPMoE(**mlp_args)
        else:
            raise ValueError(f"Invalid MLP type: {mlp_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, _ = x.shape

        xt = rearrange(x, "b c t m -> (b c) t m")
        xt = self.temporal_attn(xt, causal=True)
        xt = self.temporal_mlp(xt)
        xt = rearrange(xt, "(b c) t m -> b c t m", b=B, c=C, t=T)

        xs = rearrange(xt, "b c t m -> (b t) c m", b=B, c=C, t=T)
        xs = self.spatial_attn(xs, causal=False)
        xs = self.spatial_mlp(xs)
        xs = rearrange(xs, "(b t) c m -> b c t m", b=B, c=C, t=T)

        return xs + x


class STGPTBlockParallel(nn.Module):
    def __init__(
        self,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "standard",
        mlp_type: str = "moe",
    ):
        super().__init__()
        self.temporal_attn = AttentionBlock(attn_type=attn_type, attn_args=attn_args)
        self.spatial_attn = AttentionBlock(attn_type=attn_type, attn_args=attn_args)

        if mlp_type == "standard":
            self.mlp = MLP(**mlp_args)
        elif mlp_type == "moe":
            self.mlp = MLPMoE(**mlp_args)
        else:
            raise ValueError(f"Invalid MLP type: {mlp_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, _ = x.shape

        xt = rearrange(x, "b c t m -> (b c) t m")
        xt = self.temporal_attn(xt, causal=True)
        xt = rearrange(xt, "(b c) t m -> b c t m", b=B, c=C, t=T)

        xs = rearrange(x, "b c t m -> (b t) c m", b=B, c=C, t=T)
        xs = self.spatial_attn(xs, causal=False)
        xs = rearrange(xs, "(b t) c m -> b c t m", b=B, c=C, t=T)

        x = xt + xs
        x = self.mlp(x)

        return x


class STConvBlock(STGPTBlock):
    def __init__(
        self,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "standard",
        mlp_type: str = "standard",
        image_size: int = 32,
        row_idx: torch.Tensor | None = None,
        col_idx: torch.Tensor | None = None,
        conv_kernel: tuple[int, int, int] = (3, 3, 3),
        dropout: float = 0.0,
    ):
        """Spatial‑Temporal GPT block with an extra causal 3D conv between temporal and
        spatial attention. The conv operates on a rasterised grid using the
        channel→(H,W) mapping from ChunkDatasetImage.

        Args:     attn_args: Arguments for AttentionBlock.     mlp_args: Arguments for
        MLP/MLPMoE.     attn_type: Attention implementation to use.     mlp_type:
        "standard" or "moe".     image_size: H and W of the raster grid.     row_idx:
        LongTensor of shape (C,) mapping channel→row.     col_idx: LongTensor of shape
        (C,) mapping channel→col.     conv_kernel: (kT, kH, kW) kernel size for 3D conv.
        conv_groups: Groups for 3D conv. If None, defaults to depthwise
        (groups=d_model).     dropout: Dropout after the conv block.
        """
        super().__init__(attn_args, mlp_args, attn_type, mlp_type)

        d_model = attn_args["d_model"]
        kT, kH, kW = conv_kernel
        self.kT, self.kH, self.kW = kT, kH, kW

        self.cb1 = Conv3dBlock(
            d_model, kT, kH, kW, image_size, dropout, row_idx, col_idx
        )
        # self.cb2 = Conv3dBlock(
        #    d_model, kT, kH, kW, image_size, conv_groups, dropout, row_idx, col_idx
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cb1(x)
        return super().forward(x)


class TASA3DBlock(nn.Module):
    """Temporal-only Attention with Spatial Autoencoder (no time mixing in the
    encoder/decoder). Workflow:

    x [B,C,T,H,W]     -> stem (1x1x1) to base C     -> (Down x N): stride (1,2,2),
    channels may grow     -> flatten per T: [B,T, C_bot*H'*W'] = [B,T,D_flat]     ->
    causal temporal MHA (+ token-MLP)     -> reshape to [B,C_bot,T,H',W']     -> (Up x
    N): stride (1,2,2) back to input spatial size, channels shrink     -> out_proj
    (1x1x1), residual add to input (with projection if needed) Constraints:   H and W
    must be divisible by 2**num_down.
    """

    def __init__(
        self,
        C_in,
        input_hw: tuple[int, int],
        num_down: int = 3,
        channel_grow: int = 2,  # multiply channels each downstage (e.g., 2x)
        heads: int = 8,
        rope: bool = True,
        drop: float = 0.0,
        token_mlp_ratio: float = 4.0,
        use_spatial_emb: bool = False,
    ):
        super().__init__()
        assert (
            isinstance(channel_grow, int) and channel_grow >= 1
        ), "channel_grow must be a positive integer (e.g., 2)"
        self.C_in = C_in
        self.H0, self.W0 = input_hw
        self.num_down = num_down
        self.g = channel_grow

        # Compute bottleneck spatial size
        s = 2**num_down
        assert (
            self.H0 % s == 0 and self.W0 % s == 0
        ), f"H0 and W0 must be divisible by 2**num_down (got {self.H0},{self.W0},"
        f"{num_down})"
        self.Hb, self.Wb = self.H0 // s, self.W0 // s

        Cb = C_in * self.g**num_down
        self.D_attn = Cb * self.Hb * self.Wb

        if use_spatial_emb:
            # Learned bias per bottleneck spatial token, shared across batch/time.
            self.spatial_emb = nn.Parameter(torch.randn(1, Cb, 1, self.Hb, self.Wb))
        else:
            self.register_parameter("spatial_emb", None)

        # Build channel schedule
        C_list = [C_in]
        for _ in range(num_down):
            C_list.append(C_list[-1] * self.g)  # exact since g is integer

        self.C_list = C_list  # [C0, C1, ..., Cb]

        # Downs, ups
        self.downs = nn.ModuleList(
            [DownStage3D(C_list[i], C_list[i + 1]) for i in range(num_down)]
        )
        self.ups = nn.ModuleList(
            [UpStage3D(C_list[i], C_list[i - 1]) for i in range(num_down, 0, -1)]
        )

        # print(f"Attn dim: {self.D_attn}")

        # Temporal attention at bottleneck (works directly on flattened h)
        self.attn = MultiHeadAttention(
            d_model=self.D_attn, nheads=heads, rope=rope, dropout=drop
        )
        D_mlp = int(self.D_attn * token_mlp_ratio)
        self.token_ffn = nn.Sequential(
            nn.Linear(self.D_attn, D_mlp),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(D_mlp, self.D_attn),
            nn.Dropout(drop),
        )

        # Full-res refinement & output
        self.norm_full = nn.GroupNorm(1, C_list[0])
        self.mlp_full = nn.Sequential(
            nn.Conv3d(C_list[0], C_list[0] * 4, 1),
            nn.GELU(),
            nn.Conv3d(C_list[0] * 4, C_list[0], 1),
            nn.Dropout(drop),
        )
        self.out_proj = nn.Conv3d(C_list[0], C_in, kernel_size=1, bias=False)

    def forward(self, x):  # x: [B,C_in,T,H0,W0]
        B, C, T, H, W = x.shape
        if (H, W) != (self.H0, self.W0):
            raise ValueError(
                f"Spatial size mismatch: expected {(self.H0, self.W0)}, got {(H, W)}"
            )

        # Encode: stride only in space
        h = x
        for down in self.downs:
            h = down(h)  # [B,Cb,T,Hb,Wb]
        B, Cb, T, Hb, Wb = h.shape
        assert (Cb, Hb, Wb) == (self.C_list[-1], self.Hb, self.Wb)

        if self.spatial_emb is not None:
            h = h + self.spatial_emb

        # Flatten bottleneck per timestep -> tokens [B,T,D_attn]
        tokens = h.permute(0, 2, 1, 3, 4).contiguous().view(B, T, self.D_attn)

        # Temporal-only transformer block
        z = tokens + self.attn(tokens, tokens, tokens, causal=True)
        z = z + self.token_ffn(z)

        # Unflatten back to bottleneck map
        h = (
            z.view(B, T, Cb, Hb, Wb).permute(0, 2, 1, 3, 4).contiguous()
        )  # [B,Cb,T,Hb,Wb]

        # Decode to full res
        for up in self.ups:
            h = up(h)  # [B,C0,T,H0,W0]

        # Full-res conv-MLP & residual
        h = h + self.mlp_full(time_group_norm(self.norm_full, h))
        y = self.out_proj(h) + x  # [B,C_in,T,H0,W0]
        return y
