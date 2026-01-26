# ADAPTED FROM: https://github.com/ricsinaruto/MEG-transfer-decoding

import torch
from torch import nn
from typing import List, Optional

import torch.nn.functional as F


def wave_init_weights(m):
    """Initialize conv1d with Xavier_uniform weight and 0 bias."""
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias, mean=1e-3, std=1e-2)


class WavenetLogitsHead(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        residual_channels: int,
        head_channels: int,
        out_channels: int,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """Collates skip results and transforms them to logit predictions.

        Args:     skip_channels: number of skip channels     residual_channels: number
        of residual channels     head_channels: number of hidden channels to compute
        result     out_channels: number of output channels     bias: When true,
        convolutions use a bias term.
        """
        del residual_channels
        super().__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Dropout1d(p=dropout),
            torch.nn.LeakyReLU(),  # note, we perform non-lin first (on sum of skips)
            torch.nn.Conv1d(
                skip_channels,
                head_channels,
                kernel_size=1,
                bias=bias,
            ),  # enlarge and squeeze (not based on paper)
            torch.nn.Dropout1d(p=dropout),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                head_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
            ),  # logits
        )

    def forward(self, encoded, skips):
        """Compute logits from WaveNet layer results.

        Args:     encoded: unused last residual output of last layer     skips: list of
        skip connections of shape (B,C,T) where C is         the number of skip
        channels. Returns:     logits: (B,Q,T) tensor of logits, where Q is the number
        of output     channels.
        """
        del encoded
        return self.transform(sum(skips))


class WavenetLayer(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        dilation: int,
        dilation_channels: int,
        residual_channels: int,
        skip_channels: int,
        shift: int = 0,
        dropout: float = 0.0,
        cond_channels: Optional[int] = None,
        in_channels=None,
        bias=False,
    ):
        super(WavenetLayer, self).__init__()

        in_channels = in_channels or residual_channels
        self.shift = shift
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_channels = in_channels
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.cond_channels = cond_channels

        self.causal_left_pad = (kernel_size - 1) * dilation + shift

        self.conv_dilation = nn.Conv1d(
            in_channels,
            2 * dilation_channels,  # We stack W f,k and W g,k, similar to PixelCNN
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )

        self.conv_res = nn.Conv1d(
            dilation_channels,
            residual_channels,
            kernel_size=1,
            bias=bias,
        )
        self.conv_skip = nn.Conv1d(
            dilation_channels,
            skip_channels,
            kernel_size=1,
            bias=bias,
        )

        self.conv_cond = None
        if cond_channels is not None:
            self.conv_cond = nn.Conv1d(
                cond_channels,
                dilation_channels * 2,
                kernel_size=1,
                bias=bias,
            )

        self.conv_input = None
        if in_channels != residual_channels:
            self.conv_input = nn.Conv1d(
                in_channels,
                residual_channels,
                kernel_size=1,
                bias=bias,
            )

        self.dropout = torch.nn.Dropout1d(p=dropout)

    def forward(self, x, c, causal_pad=True):
        """Compute residual and skip output from inputs x.

        Args:     x: (B,C,T) tensor where C is the number of residual channels when
        `in_channels` was specified the number of input channels     c: optional tensor
        containing a global (B,C,1) or local (B,C,T)         condition, where C is the
        number of condition channels.     causal_pad: layer performs causal padding when
        set to True, otherwise         assumes the input is already properly padded.
        Returns     r: (B,C,T) tensor where C is the number of residual channels skip:
        (B,C,T) tensor where C is the number of skip channels
        """
        p = (self.causal_left_pad, 0) if causal_pad else (0, 0)
        x_dilated = self.conv_dilation(F.pad(x, p))

        if self.cond_channels:
            assert c is not None, "conditioning required"
            x_cond = self.conv_cond(c[:, :, -x_dilated.shape[-1] :])
            x_dilated = x_dilated + x_cond
        x_filter = torch.tanh(x_dilated[:, : self.dilation_channels])
        x_gate = torch.sigmoid(x_dilated[:, self.dilation_channels :])
        x_h = x_gate * x_filter
        skip = self.conv_skip(x_h)
        res = self.conv_res(x_h)

        if self.conv_input is not None:
            x = self.conv_input(x)  # convert to res channels

        if causal_pad:
            out = x + res
        else:
            out = x[..., -res.shape[-1] :] + res

        # dropout
        out = self.dropout(out)

        # need to keep only second half of skips
        return out, skip[:, :, -self.shift :]


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        quant_emb: int,
        quant_levels: int,
        num_classes: int = 0,
        class_emb: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        self.quant_levels = quant_levels

        # embeddings for various conditioning
        self.cond_emb = (
            nn.Embedding(num_classes, class_emb) if num_classes > 0 else None
        )

        # 306 quantazation embedding layers
        self.quant_emb = torch.randn(
            size=(in_channels, self.quant_levels, quant_emb),
            dtype=torch.float32,
            requires_grad=True,
            device=device,
        )
        self.quant_emb = torch.nn.Parameter(self.quant_emb)

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None):
        B, C, T = x.shape

        cond = None
        if condition is not None:
            try:
                inds = torch.squeeze(condition, dim=1)
                cond = self.cond_emb(inds).permute(0, 2, 1)
            except RuntimeError:
                print(condition.shape)
                print(x.shape)
                print(inds.shape)
                raise

            # set elements of cond to 0 where cond_ind is 0
            cond = cond * (condition > 0).float()

            # repeat cond across new args.channels dim
            cond = cond.unsqueeze(1).repeat(1, C, 1, 1)
            cond = cond.reshape(-1, cond.shape[-2], cond.shape[-1])

        # apply embedding to each channel separately
        x = x.permute(0, 2, 1)  # B x T x C
        x = x.reshape(-1, x.shape[-1])  # B*T x C
        x = self.quant_emb[torch.arange(x.shape[-1]), x]  # B*T x C x E

        return x, cond


class WavenetFullChannel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        kernel_size: int,
        dilations: List[int],
        quant_levels: int,
        quant_emb: int,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        cond_channels: int,
        num_classes: int = 0,
        class_emb: int = 0,
        p_drop: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()

        self.embedding_layer = EmbeddingLayer(
            in_channels=in_channels,
            quant_emb=quant_emb,
            quant_levels=quant_levels,
            num_classes=num_classes,
            class_emb=class_emb,
            device=device,
        )

        # initial convolution
        layers = [
            WavenetLayer(
                kernel_size=1,
                dilation=1,
                in_channels=quant_emb,
                residual_channels=residual_channels,
                dilation_channels=dilation_channels,
                skip_channels=skip_channels,
                cond_channels=cond_channels,
                bias=True,
                dropout=p_drop,
            )
        ]

        layers += [
            WavenetLayer(
                kernel_size=kernel_size,
                dilation=d,
                residual_channels=residual_channels,
                dilation_channels=dilation_channels,
                skip_channels=skip_channels,
                cond_channels=cond_channels,
                bias=True,
                dropout=p_drop,
            )
            for d in dilations
        ]

        self.layers = torch.nn.ModuleList(layers)

        self.logits = WavenetLogitsHead(
            skip_channels=skip_channels,
            residual_channels=residual_channels,
            head_channels=head_channels,
            out_channels=quant_levels,
            bias=True,
            dropout=p_drop,
        )

        self.apply(wave_init_weights)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        causal_pad: bool = True,
        test_mode: bool = False,
    ) -> torch.Tensor:
        """Computes logits and encoding results from observations.

        Args:     x: (B,T) or (B,Q,T) tensor containing observations     c: optional
        conditioning Tensor. (B,C,1) for global conditions,         (B,C,T) for local
        conditions. None if unused     causal_pad: Whether or not to perform causal
        padding.     test_mode: Whether to return the encoded embeddings. Returns:
        logits: (B,Q,T) tensor of logits. Note that the t-th temporal output represents
        the distribution over t+1.     encoded: same as `.encode`.
        """
        B, C, T = x.shape

        x, cond = self.embedding_layer(x, condition)

        # reshape back
        x = x.reshape(-1, T, x.shape[-2], x.shape[-1])  # B x T x C x E
        x = x.permute(0, 2, 3, 1)  # B x C x E x T
        x = x.reshape(-1, x.shape[-2], x.shape[-1])  # B*C x E x T

        if test_mode:
            x.retain_grad()

        skips = []
        for layer in self.layers:
            out, skip = layer(x, c=cond, causal_pad=causal_pad)
            skips.append(skip)

        if out.shape[-1] != skip.shape[-1]:
            print(out.shape)
            print(skip.shape)

        out = self.logits(out, skips)

        # reshape to get (B*C, Q, T) -> (B, C, T, Q)
        out = out.reshape(-1, C, out.shape[-2], out.shape[-1])
        out = out.permute(0, 1, 3, 2).contiguous()

        if test_mode:
            return out, x  # (B, C, T, Q)

        return out  # (B, C, T, Q)


class Wavenet3DLogitsHead(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        head_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Dropout3d(p=dropout),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(skip_channels, head_channels, kernel_size=1, bias=bias),
            torch.nn.Dropout3d(p=dropout),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(head_channels, out_channels, kernel_size=1, bias=bias),
        )

    def forward(self, skips: List[torch.Tensor]) -> torch.Tensor:
        return self.transform(sum(skips))


class Wavenet3DLayer(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        dilation: int,
        dilation_channels: int,
        residual_channels: int,
        skip_channels: int,
        shift: int = 0,
        dropout: float = 0.0,
        cond_channels: Optional[int] = None,
        in_channels: Optional[int] = None,
        bias: bool = True,
        spatial_kernel: tuple[int, int] = (1, 1),
        spatial_dilation: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        in_channels = in_channels or residual_channels

        # Causal padding along time only
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal_left_pad = (kernel_size - 1) * dilation + shift

        self.in_channels = in_channels
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.cond_channels = cond_channels
        self.spatial_kernel = spatial_kernel
        self.spatial_dilation = spatial_dilation

        # Dilated temporal conv
        k_d, k_h = self.spatial_kernel
        d_d, d_h = self.spatial_dilation
        # padding to preserve spatial sizes
        pad_d = ((k_d - 1) * d_d) // 2
        pad_h = ((k_h - 1) * d_h) // 2

        self.conv_dilation = nn.Conv3d(
            in_channels,
            2 * dilation_channels,
            kernel_size=(k_d, k_h, kernel_size),
            dilation=(d_d, d_h, dilation),
            padding=(pad_d, pad_h, 0),
            bias=bias,
        )

        self.conv_res = nn.Conv3d(
            dilation_channels, residual_channels, kernel_size=1, bias=bias
        )
        self.conv_skip = nn.Conv3d(
            dilation_channels, skip_channels, kernel_size=1, bias=bias
        )

        self.conv_cond = None
        if cond_channels is not None:
            self.conv_cond = nn.Conv3d(
                cond_channels, 2 * dilation_channels, kernel_size=1, bias=bias
            )

        self.conv_input = None
        if in_channels != residual_channels:
            self.conv_input = nn.Conv3d(
                in_channels, residual_channels, kernel_size=1, bias=bias
            )

        self.dropout = torch.nn.Dropout3d(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor],
        causal_pad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:

        x: (B, C, H, W, T) c: optional conditioning, (B, Cc, H, W, T) or broadcastable
        causal_pad: whether to apply causal left padding in time Returns: residual_out:
        (B, C_res, H, W, T)     skip: (B, C_skip, H, W, T)
        """
        if causal_pad:
            # pad order for 5D input: (W_left, W_right, H_left,
            # H_right, D_left, D_right); here we treat temporal dim
            # as "W"
            p = (self.causal_left_pad, 0)
            x_dilated = self.conv_dilation(F.pad(x, p))
        else:
            x_dilated = self.conv_dilation(x)

        if self.cond_channels is not None:
            assert c is not None, "conditioning required"
            # match time length of x_dilated
            t = x_dilated.shape[-1]
            c_proj = self.conv_cond(c)[..., -t:]
            x_dilated = x_dilated + c_proj

        x_filter = torch.tanh(x_dilated[:, : self.dilation_channels])
        x_gate = torch.sigmoid(x_dilated[:, self.dilation_channels :])
        x_h = x_gate * x_filter

        skip = self.conv_skip(x_h)
        res = self.conv_res(x_h)

        if self.conv_input is not None:
            x = self.conv_input(x)

        if causal_pad:
            out = x + res
        else:
            out = x[..., -res.shape[-1] :] + res

        out = self.dropout(out)
        return out, skip


class Wavenet3D(nn.Module):
    """WaveNet-style model operating on 3D volumes over time.

    Expects inputs shaped as (B, C, H, W, T), where C is the channel/embedding
    dimension. The network performs causal convolutions along the temporal axis using 3D
    convolutions with kernel size (1, 1, k), so spatial dimensions are preserved while
    receptive field grows only in time.

    The model mirrors the gating, residual, and skip-connection structure of a classical
    WaveNet, but generalized to 3D.
    """

    def __init__(
        self,
        quant_emb: int,
        quant_levels: int,
        head_channels: int,
        kernel_size: int,
        dilations: List[int],
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        cond_channels: Optional[int] = None,
        p_drop: float = 0.0,
        bias: bool = True,
        spatial_kernel_size: int | tuple[int, int] = 1,
        spatial_dilation: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.cond_channels = cond_channels

        # normalize spatial kernel/dilation to tuples (kH, kW) and (dH, dW)
        if isinstance(spatial_kernel_size, int):
            spatial_kernel_size = (spatial_kernel_size, spatial_kernel_size)
        if isinstance(spatial_dilation, int):
            spatial_dilation = (spatial_dilation, spatial_dilation)

        self.spatial_kernel_size = spatial_kernel_size
        self.spatial_dilation = spatial_dilation

        self.embedding = nn.Embedding(quant_levels, quant_emb)

        # construct layers: initial 1x1x1 to get into residual space,
        # then dilated blocks
        layers: List[nn.Module] = [
            Wavenet3DLayer(
                kernel_size=1,
                dilation=1,
                in_channels=quant_emb,
                residual_channels=residual_channels,
                dilation_channels=dilation_channels,
                skip_channels=skip_channels,
                cond_channels=cond_channels,
                bias=bias,
                dropout=p_drop,
                spatial_kernel=(1, 1),
                spatial_dilation=(1, 1),
            )
        ]

        layers += [
            Wavenet3DLayer(
                kernel_size=kernel_size,
                dilation=d,
                residual_channels=residual_channels,
                dilation_channels=dilation_channels,
                skip_channels=skip_channels,
                cond_channels=cond_channels,
                bias=bias,
                dropout=p_drop,
                spatial_kernel=self.spatial_kernel_size,
                spatial_dilation=self.spatial_dilation,
            )
            for d in dilations
        ]

        self.layers = nn.ModuleList(layers)
        self.logits = Wavenet3DLogitsHead(
            skip_channels=skip_channels,
            head_channels=head_channels,
            out_channels=quant_levels,
            dropout=p_drop,
            bias=bias,
        )

        self.apply(wave_init_weights)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        causal_pad: bool = True,
        test_mode: bool = False,
    ) -> torch.Tensor:
        """Args:

        x: (B, H, W, T) condition: optional conditioning tensor broadcastable to (B, Cc,
        H, W, T) causal_pad: whether to apply causal left padding on temporal axis
        Returns:     logits: (B, out_channels, H, W, T)
        """

        x = self.embedding(x)  # (B, H, W, T, E)
        x = x.permute(0, 4, 1, 2, 3)  # (B, E, H, W, T)

        if test_mode:
            x.retain_grad()

        cond = None
        if self.cond_channels is not None:
            assert (
                condition is not None
            ), "conditioning tensor required when cond_channels is set"
            cond = condition

        x_out = x
        skips: List[torch.Tensor] = []
        for layer in self.layers:
            x_out, skip = layer(x_out, c=cond, causal_pad=causal_pad)
            skips.append(skip)

        out = self.logits(skips)

        out = out.permute(0, 2, 3, 4, 1).contiguous()  # (B, H, W, T, Q)

        if test_mode:
            return out, x  # (B, H, W, T, Q)

        return out  # (B, H, W, T, Q)
