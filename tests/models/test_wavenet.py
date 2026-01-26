import torch

from brain_gen.models.wavenet import Wavenet3D, WavenetFullChannel
from tests.models.utils import assert_future_grad_zero


def test_grad_causality_wavenetfullchannel():
    # Use full WavenetFullChannel and verify causal gradients by checking that
    # embeddings used only at the last timestep receive zero gradient when the
    # loss excludes the last timestep.

    B, C, T = 1, 3, 200
    Q = 256  # quantization levels

    model = WavenetFullChannel(
        in_channels=C,
        head_channels=16,
        kernel_size=2,
        dilations=[1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128],
        quant_levels=Q,
        quant_emb=16,
        residual_channels=16,
        dilation_channels=16,
        skip_channels=16,
        cond_channels=None,
        p_drop=0.0,
    )

    # Build input token indices: early timesteps draw from low range [0, Q//2),
    # the final timestep uses a code (Q-1) that does not appear earlier.
    x = torch.randint(0, Q // 2, (B, C, T))
    x[:, :, -1] = Q - 1

    out, emb = model(x, condition=None, causal_pad=True, test_mode=True)  # (B, C, T, Q)

    loss = out[:, :, :-1, :].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


def test_grad_causality_wavenet3D():
    # Use full WavenetFullChannel and verify causal gradients by checking that
    # embeddings used only at the last timestep receive zero gradient when the
    # loss excludes the last timestep.

    B, H, W, T = 1, 32, 32, 200
    Q = 128  # quantization levels

    model = Wavenet3D(
        quant_emb=8,
        quant_levels=Q,
        head_channels=16,
        kernel_size=2,
        dilations=[1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128],
        residual_channels=16,
        dilation_channels=16,
        skip_channels=16,
        spatial_kernel_size=3,
        spatial_dilation=1,
    )

    # Build input token indices: early timesteps draw from low range [0, Q//2),
    # the final timestep uses a code (Q-1) that does not appear earlier.
    x = torch.randint(0, Q, (B, H, W, T))

    out, emb = model(x, condition=None, causal_pad=True, test_mode=True)

    loss = out[:, :, :, :-1, :].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)
