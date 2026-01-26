import torch

from brain_gen.models import TASA3D
from tests.models.utils import assert_future_grad_zero


def test_grad_causality_tasa3d():
    B, H, W, T, E = 1, 16, 16, 50, 16
    # Random differentiable embeddings at block input
    emb = torch.randn(B, H, W, T, E, requires_grad=True)
    x_int = torch.randint(0, 256, (B, H, W, T))

    model = TASA3D(
        emb_dim=E,
        input_hw=(H, W),
        depth=4,
        num_down=3,
        channel_grow=2,
    )

    model.eval()
    y = model(x_int, embeds=emb)

    loss = y[..., :-1, :].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)
