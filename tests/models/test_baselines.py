import torch

from brain_gen.models.baselines import (
    CNNMultivariate,
    CNNUnivariate,
    CNNMultivariateQuantized,
    CNNUnivariateQuantized,
)


def test_cnn_multivariate_causal_prefix_invariance():
    B, C, T = 1, 4, 32
    num_layers = 2
    ksize = 5
    model = CNNMultivariate(
        in_channels=C, num_layers=num_layers, hidden_channels=8, kernel_size=ksize
    )
    x = torch.randn(B, C, T)
    y_a = model(x)

    # change future after t0; earlier outputs should not change
    t0 = T // 2
    x_b = x.clone()
    x_b[..., t0:] = torch.randn_like(x_b[..., t0:])
    y_b = model(x_b)
    # Allow for receptive-field margin
    margin = num_layers * (ksize - 1)
    if t0 - margin > 0:
        assert torch.allclose(y_a[..., : t0 - margin], y_b[..., : t0 - margin])


def test_cnn_univariate_shapes_and_causality():
    B, C, T = 2, 3, 24
    num_layers = 1
    ksize = 3
    model = CNNUnivariate(num_layers=num_layers, hidden_channels=4, kernel_size=ksize)
    x = torch.randn(B, C, T)
    y = model(x)
    assert y.shape == (B, C, T)

    t0 = T // 3
    x2 = x.clone()
    x2[..., t0:] = torch.randn_like(x2[..., t0:])
    y2 = model(x2)
    margin = num_layers * (ksize - 1)
    if t0 - margin > 0:
        assert torch.allclose(y[..., : t0 - margin], y2[..., : t0 - margin])


def test_cnn_quantized_shapes():
    B, C, T = 2, 3, 16
    V = 32
    x = torch.randint(0, V, (B, C, T))
    mv = CNNMultivariateQuantized(in_channels=C, num_classes=V, num_embeddings=2)
    y = mv(x)
    assert y.shape == (B, C, T, V)

    uv = CNNUnivariateQuantized(num_classes=V, num_embeddings=2)
    y2 = uv(x)
    assert y2.shape == (B, C, T, V)
