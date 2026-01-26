import torch

from brain_gen.layers.quantizers import RVQRegularizer


def test_rvq_regularizer_returns_aux_loss():
    torch.manual_seed(0)
    reg = RVQRegularizer(
        dim=4,
        codebook_dim=4,
        codebook_size=8,
        num_quantizers=2,
    )
    z = torch.randn(2, 3, 4)
    _, reg_log = reg(z)

    assert "aux_loss" in reg_log
    assert "rvq_loss" not in reg_log
