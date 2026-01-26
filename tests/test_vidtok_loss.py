import torch

from brain_gen.losses.vidtok import VidtokLoss


def test_vidtok_covariance_loss_zero_for_identity():
    torch.manual_seed(0)
    loss_fn = VidtokLoss(temporal_cov_weight=1.0)
    inputs = torch.randn(2, 1, 4, 4, 4)
    outputs = (inputs.clone(), {})
    loss = loss_fn(outputs, inputs)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
