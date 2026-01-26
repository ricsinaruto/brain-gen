import torch

from brain_gen.models.ntd import NTD
from tests.models.utils import assert_future_grad_zero


@torch.no_grad()
def test_ntd_forward_and_loss_call():
    B, C, L = 2, 3, 64
    x = torch.randn(B, C, L)

    model = NTD(
        signal_length=L,
        signal_channel=C,
        diffusion_time_steps=8,
        net_hidden_channel=4,
        net_num_scales=2,
        net_num_blocks=1,
    )
    noise, pred_noise, mask = model(x)
    assert noise.shape == pred_noise.shape

    # loss should reduce
    loss = model.loss(noise, pred_noise, mask)
    assert loss.ndim == 0


@torch.no_grad()
def test_ntd_forecast_shape_and_mask_respected():
    B, C, Lp, horizon = 1, 2, 64, 8
    model = NTD(
        signal_length=Lp + horizon,
        signal_channel=C,
        diffusion_time_steps=4,
        net_hidden_channel=4,
        net_num_scales=3,
        net_num_blocks=1,
        mask_channel=1,
    )
    past = torch.randn(B, C, Lp)
    # cond with Cc=0 (None) works; ensure mask channel increases conv input
    seq = model.forecast(past, horizon=horizon)
    assert seq.shape == (B, C, horizon)


@torch.no_grad()
def test_ntd_forecast_eval_runner_signature():
    B, C, Lp, horizon = 1, 2, 32, 5
    model = NTD(
        signal_length=40,
        signal_channel=C,
        diffusion_time_steps=4,
        net_hidden_channel=4,
        net_num_scales=2,
        net_num_blocks=1,
        mask_channel=1,
    )
    past = torch.randn(B, C, Lp)
    dummy_pos = torch.zeros(B, C, 2)
    dummy_type = torch.zeros(B, C, dtype=torch.long)

    seq = model.forecast(
        (past, dummy_pos, dummy_type),
        horizon,
        sample_fn=lambda x: x,
        sliding_window_overlap=0.25,
        use_cache=True,
    )
    assert seq.shape == (B, C, horizon)


def test_grad_causality_ntd():
    B, C, T = 1, 272, 200
    x = torch.randn(B, C, T, requires_grad=True)
    model = NTD(
        signal_length=T,
        signal_channel=C,
        diffusion_time_steps=8,
        net_hidden_channel=4,
        net_num_scales=1,
        net_num_blocks=3,
        net_slconv_kernel_size=3,
        p_forecast=1.0,
        mask_channel=1,
    )
    noise, pred_noise, mask = model(x)

    loss = model.loss(noise, pred_noise, mask, reduce="none")
    loss[..., :-1].mean().backward()

    assert_future_grad_zero(x, T - 1)
