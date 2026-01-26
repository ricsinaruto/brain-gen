import torch

from brain_gen.models.tokenizers.vidtok import Vidtok, VidtokRVQ
from tests.models.utils import assert_future_grad_zero


def _build_vidtok() -> Vidtok:
    encoder = {
        "double_z": False,
        "z_channels": 2,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 16,
        "ch_mult": (1, 2),
        "tempo_ds": [0],
        "tempo_us": [1],
        "time_downsample_factor": 2,
        "num_res_blocks": 1,
        "dropout": 0.0,
        "use_checkpoint": False,
        "init_pad_mode": "replicate",
        "norm_type": "groupnorm",
        "interpolation_mode": "nearest",
        "fix_encoder": False,
        "fix_decoder": False,
    }
    regularizer = {
        "levels": [4, 4],
        "entropy_loss_weight": 0.0,
        "commitment_loss_weight": 0.0,
    }
    return Vidtok(encoder=encoder, regularizer=regularizer)


def _build_vidtok_rvq() -> VidtokRVQ:
    encoder = {
        "double_z": False,
        "z_channels": 2,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 16,
        "ch_mult": (1, 2),
        "tempo_ds": [0],
        "tempo_us": [1],
        "time_downsample_factor": 2,
        "num_res_blocks": 1,
        "dropout": 0.0,
        "use_checkpoint": False,
        "init_pad_mode": "replicate",
        "norm_type": "groupnorm",
        "interpolation_mode": "nearest",
        "fix_encoder": False,
        "fix_decoder": False,
    }
    regularizer = {
        "codebook_dim": 2,
        "codebook_size": 8,
        "num_quantizers": 2,
    }
    return VidtokRVQ(encoder=encoder, regularizer=regularizer)


def test_vidtok_grad_causality():
    torch.manual_seed(0)
    model = _build_vidtok()
    model.eval()

    B, T, H, W = 1, 8, 8, 8
    x = torch.randn(B, 1, T, H, W, requires_grad=True)

    out, _ = model(x, global_step=0)
    t_split = T // 2
    loss = out[:, :, :t_split].sum()
    loss.backward()

    assert_future_grad_zero(x, t_split, time_dim=2)


def test_vidtok_rvq_grad_causality():
    torch.manual_seed(0)
    model = _build_vidtok_rvq()
    model.train()

    B, T, H, W = 1, 8, 8, 8
    x = torch.randn(B, 1, T, H, W, requires_grad=True)

    out, _ = model(x, global_step=0)
    t_split = T // 2
    loss = out[:, :, :t_split].sum()
    loss.backward()

    assert_future_grad_zero(x, t_split, time_dim=2)
