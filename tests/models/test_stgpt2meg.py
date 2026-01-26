import torch
from brain_gen.models import STGPT2MEG
from brain_gen.dataset.datasets import ChunkDataset
from brain_gen.dataset.datasplitter import build_indices
from tests.models.utils import make_dummy_session, assert_future_grad_zero


@torch.no_grad()
def test_stgpt2meg_forward_shapes():
    B, C, T = 2, 5, 12
    x = torch.randint(0, 64, (B, C, T))

    trf_args = dict(attn_args={"d_model": 32, "nheads": 4}, mlp_args={"d_model": 32})
    model = STGPT2MEG(
        num_channels=C,
        vocab_size=64,
        d_model=32,
        layers=2,
        trf_args=trf_args,
        embedding_args={},
    )
    model.eval()
    out = model(x)
    assert out.shape == (B, C, T, 64)


def _test_grad_causality_stgpt2meg(block: str):
    B, C, T, V = 2, 272, 64, 64
    # Random differentiable embeddings at block input
    emb = torch.randn(B, C, T, 32, requires_grad=True)
    trf_args = dict(attn_args={"d_model": 32, "nheads": 4}, mlp_args={"d_model": 32})
    model = STGPT2MEG(
        num_channels=C,
        vocab_size=V,
        d_model=32,
        layers=2,
        trf_args=trf_args,
        trf_block=block,
        embedding_args={},
    )
    model.eval()
    # forward through blocks
    h = emb
    for blk in model.blocks:
        h = blk(h.contiguous())
    y = model.head(model.norm(h))
    loss = y[..., :-1, :].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


def test_grad_causality_stgpt2meg(tmp_path):
    _test_grad_causality_stgpt2meg("STGPTBlockParallel")
    _test_grad_causality_stgpt2meg("STGPTBlock")
    _test_grad_causality_stgpt2meg("STBlock")

    # Need to run a dataset first
    root = tmp_path / "omega"
    make_dummy_session(str(root), "sub-001", C=272, T=32)

    indices, example_len, _, ch_names, pos_2d, sfreq = build_indices(
        str(root), example_len=0.16, overlap=0.0
    )

    # Use a short fixed window (example_len=0 leads to full-length windows here)
    _ = ChunkDataset(
        str(root),
        indices=indices,
        length=example_len,
        ch_names=ch_names,
        pos_2d=pos_2d,
        sfreq=sfreq,
    )
    _test_grad_causality_stgpt2meg("STConvBlock")
