import math
import torch
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig

from brain_gen.models.hf_adapters.masking import (
    _clear_sparse_mask_cache,
    _sparse_causal_mask,
)


def _make_inputs(batch: int = 2, seq_len: int = 8, hidden: int = 4) -> torch.Tensor:
    return torch.zeros(batch, seq_len, hidden)


class _PastKeyValues:
    def __init__(self, seq_len: int) -> None:
        self._seq_len = int(seq_len)

    def get_seq_length(self) -> int:
        return self._seq_len


def test_sparse_causal_mask_cache_reuses_blockmask() -> None:
    config = PretrainedConfig(num_attention_heads=8)
    inputs = _make_inputs()

    _clear_sparse_mask_cache()
    mask1 = _sparse_causal_mask(
        config,
        inputs,
        T=4,
        tokens_per_frame=2,
        block_size=4,
        local_window_frames=2,
        global_window_frames=2,
        anchor_stride_frames=2,
        sink_frames=0,
        global_heads=(3, 7),
        device=inputs.device,
    )
    mask2 = _sparse_causal_mask(
        config,
        inputs,
        T=4,
        tokens_per_frame=2,
        block_size=4,
        local_window_frames=2,
        global_window_frames=2,
        anchor_stride_frames=2,
        sink_frames=0,
        global_heads=(3, 7),
        device=inputs.device,
    )

    assert mask1 is mask2


def test_sparse_causal_mask_infers_T_from_seq_len() -> None:
    config = PretrainedConfig(num_attention_heads=8)
    inputs = _make_inputs()

    _clear_sparse_mask_cache()
    mask = _sparse_causal_mask(
        config,
        inputs,
        T=3,
        tokens_per_frame=2,
        block_size=4,
        local_window_frames=2,
        global_window_frames=2,
        anchor_stride_frames=2,
        sink_frames=0,
        global_heads=(3, 7),
        device=inputs.device,
    )

    assert mask.kv_num_blocks.shape[2] == math.ceil(inputs.shape[1] / 4)


def test_sparse_causal_mask_block_layout_matches_reference() -> None:
    config = PretrainedConfig(num_attention_heads=8)
    inputs = _make_inputs(batch=2, seq_len=8, hidden=4)

    local_window_frames = 2
    global_window_frames = 4
    anchor_stride_frames = 4
    sink_frames = 0
    tokens_per_frame = 2
    block_size = 4

    _clear_sparse_mask_cache()
    mask = _sparse_causal_mask(
        config,
        inputs,
        T=4,
        tokens_per_frame=tokens_per_frame,
        block_size=block_size,
        local_window_frames=local_window_frames,
        global_window_frames=global_window_frames,
        anchor_stride_frames=anchor_stride_frames,
        sink_frames=sink_frames,
        global_heads=(3, 7),
        device=inputs.device,
    )

    frames_per_block = block_size // tokens_per_frame
    W_local_blocks = (local_window_frames + frames_per_block - 1) // frames_per_block
    W_global_blocks = (global_window_frames + frames_per_block - 1) // frames_per_block
    stride_blocks = max(1, anchor_stride_frames // frames_per_block)
    sink_blocks = (sink_frames + frames_per_block - 1) // frames_per_block
    g0, g1 = (3, 7)

    def mask_mod(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        qb = q_idx // block_size
        kb = kv_idx // block_size
        d = qb - kb

        local_ok = d < W_local_blocks
        global_window_ok = d < W_global_blocks
        anchor_ok = (d >= W_global_blocks) & ((kb % stride_blocks) == 0)
        sink_ok = kb < sink_blocks
        global_ok = global_window_ok | anchor_ok | sink_ok

        is_global_head = (h == g0) | (h == g1)
        allow = torch.where(is_global_head, global_ok, local_ok)
        return causal & allow

    ref = create_block_mask(
        mask_mod,
        B=inputs.shape[0],
        H=config.num_attention_heads,
        Q_LEN=inputs.shape[1],
        KV_LEN=inputs.shape[1],
        device=inputs.device,
        BLOCK_SIZE=block_size,
    )

    def block_sets(block_mask):
        kv_num = block_mask.kv_num_blocks
        kv_idx = block_mask.kv_indices
        full_num = block_mask.full_kv_num_blocks
        full_idx = block_mask.full_kv_indices
        out = {}
        for b in range(kv_num.shape[0]):
            for h in range(kv_num.shape[1]):
                for qb in range(kv_num.shape[2]):
                    n_partial = int(kv_num[b, h, qb].item())
                    partial = set(kv_idx[b, h, qb, :n_partial].tolist())
                    full = set()
                    if full_num is not None and full_idx is not None:
                        n_full = int(full_num[b, h, qb].item())
                        full = set(full_idx[b, h, qb, :n_full].tolist())
                    out[(b, h, qb)] = partial | full
        return out

    assert block_sets(mask) == block_sets(ref)


def test_sparse_causal_mask_allows_cached_partial_sequence() -> None:
    config = PretrainedConfig(num_attention_heads=8)
    inputs = _make_inputs(batch=1, seq_len=2, hidden=4)
    past = _PastKeyValues(seq_len=5)

    _clear_sparse_mask_cache()
    mask1 = _sparse_causal_mask(
        config,
        inputs,
        past_key_values=past,
        T=1,
        tokens_per_frame=4,
        block_size=4,
        local_window_frames=2,
        global_window_frames=2,
        anchor_stride_frames=2,
        sink_frames=0,
        global_heads=(3, 7),
        device=inputs.device,
    )
    mask2 = _sparse_causal_mask(
        config,
        inputs,
        past_key_values=past,
        T=1,
        tokens_per_frame=4,
        block_size=4,
        local_window_frames=2,
        global_window_frames=2,
        anchor_stride_frames=2,
        sink_frames=0,
        global_heads=(3, 7),
        device=inputs.device,
    )

    assert mask1 is mask2
    assert mask1.kv_num_blocks.shape[2] == math.ceil(inputs.shape[1] / 4)
