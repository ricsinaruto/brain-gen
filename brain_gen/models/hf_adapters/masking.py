from collections import OrderedDict
import math

import torch

from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask

from torch.nn.attention.flex_attention import BlockMask

_SPARSE_MASK_CACHE_MAX = 8
_SPARSE_MASK_CACHE: OrderedDict[tuple, BlockMask] = OrderedDict()


def _sparse_mask_cache_key(
    *,
    device: torch.device,
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
    q_block_offset: int,
    tokens_per_frame: int,
    block_size: int,
    local_window_frames: int,
    global_window_frames: int,
    anchor_stride_frames: int,
    sink_frames: int,
    global_heads: tuple[int, int],
) -> tuple:
    return (
        str(device),
        batch_size,
        num_heads,
        q_len,
        kv_len,
        q_block_offset,
        tokens_per_frame,
        block_size,
        local_window_frames,
        global_window_frames,
        anchor_stride_frames,
        sink_frames,
        global_heads,
    )


def _get_cached_sparse_mask(cache_key: tuple) -> BlockMask | None:
    block_mask = _SPARSE_MASK_CACHE.get(cache_key)
    if block_mask is not None:
        _SPARSE_MASK_CACHE.move_to_end(cache_key)
    return block_mask


def _set_cached_sparse_mask(cache_key: tuple, block_mask: BlockMask) -> None:
    _SPARSE_MASK_CACHE[cache_key] = block_mask
    _SPARSE_MASK_CACHE.move_to_end(cache_key)
    if len(_SPARSE_MASK_CACHE) > _SPARSE_MASK_CACHE_MAX:
        _SPARSE_MASK_CACHE.popitem(last=False)


def _clear_sparse_mask_cache() -> None:
    _SPARSE_MASK_CACHE.clear()


def _build_sparse_block_mask(
    *,
    batch_size: int,
    num_heads: int,
    q_len: int,
    kv_len: int,
    block_size: int,
    global_heads: tuple[int, int],
    W_local_blocks: int,
    W_global_blocks: int,
    stride_blocks: int,
    sink_blocks: int,
    mask_mod,
    device: torch.device,
    q_block_offset: int = 0,
) -> BlockMask:
    num_q_blocks = math.ceil(q_len / block_size)
    num_kv_blocks = math.ceil(kv_len / block_size)
    global_set = set(global_heads)

    def build_lists(is_global: bool) -> tuple[list[list[int]], list[list[int]]]:
        full_lists: list[list[int]] = []
        partial_lists: list[list[int]] = []
        for qb_local in range(num_q_blocks):
            qb = qb_local + q_block_offset
            full_row: list[int] = []
            partial_row: list[int] = []
            max_kb = min(qb, num_kv_blocks - 1)
            for kb in range(max_kb + 1):
                d = qb - kb
                if is_global:
                    allow = (
                        d < W_global_blocks
                        or (d >= W_global_blocks and (kb % stride_blocks) == 0)
                        or (kb < sink_blocks)
                    )
                else:
                    allow = d < W_local_blocks
                if not allow:
                    continue
                if kb == qb:
                    partial_row.append(kb)
                else:
                    full_row.append(kb)
            full_lists.append(full_row)
            partial_lists.append(partial_row)
        return full_lists, partial_lists

    local_full, local_partial = build_lists(False)
    global_full, global_partial = build_lists(True)

    partial_lists_by_head = [
        global_partial if h in global_set else local_partial for h in range(num_heads)
    ]
    full_lists_by_head = [
        global_full if h in global_set else local_full for h in range(num_heads)
    ]

    kv_num_blocks = torch.zeros((num_heads, num_q_blocks), dtype=torch.int32)
    kv_indices = torch.zeros(
        (num_heads, num_q_blocks, num_kv_blocks), dtype=torch.int32
    )
    for h, row_lists in enumerate(partial_lists_by_head):
        for qb, cols in enumerate(row_lists):
            if cols:
                kv_num_blocks[h, qb] = len(cols)
                kv_indices[h, qb, : len(cols)] = torch.tensor(cols, dtype=torch.int32)

    kv_num_blocks = kv_num_blocks.unsqueeze(0).repeat(batch_size, 1, 1)
    kv_indices = kv_indices.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    kv_num_blocks = kv_num_blocks.to(device)
    kv_indices = kv_indices.to(device)

    full_kv_num_blocks = torch.zeros((num_heads, num_q_blocks), dtype=torch.int32)
    full_kv_indices = torch.zeros(
        (num_heads, num_q_blocks, num_kv_blocks), dtype=torch.int32
    )
    has_full = False
    for h, row_lists in enumerate(full_lists_by_head):
        for qb, cols in enumerate(row_lists):
            if cols:
                has_full = True
                full_kv_num_blocks[h, qb] = len(cols)
                full_kv_indices[h, qb, : len(cols)] = torch.tensor(
                    cols, dtype=torch.int32
                )

    if has_full:
        full_kv_num_blocks = (
            full_kv_num_blocks.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        )
        full_kv_indices = (
            full_kv_indices.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        )
    else:
        full_kv_num_blocks = None
        full_kv_indices = None

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(q_len, kv_len),
    )


def _sparse_causal_mask(
    config: PretrainedConfig,
    inputs_embeds: torch.Tensor,
    past_key_values: tuple | None = None,
    T: int | None = 720,
    tokens_per_frame: int = 32,  # 4 sites * 8 RVQ
    block_size: int = 256,  # 128 tokens = 4 frames (since 32 tok/frame)
    # Local attention span (for most heads)
    local_window_frames: int = 8,  # e.g. 128 frames
    # Global heads get larger window + anchors
    global_window_frames: int = 24,  # e.g. 256 frames
    anchor_stride_frames: int = 120,  # anchors every 256 frames (coarse long-range)
    sink_frames: int = 0,  # optionally always-visible earliest frames
    # Which heads are "global" (2 heads recommended: one per KV group)
    global_heads: tuple[int, int] = (3, 7),
    device: str | torch.device | None = None,
):
    """
    Returns a BlockMask for causal video-token attention on a flattened sequence:
      [frame0 tokens (32)] [frame1 tokens (32)] ... [frameT-1 tokens (32)]

    Assumes:
      - Query shape: (B, Hq, L, Dh)
      - Key/Val shape: (B, Hkv, L, Dh)
      - Use flex_attention(..., enable_gqa=True) with Hq=8, Hkv=2 (rep=4).
    """
    if device is None:
        device = inputs_embeds.device
    device = torch.device(device)

    past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    q_len = int(inputs_embeds.shape[1])
    kv_len = int(past_len + q_len)

    if past_len == 0:
        if kv_len % tokens_per_frame != 0:
            raise ValueError(
                f"Sequence length {kv_len} must be divisible by "
                f"tokens_per_frame={tokens_per_frame}."
            )
        inferred_T = kv_len // tokens_per_frame
        if T is None or inferred_T != T:
            T = inferred_T

    if block_size % tokens_per_frame != 0:
        raise ValueError(
            f"block_size={block_size} must be a "
            " multiple of tokens_per_frame={tokens_per_frame} "
            "so blocks align cleanly to frames."
        )
    if kv_len % block_size != 0 and past_len == 0:
        raise ValueError(
            f"Sequence length L={kv_len} must be divisible by block_size={block_size} "
            "(choose block_size 32/64/128/256... that divides L)."
        )
    if config.num_attention_heads != 8:
        raise ValueError(
            "This helper assumes Hq=8 (edit/remove this check if you want other Hq)."
        )

    frames_per_block = block_size // tokens_per_frame  # 128/32 = 4 frames/block

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    W_local_blocks = ceil_div(local_window_frames, frames_per_block)
    W_global_blocks = ceil_div(global_window_frames, frames_per_block)
    stride_blocks = max(1, anchor_stride_frames // frames_per_block)
    sink_blocks = ceil_div(sink_frames, frames_per_block)

    g0, g1 = global_heads

    # mask_mod(b, h, q_idx, kv_idx) -> bool tensor (True = allowed)
    def mask_mod(b, h, q_idx, kv_idx):
        q_global = q_idx + past_len
        kv_global = kv_idx
        # Causality in token indices
        causal = kv_global <= q_global

        # Work at block granularity (keeps blocks full off-diagonal)
        qb = q_global // block_size
        kb = kv_global // block_size
        d = qb - kb  # block distance (>=0 for causal entries)

        # Local contiguous sliding window (fast)
        local_ok = d < W_local_blocks

        # Global heads: larger window + anchors + optional sinks
        global_window_ok = d < W_global_blocks
        anchor_ok = (d >= W_global_blocks) & ((kb % stride_blocks) == 0)
        sink_ok = kb < sink_blocks  # if sink_blocks==0, this is always False

        global_ok = global_window_ok | anchor_ok | sink_ok

        is_global_head = (h == g0) | (h == g1)

        # Head-dependent choice (local for most heads, global for selected heads)
        allow = torch.where(is_global_head, global_ok, local_ok)

        return causal & allow

    cache_key = _sparse_mask_cache_key(
        device=device,
        batch_size=inputs_embeds.shape[0],
        num_heads=config.num_attention_heads,
        q_len=q_len,
        kv_len=kv_len,
        q_block_offset=past_len // block_size if block_size > 0 else 0,
        tokens_per_frame=tokens_per_frame,
        block_size=block_size,
        local_window_frames=local_window_frames,
        global_window_frames=global_window_frames,
        anchor_stride_frames=anchor_stride_frames,
        sink_frames=sink_frames,
        global_heads=global_heads,
    )
    cached = _get_cached_sparse_mask(cache_key)
    if cached is not None:
        return cached

    block_mask = _build_sparse_block_mask(
        batch_size=inputs_embeds.shape[0],
        num_heads=config.num_attention_heads,
        q_len=q_len,
        kv_len=kv_len,
        block_size=block_size,
        global_heads=global_heads,
        W_local_blocks=W_local_blocks,
        W_global_blocks=W_global_blocks,
        stride_blocks=stride_blocks,
        sink_blocks=sink_blocks,
        mask_mod=mask_mod,
        device=device,
        q_block_offset=past_len // block_size if block_size > 0 else 0,
    )
    _set_cached_sparse_mask(cache_key, block_mask)
    return block_mask


def make_block_causal_mask(block_len: int):
    def mask_mod(b, h, q_idx, kv_idx):
        q_blk = q_idx // block_len
        k_blk = kv_idx // block_len
        # allow attending to any token in same block (non-causal)
        # and to any token in earlier blocks (causal across blocks)
        return k_blk <= q_blk

    return mask_mod


def _block_causal_mask(
    config: PretrainedConfig,
    block_size: int,
    past_key_values: tuple,
    inputs_embeds: torch.Tensor,
    include_position_ids: bool = True,
) -> torch.Tensor:
    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )

    position_ids = None
    if include_position_ids:
        position_ids = cache_position.unsqueeze(0)

    config._attn_implementation = (
        "flex_attention"
        if past_key_values is None and inputs_embeds.device.type == "cuda"
        else "sdpa"
    )

    mask_kwargs = {
        "config": config,
        "input_embeds": inputs_embeds,
        "attention_mask": None,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "or_mask_function": make_block_causal_mask(block_size),
    }

    return create_causal_mask(**mask_kwargs)
