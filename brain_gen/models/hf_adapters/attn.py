# ADAPTED FROM: https://github.com/huggingface/transformers

from typing import Optional
import warnings
import torch

from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import flex_attention

_COMPILED_FLEX_ATTENTION = None


def _get_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is not None:
        return _COMPILED_FLEX_ATTENTION

    if not torch.cuda.is_available():
        _COMPILED_FLEX_ATTENTION = flex_attention
        return _COMPILED_FLEX_ATTENTION

    try:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention)
    except Exception as exc:
        warnings.warn(
            f"Failed to compile flex_attention; using eager path. ({exc})",
            RuntimeWarning,
        )
        _COMPILED_FLEX_ATTENTION = flex_attention
    return _COMPILED_FLEX_ATTENTION


def sdpa_flash_check(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, tag=""):
    # Make sure warning messages actually show up
    warnings.simplefilter("always")

    # k = k.repeat_interleave(q.size(-3)//k.size(-3), -3)
    # v = v.repeat_interleave(q.size(-3)//v.size(-3), -3)

    # Basic sanity prints (optional)
    print(
        f"[{tag}] q={tuple(q.shape)} {q.dtype} {q.device} contiguous={q.is_contiguous()}"
    )
    print(
        f"[{tag}] k={tuple(k.shape)} {k.dtype} {k.device} contiguous={k.is_contiguous()}"
    )
    print(
        f"[{tag}] v={tuple(v.shape)} {v.dtype} {v.device} contiguous={v.is_contiguous()}"
    )
    if attn_mask is None:
        print(f"[{tag}] attn_mask=None")
    else:
        print(
            f"[{tag}] attn_mask={tuple(attn_mask.shape)} {attn_mask.dtype} {attn_mask.device}"
        )

    # 1) Is FlashAttention even compiled into this PyTorch build?
    print(
        f"[{tag}] is_flash_attention_available = {torch.backends.cuda.is_flash_attention_available()}"
    )

    # 2) Construct SDPAParams and ask PyTorch if FlashAttention can be used.
    params = torch.backends.cuda.SDPAParams(
        q, k, v, attn_mask, dropout_p, is_causal, True
    )  # :contentReference[oaicite:1]{index=1}

    ok_flash = torch.backends.cuda.can_use_flash_attention(
        params, debug=True
    )  # :contentReference[oaicite:2]{index=2}
    ok_eff = torch.backends.cuda.can_use_efficient_attention(
        params, debug=True
    )  # :contentReference[oaicite:3]{index=3}
    ok_cudnn = torch.backends.cuda.can_use_cudnn_attention(
        params, debug=True
    )  # :contentReference[oaicite:4]{index=4}

    print(f"[{tag}] can_use_flash_attention     = {ok_flash}")
    print(f"[{tag}] can_use_efficient_attention = {ok_eff}")
    print(f"[{tag}] can_use_cudnn_attention     = {ok_cudnn}")

    return ok_flash, ok_eff, ok_cudnn


@torch._dynamo.disable
def debug_sdpa(q, k, v, attn_mask, dropout, is_causal):
    params = torch.backends.cuda.SDPAParams(
        q, k, v, attn_mask, dropout, is_causal, True
    )
    print("flash", torch.backends.cuda.can_use_flash_attention(params, debug=True))
    print("eff", torch.backends.cuda.can_use_efficient_attention(params, debug=True))
    print("cudnn", torch.backends.cuda.can_use_cudnn_attention(params, debug=True))
    print("q/k/v strides", q.stride(), k.stride(), v.stride())


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch,
    num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def use_gqa_in_sdpa(attention_mask: Optional[torch.Tensor], key: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    #   - key is not a torch.fx.Proxy (otherwise it will fail with a tracing error)
    return attention_mask is None and not isinstance(key, torch.fx.Proxy)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this
    # 'is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options.
    # An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape,
    # otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which
        # specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations
        # for encoder, decoder, and encoder-decoder attns
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing,
    # resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # sdpa_flash_check(
    #     query,
    #     key,
    #     value,
    #     attn_mask=attention_mask,
    #     dropout_p=dropout,
    #     is_causal=is_causal,
    #     tag="sdpa_attention_forward",
    # )

    # Prefer FlashAttention but allow graceful fallback (e.g. float32 eval) instead
    # of enforcing a kernel that may be unavailable for the given dtype/mask.
    allowed_backends = (
        [SDPBackend.CUDNN_ATTENTION]
        if query.shape[2] > 1
        else [SDPBackend.FLASH_ATTENTION]
    )
    if query.device.type != "cuda" or attention_mask is not None:
        print("Using fallback backends")
        allowed_backends.extend(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ]
        )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # debug_sdpa(query, key, value, attention_mask, dropout, is_causal)
    # sdpa_flash_check(query, key, value, attention_mask, dropout, is_causal)

    with sdpa_kernel(allowed_backends):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            **sdpa_kwargs,
        )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if dropout:
        warnings.warn(
            "Flex attention does not support dropout; ignoring provided dropout value.",
            RuntimeWarning,
        )

    # Use smaller block sizes to reduce Triton shared memory requirements.
    # Default BLOCK_M/BLOCK_N (128) can exceed GPU shared memory limits on some.
    kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64}

    flex_attn = _get_flex_attention()
    attn_output = flex_attn(
        query,
        key,
        value,
        block_mask=attention_mask,
        scale=scaling,
        enable_gqa=True,
        kernel_options=kernel_options,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
