import torch

from brain_gen.layers.flatgpt_layers import ChannelHead, EmbeddingCorruptor


@torch.no_grad()
def _embedding_corruptor_reference(
    corruptor: EmbeddingCorruptor, x: torch.Tensor
) -> torch.Tensor:
    x = x.reshape(x.shape[0], corruptor.n_time, corruptor.n_space, corruptor.n_levels)
    B, T, S, Q = x.shape
    device = x.device

    p_levels = corruptor.p_levels_tensor
    if p_levels.numel() not in (1, Q):
        raise ValueError(
            f"Expected 1 or {Q} corruption probabilities, got {p_levels.numel()}."
        )
    if p_levels.device != device:
        p_levels = p_levels.to(device=device)
    p = p_levels.view(1, 1, 1, -1)
    corrupt = torch.rand(B, T, S, Q, device=device) < p

    num_spans = max(1, T // 64)
    t_idx = torch.arange(T, device=device)
    for _ in range(num_spans):
        start = torch.randint(0, T, (1,), device=device)
        length = torch.randint(2, 9, (1,), device=device)
        end = (start + length).clamp(max=T)

        q_sel = (torch.rand(Q, device=device) < 0.5)[None, None, None, :]
        span_mask = (t_idx >= start) & (t_idx < end)
        corrupt[:, span_mask, :, :] |= q_sel

    u = torch.rand(B, T, S, Q, device=device)
    use_null = corrupt & (u < 0.7)
    use_rand = corrupt & (u >= 0.7)

    x_corrupt = x.clone()
    x_flat = x_corrupt.reshape(B, T * S, Q)
    use_rand_flat = use_rand.reshape(B, T * S, Q)
    for q in range(Q):
        Kq = corruptor.embedding.emb[q].num_embeddings
        idx = use_rand_flat[..., q]
        num_rand = int(idx.sum())
        if num_rand:
            x_flat[..., q][idx] = torch.randint(0, Kq, (num_rand,), device=device)

    emb_levels = corruptor.embedding(x_flat).reshape(B, T, S, Q, -1)
    for q in range(Q):
        idx = use_null[..., q]
        emb_levels[..., q, :][idx] = corruptor.null_embed[q]

    return emb_levels.reshape(B, T * S * Q, -1)


def test_embedding_corruptor_matches_reference():
    torch.manual_seed(0)
    B, T, S, Q = 2, 8, 3, 4
    vocab_size = 32
    hidden_size = 12

    corruptor = EmbeddingCorruptor(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        reduced_shape=(T, S, Q),
    ).train()
    x = torch.randint(0, vocab_size, (B, T, S, Q))

    rng_state = torch.get_rng_state()
    ref = _embedding_corruptor_reference(corruptor, x)
    torch.set_rng_state(rng_state)
    out = corruptor(x)

    torch.testing.assert_close(out, ref)


def _make_channel_embeddings(num_channels: int, vocab_size: int, hidden_size: int):
    embeddings = torch.nn.ModuleList(
        [torch.nn.Embedding(vocab_size, hidden_size) for _ in range(num_channels)]
    )
    torch.manual_seed(0)
    for emb in embeddings:
        torch.nn.init.uniform_(emb.weight, a=-0.5, b=0.5)
    return embeddings


def test_channel_head_supports_scalar_chid_batch():
    num_channels = 4
    vocab_size = 8
    hidden_size = 6
    embeddings = _make_channel_embeddings(num_channels, vocab_size, hidden_size)
    head = ChannelHead(embeddings)
    x = torch.randn(3, 1, hidden_size)
    chid = 2

    expected = head.layers[(num_channels - 1 + chid) % num_channels](x)
    out = head(x, chid=chid)

    torch.testing.assert_close(out, expected)
