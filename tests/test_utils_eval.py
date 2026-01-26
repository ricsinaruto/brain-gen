import torch

from brain_gen.utils.eval import sample


def test_sample_accepts_tensor_temperature():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    temperature = torch.tensor([1.0, 0.5, 0.25]).view(1, 3, 1)
    out = sample(logits, strategy="top_k", top_k=1, temperature=temperature)
    assert out.shape == (2, 3)
    assert torch.equal(out, logits.argmax(dim=-1))


def test_sample_top_p_one_matches_roulette_distribution():
    # top_p=1.0 should keep the full distribution (no truncation).
    torch.manual_seed(0)
    logits = torch.tensor([[[2.0, 1.0, 0.5, -0.3, -1.2]]])
    draws = 20000
    batch_logits = logits.expand(draws, 1, -1)

    torch.manual_seed(1)
    roulette_samples = sample(batch_logits, strategy="roulette").view(-1)
    roulette_probs = torch.bincount(roulette_samples, minlength=logits.size(-1)).float()
    roulette_probs = roulette_probs / roulette_probs.sum()

    torch.manual_seed(2)
    top_p_samples = sample(batch_logits, strategy="top_p", top_p=1.0).view(-1)
    top_p_probs = torch.bincount(top_p_samples, minlength=logits.size(-1)).float()
    top_p_probs = top_p_probs / top_p_probs.sum()

    max_diff = (roulette_probs - top_p_probs).abs().max().item()
    assert max_diff < 0.02


def test_top_p_one_preserves_softmax_distribution():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    probs = torch.softmax(logits, dim=-1)

    sorted_p, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    mask = cum > 1.0
    mask[..., 0] = False
    sorted_p = sorted_p.masked_fill(mask, 0.0)
    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
    restored = torch.zeros_like(sorted_p).scatter_(-1, sorted_idx, sorted_p)

    assert torch.allclose(restored, probs, rtol=0.0, atol=1e-6)
