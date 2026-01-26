import torch

from brain_gen.layers.quantizers import kmeans, sample_vectors


@torch.no_grad()
def _dense_kmeans_reference(
    samples: torch.Tensor, num_clusters: int, kmeans_iters: int
):
    samples = samples.reshape(-1, samples.shape[-1])
    dim, dtype = samples.shape[1], samples.dtype

    if samples.shape[0] < num_clusters:
        random_noise = torch.randn(
            size=(num_clusters - samples.shape[0], dim),
            device=samples.device,
            dtype=dtype,
        )
        samples = torch.cat([samples, random_noise], dim=0)

    centers = sample_vectors(samples, num_clusters)
    sample_norm = (samples**2).sum(1, keepdim=True)

    for _ in range(kmeans_iters):
        center_norm = (centers**2).sum(1)
        dist = sample_norm + center_norm - 2 * (samples @ centers.t())
        buckets = dist.argmin(dim=1)
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins[zero_mask] = 1

        new_centers = centers.new_zeros(num_clusters, dim, dtype=dtype)
        new_centers.scatter_add_(0, buckets.unsqueeze(-1).expand(-1, dim), samples)
        new_centers = new_centers / bins[..., None]
        centers = torch.where(zero_mask[..., None], centers, new_centers)

    return centers, bins


def test_chunked_kmeans_matches_dense_small_inputs():
    torch.manual_seed(0)
    samples = torch.randn(200, 8)

    rng_state = torch.get_rng_state()
    ref_centers, ref_bins = _dense_kmeans_reference(
        samples, num_clusters=16, kmeans_iters=4
    )

    torch.set_rng_state(rng_state)
    chunk_centers, chunk_bins = kmeans(
        samples, nums_clusters=16, kmeans_iters=4, chunk_size=32
    )

    torch.testing.assert_close(ref_centers, chunk_centers)
    torch.testing.assert_close(ref_bins, chunk_bins)


def test_kmeans_handles_more_clusters_than_samples_with_chunks():
    torch.manual_seed(0)
    samples = torch.randn(10, 4)

    centers, bins = kmeans(samples, nums_clusters=16, kmeans_iters=3, chunk_size=3)

    assert centers.shape == (16, 4)
    assert bins.shape == (16,)
    assert centers.device == samples.device
    assert bins.device == samples.device
    assert bins.min() >= 1
    assert not torch.isnan(centers).any()
