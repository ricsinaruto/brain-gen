"""
Unit tests for µ-law quantization utilities in brain_gen.utils.quantizers.

These tests verify that the µ-law companding is (approximately) invertible
and that corner cases are handled correctly.
"""

import numpy as np
import pytest
import torch

from brain_gen.utils.quantizers import (
    mulaw,
    mulaw_inv,
    mulaw_torch,
    mulaw_inv_torch,
)


class TestMulawNumpy:
    """Tests for NumPy-based µ-law quantization."""

    @pytest.mark.parametrize("mu", [63, 127, 255, 511, 1023])
    def test_roundtrip_recovers_signal_within_tolerance(self, mu: int):
        """Quantize + inverse should approximately recover the input."""
        rng = np.random.default_rng(seed=42)
        x = rng.uniform(-0.999, 0.999, size=(100,)).astype(np.float32)

        codes, x_recon = mulaw(x, mu=mu)
        assert codes.shape == x.shape
        assert x_recon.shape == x.shape

        # Check reconstruction is within quantization step tolerance
        # µ-law has non-uniform quantization - error is larger for larger values
        # Use log(1+mu)/mu as a more realistic bound for the worst case
        max_error = 2.0 * np.log1p(mu) / mu + 0.02
        np.testing.assert_allclose(x_recon, x, atol=max_error)

    def test_output_codes_in_valid_range(self):
        """Codes should be integers in [0, mu]."""
        mu = 255
        x = np.linspace(-0.999, 0.999, 100)
        codes, _ = mulaw(x, mu=mu)

        assert codes.dtype == np.uint8
        assert codes.min() >= 0
        assert codes.max() <= mu

    def test_monotonicity_preserved(self):
        """Increasing inputs should map to non-decreasing codes."""
        x = np.linspace(-0.999, 0.999, 200)
        codes, _ = mulaw(x, mu=255)

        # Codes should be monotonically non-decreasing
        assert np.all(np.diff(codes.astype(np.int32)) >= 0)

    def test_zero_maps_to_midpoint(self):
        """Zero input should map to approximately mu/2."""
        mu = 255
        x = np.array([0.0])
        codes, _ = mulaw(x, mu=mu)
        # Zero maps to bin mu/2 (accounting for rounding)
        expected = mu // 2
        assert abs(int(codes[0]) - expected) <= 1

    def test_inverse_is_inverse(self):
        """mulaw_inv should invert mulaw_inv on code space."""
        mu = 255
        codes = np.arange(0, mu + 1, dtype=np.uint8)
        recovered = mulaw_inv(codes, mu=mu)

        # Recovered values should be in [-1, 1]
        assert recovered.min() >= -1.0
        assert recovered.max() <= 1.0
        # Monotonicity should be preserved
        assert np.all(np.diff(recovered) >= 0)


class TestMulawTorch:
    """Tests for PyTorch-based µ-law quantization."""

    @pytest.mark.parametrize("mu", [63, 127, 255])
    def test_roundtrip_torch(self, mu: int):
        """PyTorch mulaw + inverse should approximately recover input."""
        torch.manual_seed(42)
        x = torch.rand(100) * 1.998 - 0.999  # uniform in ~[-0.999, 0.999]

        codes = mulaw_torch(x, mu=mu)
        x_recon = mulaw_inv_torch(codes, mu=mu)

        # µ-law has non-uniform quantization - error is larger for larger values
        max_error = 2.0 * float(torch.log1p(torch.tensor(float(mu)))) / mu + 0.02
        torch.testing.assert_close(x_recon, x, atol=max_error, rtol=0.0)

    def test_output_codes_dtype_and_range(self):
        """Codes should be long tensors in [0, mu]."""
        mu = 255
        x = torch.linspace(-0.999, 0.999, 100)
        codes = mulaw_torch(x, mu=mu)

        assert codes.dtype == torch.long
        assert codes.min().item() >= 0
        assert codes.max().item() <= mu

    def test_batched_shapes_preserved(self):
        """Quantization should preserve input tensor shape."""
        x = torch.randn(2, 10, 50).clamp(-0.999, 0.999)
        codes = mulaw_torch(x, mu=255)
        assert codes.shape == x.shape

    def test_gradient_does_not_flow_through_quantization(self):
        """Quantization is not differentiable (discrete operation)."""
        x = torch.randn(10, requires_grad=True).clamp(-0.999, 0.999)
        codes = mulaw_torch(x, mu=255)
        # Codes are long, no gradient connection
        assert codes.dtype == torch.long
        assert not codes.requires_grad

    def test_numpy_torch_consistency(self):
        """NumPy and PyTorch implementations should produce same codes."""
        np.random.seed(42)
        x_np = np.random.uniform(-0.999, 0.999, size=100).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        codes_np, _ = mulaw(x_np, mu=255)
        codes_torch = mulaw_torch(x_torch, mu=255)

        np.testing.assert_array_equal(codes_np, codes_torch.numpy())


class TestMulawEdgeCases:
    """Edge case tests for µ-law quantization."""

    def test_extreme_values_clipped(self):
        """Values outside [-1, 1] should be clipped."""
        x = np.array([-2.0, -1.5, 1.5, 2.0])
        codes, _ = mulaw(x, mu=255)
        # Should not crash, codes should be valid
        assert codes.min() >= 0
        assert codes.max() <= 255

    def test_empty_array(self):
        """Empty arrays should work."""
        x = np.array([])
        codes, recon = mulaw(x, mu=255)
        assert codes.shape == (0,)
        assert recon.shape == (0,)

    def test_single_value(self):
        """Single value should work."""
        x = np.array([0.5])
        codes, recon = mulaw(x, mu=255)
        assert codes.shape == (1,)
        assert recon.shape == (1,)
