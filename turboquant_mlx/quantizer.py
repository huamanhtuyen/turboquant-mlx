"""PolarQuant: core quantization for TurboQuant.

Implements Algorithm 1 from the TurboQuant paper:
1. Extract norm
2. Random rotation (Randomized Hadamard Transform)
3. Scalar quantization with optimal Lloyd-Max codebook
"""

import mlx.core as mx
import math
from turboquant_mlx.rotation import (
    random_diagonal_sign,
    randomized_hadamard_transform,
    inverse_randomized_hadamard,
)


def _compute_gaussian_codebook(bits: int) -> mx.array:
    """Precompute optimal Lloyd-Max centroids for N(0,1) distribution.

    These are well-known values for small bit widths.
    After rotation, coordinates are approximately N(0, 1/sqrt(d)),
    but we normalize per-coordinate so these work directly.
    """
    # Optimal Lloyd-Max centroids for Gaussian N(0,1)
    # Precomputed from scipy.stats / Lloyd's algorithm
    codebooks = {
        1: [-0.7979, 0.7979],
        2: [-1.5104, -0.4528, 0.4528, 1.5104],
        3: [-2.1520, -1.3440, -0.7560, -0.2451,
             0.2451, 0.7560, 1.3440, 2.1520],
        4: [-2.7326, -2.0690, -1.6180, -1.2562,
            -0.9423, -0.6568, -0.3881, -0.1284,
             0.1284, 0.3881, 0.6568, 0.9423,
             1.2562, 1.6180, 2.0690, 2.7326],
    }
    if bits not in codebooks:
        raise ValueError(f"Unsupported bit width: {bits}. Use 1-4.")
    return mx.array(codebooks[bits], dtype=mx.float32)


def _compute_gaussian_boundaries(centroids: mx.array) -> mx.array:
    """Compute decision boundaries (midpoints between adjacent centroids)."""
    return (centroids[:-1] + centroids[1:]) / 2.0


class PolarQuantizer:
    """PolarQuant quantizer for a fixed dimension and bit width.

    Precomputes rotation signs and codebook at init.
    Quantize/dequantize are pure MLX ops, GPU-friendly.
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 42):
        """
        Args:
            dim: head dimension (must be power of 2)
            bits: quantization bits (1-4)
            seed: random seed for rotation
        """
        self.dim = dim
        self.bits = bits
        self.n_levels = 2 ** bits

        # Precompute rotation signs
        self.signs = random_diagonal_sign(dim, seed=seed)

        # Precompute codebook
        self.centroids = _compute_gaussian_codebook(bits)
        self.boundaries = _compute_gaussian_boundaries(self.centroids)

        # Scale factor: after rotation, coordinates have std ≈ 1/sqrt(d)
        self.scale = 1.0 / math.sqrt(dim)

    def quantize(self, x: mx.array):
        """Quantize vectors using PolarQuant.

        Args:
            x: (..., dim) input vectors

        Returns:
            indices: (..., dim) uint8 codebook indices
            norms: (...,) vector L2 norms
        """
        # Upcast to float32 for numerical stability (fp16 overflows in norm)
        x = x.astype(mx.float32)

        # Extract norms
        norms = mx.linalg.norm(x, axis=-1, keepdims=True)
        safe_norms = mx.maximum(norms, 1e-8)
        x_unit = x / safe_norms

        # Random rotation
        x_rotated = randomized_hadamard_transform(x_unit, self.signs)

        # Normalize by expected std after rotation (1/sqrt(d))
        x_scaled = x_rotated / self.scale

        # Nearest centroid: compare against boundaries
        # x_scaled: (..., d), boundaries: (n_levels-1,)
        # indices[i] = number of boundaries that x_scaled[i] exceeds
        indices = mx.zeros(x_scaled.shape, dtype=mx.uint8)
        for i, b in enumerate(self.boundaries.tolist()):
            indices = indices + (x_scaled > b).astype(mx.uint8)

        return indices, norms.squeeze(-1)

    def dequantize(self, indices: mx.array, norms: mx.array) -> mx.array:
        """Dequantize from indices + norms back to vectors.

        Args:
            indices: (..., dim) uint8 codebook indices
            norms: (...,) vector norms

        Returns:
            (..., dim) reconstructed vectors
        """
        # Look up centroids
        flat_idx = indices.reshape(-1).astype(mx.int32)
        flat_vals = self.centroids[flat_idx]
        y_hat = flat_vals.reshape(indices.shape)

        # Undo scaling
        y_hat = y_hat * self.scale

        # Inverse rotation
        x_hat_unit = inverse_randomized_hadamard(y_hat, self.signs)

        # Restore norms
        x_hat = x_hat_unit * norms[..., None]

        return x_hat
