"""Fused Metal kernels for TurboQuant decode.

The key bottleneck in TurboQuant decode is:
1. Unpack indices → codebook lookup → get quantized coordinates
2. Inverse rotation (WHT) → reconstruct original vector
3. Dot product with query

Doing these as separate MLX ops means 3 GPU kernel launches and 3 memory round-trips.
A fused kernel does all 3 in one pass: indices → dot product, no intermediate storage.

For decode (single token generation), this is the hot path: we compute
attention scores between one query and all cached keys/values.
"""

import mlx.core as mx
import math


# Fused TurboQuant dequantize kernel
# Input: packed indices (uint8), norms, codebook centroids, WHT signs
# Output: dequantized vectors
TURBOQUANT_DEQUANT_KERNEL = """
    // Thread processes one vector (one sequence position)
    uint pos = thread_position_in_grid.x;
    uint dim = dims[0];
    uint n_centroids = dims[1];

    // Each thread reconstructs one full vector
    // Step 1: Codebook lookup
    for (uint i = 0; i < dim; i++) {
        uint idx = indices[pos * dim + i];
        T centroid_val = centroids[idx];

        // Step 2: Scale by 1/sqrt(d) (undo the normalization)
        centroid_val *= scale[0];

        // Store for WHT (in threadgroup memory would be better, but start simple)
        out[pos * dim + i] = centroid_val;
    }

    // Step 3: Inverse WHT butterfly
    // WHT is self-inverse (up to normalization), so we apply forward WHT
    uint h = 1;
    while (h < dim) {
        for (uint i = 0; i < dim; i += 2 * h) {
            for (uint j = i; j < i + h; j++) {
                T a = out[pos * dim + j];
                T b = out[pos * dim + j + h];
                out[pos * dim + j] = a + b;
                out[pos * dim + j + h] = a - b;
            }
        }
        h *= 2;
    }

    // Normalize WHT by 1/sqrt(dim)
    T wht_scale = scale[0];  // reuse: 1/sqrt(dim)
    for (uint i = 0; i < dim; i++) {
        // Multiply by signs (inverse rotation = WHT then multiply by signs)
        out[pos * dim + i] = out[pos * dim + i] * wht_scale * signs[i];
        // Restore original norm
        out[pos * dim + i] *= norms[pos];
    }
"""

# Fused dequant + dot product kernel (the real prize)
# Computes: dot(query, dequant(indices, norms)) for all cached positions
# This avoids materializing the full dequantized KV cache
TURBOQUANT_FUSED_ATTENTION_KERNEL = """
    // Thread computes one attention score: dot(query, dequant(key[pos]))
    uint pos = thread_position_in_grid.x;
    uint dim = dims[0];

    // Reconstruct key vector in registers and accumulate dot product
    // Step 1+2: Codebook lookup + store for WHT
    // Use local array for WHT (dim must be known at compile time or small enough)
    T local_vec[256];  // max head_dim = 256

    for (uint i = 0; i < dim; i++) {
        uint idx = indices[pos * dim + i];
        local_vec[i] = centroids[idx] * scale[0];
    }

    // Step 3: Inverse WHT butterfly (in-place on local_vec)
    uint h = 1;
    while (h < dim) {
        for (uint i = 0; i < dim; i += 2 * h) {
            for (uint j = i; j < i + h; j++) {
                T a = local_vec[j];
                T b = local_vec[j + h];
                local_vec[j] = a + b;
                local_vec[j + h] = a - b;
            }
        }
        h *= 2;
    }

    // Step 4: Apply signs, norm, and accumulate dot product with query
    T dot = 0;
    T wht_norm = scale[0];
    T vec_norm = norms[pos];

    for (uint i = 0; i < dim; i++) {
        T val = local_vec[i] * wht_norm * signs[i] * vec_norm;
        dot += val * query[i];
    }

    out[pos] = dot;
"""


def create_dequant_kernel():
    """Create the fused dequantize Metal kernel."""
    return mx.fast.metal_kernel(
        name="turboquant_dequant",
        input_names=["indices", "norms", "centroids", "signs", "scale", "dims"],
        output_names=["out"],
        source=TURBOQUANT_DEQUANT_KERNEL,
    )


def create_fused_attention_kernel():
    """Create the fused dequant+dot product Metal kernel."""
    return mx.fast.metal_kernel(
        name="turboquant_fused_attn",
        input_names=["indices", "norms", "centroids", "signs", "scale", "dims", "query"],
        output_names=["out"],
        source=TURBOQUANT_FUSED_ATTENTION_KERNEL,
    )


# Cached kernel instances
_dequant_kernel = None
_fused_attn_kernel = None


def fused_dequantize(
    indices: mx.array,
    norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    dim: int,
) -> mx.array:
    """Fused TurboQuant dequantize via Metal kernel.

    Args:
        indices: (seq_len, dim) uint8 codebook indices
        norms: (seq_len,) float32 vector norms
        centroids: (n_levels,) float32 codebook centroids
        signs: (dim,) float32 ±1 rotation signs
        dim: head dimension

    Returns:
        (seq_len, dim) dequantized vectors
    """
    global _dequant_kernel
    if _dequant_kernel is None:
        _dequant_kernel = create_dequant_kernel()

    seq_len = indices.shape[0]
    scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    dims = mx.array([dim, len(centroids)], dtype=mx.uint32)

    outputs = _dequant_kernel(
        inputs=[indices.astype(mx.uint32), norms, centroids, signs, scale, dims],
        template=[("T", mx.float32)],
        grid=(seq_len, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(seq_len, dim)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


def fused_attention_scores(
    query: mx.array,
    k_indices: mx.array,
    k_norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    dim: int,
) -> mx.array:
    """Compute attention scores without materializing dequantized keys.

    dot(query, dequant(key[i])) for all cached positions.

    Args:
        query: (dim,) single query vector
        k_indices: (seq_len, dim) uint8 key indices
        k_norms: (seq_len,) float32 key norms
        centroids: (n_levels,) codebook
        signs: (dim,) rotation signs
        dim: head dimension

    Returns:
        (seq_len,) attention scores (pre-softmax)
    """
    global _fused_attn_kernel
    if _fused_attn_kernel is None:
        _fused_attn_kernel = create_fused_attention_kernel()

    seq_len = k_indices.shape[0]
    scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    dims = mx.array([dim, len(centroids)], dtype=mx.uint32)

    outputs = _fused_attn_kernel(
        inputs=[
            k_indices.astype(mx.uint32), k_norms, centroids, signs, scale, dims,
            query.astype(mx.float32),
        ],
        template=[("T", mx.float32)],
        grid=(seq_len, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(seq_len,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]
