"""Layer-adaptive TurboQuant: FP16 for critical layers, compressed for the rest.

First and last N layers use standard KVCache (FP16) — these are most sensitive
to quantization error. Middle layers use TurboQuantKVCache.

This matches the approach in turboquant_plus "layer-adaptive mode 2".
"""

from mlx_lm.models.cache import KVCache
from turboquant_mlx.cache import TurboQuantKVCache


def make_adaptive_cache(
    num_layers: int,
    bits: int = 3,
    fp16_layers: int = 4,
    seed: int = 42,
    fused: bool = False,
):
    """Create layer-adaptive cache list.

    Args:
        num_layers: total number of transformer layers
        bits: TurboQuant bits for compressed layers (1-4)
        fp16_layers: number of first AND last layers to keep in FP16
        seed: random seed for rotation
        fused: use fused attention path for compressed layers

    Returns:
        list of cache objects (one per layer)
    """
    caches = []
    for i in range(num_layers):
        if i < fp16_layers or i >= num_layers - fp16_layers:
            caches.append(KVCache())
        else:
            caches.append(TurboQuantKVCache(bits=bits, seed=seed, fused=fused))
    return caches
