"""Work out what fits.

The question a user actually has is not "how many gigabytes is this model",
it is "will it run on my machine, and how much conversation will I get".
Those are different questions because the KV cache grows with context and can
easily exceed the weights on a long conversation.

Hybrid models change the answer substantially. Qwen3.5-9B uses linear
attention in 24 of its 32 layers, and linear-attention layers carry a fixed
recurrent state rather than a cache that grows per token. So its KV cost is
roughly a quarter of what a conventional model of the same shape would need,
and the context headroom is correspondingly larger.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024 ** 3

# llama.cpp allocates compute buffers, the logits buffer and CUDA/Metal
# scratch beyond the weights and cache. Measured at roughly 600MB-1GB on the
# models tested; we reserve 1GiB so the estimate errs toward refusing rather
# than promising a model that then fails to allocate.
OVERHEAD_BYTES = 1 * GIB


@dataclass
class ModelShape:
    """The parts of a model's architecture that drive memory use."""
    n_layer: int
    n_head_kv: int
    head_dim: int
    # Layers using full attention. On a conventional model this equals
    # n_layer. On a hybrid it is only the subset that keeps a growing cache.
    n_attn_layer: int | None = None

    @property
    def cache_layers(self) -> int:
        return self.n_attn_layer if self.n_attn_layer is not None else self.n_layer


def kv_bytes_per_token(shape: ModelShape, kv_bits: int = 16) -> int:
    """Bytes of KV cache each token of context costs.

    Two tensors (K and V) per attention layer, n_head_kv heads of head_dim
    each. kv_bits is 16 by default; llama.cpp can quantise the cache to 8 with
    --cache-type-k/v, which halves this.
    """
    bytes_per_elem = kv_bits // 8
    return 2 * shape.cache_layers * shape.n_head_kv * shape.head_dim * bytes_per_elem


@dataclass
class Budget:
    """What a given model costs on a given machine."""
    weights_bytes: int
    overhead_bytes: int
    kv_per_token: int
    available_bytes: int

    @property
    def fixed_bytes(self) -> int:
        return self.weights_bytes + self.overhead_bytes

    @property
    def fits(self) -> bool:
        return self.available_bytes > self.fixed_bytes

    @property
    def max_context(self) -> int:
        """Tokens of context the remaining memory will hold."""
        spare = self.available_bytes - self.fixed_bytes
        if spare <= 0 or self.kv_per_token <= 0:
            return 0
        return int(spare // self.kv_per_token)

    def kv_bytes_at(self, n_ctx: int) -> int:
        return n_ctx * self.kv_per_token

    def free_at(self, n_ctx: int) -> int:
        return self.available_bytes - self.fixed_bytes - self.kv_bytes_at(n_ctx)

    def to_dict(self, n_ctx: int | None = None) -> dict:
        d = {
            "fits": self.fits,
            "weights_gib": round(self.weights_bytes / GIB, 2),
            "overhead_gib": round(self.overhead_bytes / GIB, 2),
            "available_gib": round(self.available_bytes / GIB, 2),
            "kv_bytes_per_token": self.kv_per_token,
            "max_context": self.max_context,
        }
        if n_ctx:
            d["n_ctx"] = n_ctx
            d["kv_gib"] = round(self.kv_bytes_at(n_ctx) / GIB, 2)
            d["free_gib"] = round(self.free_at(n_ctx) / GIB, 2)
        return d


def budget_for(weights_bytes: int, shape: ModelShape, available_bytes: int,
               kv_bits: int = 16) -> Budget:
    return Budget(
        weights_bytes=weights_bytes,
        overhead_bytes=OVERHEAD_BYTES,
        kv_per_token=kv_bytes_per_token(shape, kv_bits),
        available_bytes=available_bytes,
    )


def round_context(n: int) -> int:
    """Snap to a sensible context length at or below n.

    Users recognise 8k and 32k. They do not recognise 41,318.
    """
    for step in (131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512):
        if n >= step:
            return step
    return 0


def human(n_bytes: int) -> str:
    return f"{n_bytes / GIB:.2f} GiB"


def human_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return str(n)
