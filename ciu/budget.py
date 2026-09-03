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

MIB = 1024 ** 2

# Beyond weights and cache, llama.cpp allocates a compute buffer, a logits
# buffer and some backend scratch. The compute buffer scales with the BATCH
# size, not the context, which is why -b 2048 costs hundreds of megabytes and
# -b 128 costs tens.
#
# A flat one-gigabyte reserve was the earlier guess here and it was both wrong
# and harmful: it refused models that run perfectly well, which on an 8GB
# machine meant refusing everything. The estimate below is smaller and scales
# with batch. It can still be wrong, so it is advisory: the user may load
# anyway and the backend's own out-of-memory error is the real answer.
BASE_OVERHEAD = 192 * MIB          # backend scratch, logits, allocator slack
DEFAULT_BATCH = 512


def overhead_bytes(n_embd: int = 4096, n_batch: int = DEFAULT_BATCH) -> int:
    """Rough non-weight, non-cache allocation.

    The compute buffer holds a few activation tensors of batch x embedding in
    fp16. Four is a workable multiplier across the models measured.
    """
    compute = 4 * n_batch * n_embd * 2
    return BASE_OVERHEAD + compute


# Kept for callers that have no shape to hand.
OVERHEAD_BYTES = overhead_bytes()


@dataclass
class ModelShape:
    """The parts of a model's architecture that drive memory use."""
    n_layer: int
    n_head_kv: int
    head_dim: int
    n_embd: int = 4096
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
               kv_bits: int = 16, n_batch: int = DEFAULT_BATCH) -> Budget:
    return Budget(
        weights_bytes=weights_bytes,
        overhead_bytes=overhead_bytes(shape.n_embd, n_batch),
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


# Context lengths worth offering. Users recognise these; they do not recognise
# 41,318.
CONTEXT_STEPS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)


def context_options(b: "Budget", n_ctx_train: int | None = None) -> list[dict]:
    """Every offerable context length with what it would cost.

    Includes lengths that do not fit, marked as such, because "8k needs 5.6 GiB
    and you have 3.4" is more useful than hiding the option.
    """
    cap = n_ctx_train or CONTEXT_STEPS[-1]
    out = []
    for n in CONTEXT_STEPS:
        if n > cap:
            break
        total = b.fixed_bytes + b.kv_bytes_at(n)
        out.append({
            "n_ctx": n,
            "total_gib": round(total / GIB, 2),
            "kv_gib": round(b.kv_bytes_at(n) / GIB, 3),
            "fits": total <= b.available_bytes,
        })
    return out


def max_fitting_context(b: "Budget", n_ctx_train: int | None = None) -> int:
    """Largest offered context length that fits entirely in device memory.

    Zero means the model will not load at all: the weights and overhead alone
    exceed what is free, so no context length helps.
    """
    cap = n_ctx_train or CONTEXT_STEPS[-1]
    best = 0
    for n in CONTEXT_STEPS:
        if n > cap:
            break
        if b.fixed_bytes + b.kv_bytes_at(n) <= b.available_bytes:
            best = n
    return best
