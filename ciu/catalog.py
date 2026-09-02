"""The models CIU knows how to run.

Every entry needs its architecture shape, not just its file size, because the
context headroom depends on the shape and that is the number users care about.

The shapes below are recorded from the GGUF metadata of the published files.
CIU re-reads them from the file once it is downloaded, so a wrong entry here
costs an inaccurate estimate before download, not a broken run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .budget import ModelShape

GIB = 1024 ** 3


@dataclass
class Model:
    id: str
    name: str
    repo: str                 # HuggingFace repo id
    filename: str
    size_bytes: int
    shape: ModelShape
    quant: str = "NF4DQ"
    notes: str = ""
    # Some models carry a multi-token-prediction head, which llama.cpp can use
    # for speculative decoding. Measured at 1.79x on an L4.
    has_mtp: bool = False
    n_ctx_train: int | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def size_gib(self) -> float:
        return self.size_bytes / GIB

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "repo": self.repo,
            "filename": self.filename,
            "size_gib": round(self.size_gib, 2),
            "quant": self.quant,
            "notes": self.notes,
            "has_mtp": self.has_mtp,
            "tags": self.tags,
        }


CATALOG: list[Model] = [
    Model(
        id="qwen35-9b-nf4dq",
        name="Qwen3.5 9B",
        repo="KayceeSamuel/Qwen3.5-9B-NF4DQ",
        filename="Qwen3.5-9B-NF4DQ.gguf",
        size_bytes=4_666_442_720,
        # Hybrid: 8 of 32 layers use full attention, the rest are linear and
        # carry a fixed recurrent state instead of a growing cache.
        shape=ModelShape(n_layer=32, n_head_kv=4, head_dim=256, n_attn_layer=8),
        notes="Runs on 8GB machines. Verified on an Apple M1.",
        n_ctx_train=262144,
        tags=["small", "hybrid"],
    ),
    Model(
        id="qwen38-27b-nf4dq",
        name="Qwen3.8 27B",
        repo="KayceeSamuel/Qwen3.8-27B-NF4DQ",
        filename="Qwen3.8-27B-NF4DQ.gguf",
        size_bytes=14_214_251_264,
        shape=ModelShape(n_layer=64, n_head_kv=4, head_dim=256, n_attn_layer=16),
        notes="Beats Q4_K_M on quality per gigabyte and decodes 25% faster.",
        has_mtp=True,
        n_ctx_train=262144,
        tags=["large", "hybrid"],
    ),
]


def by_id(model_id: str) -> Model | None:
    return next((m for m in CATALOG if m.id == model_id), None)
