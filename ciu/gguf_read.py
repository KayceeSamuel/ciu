"""Read a model's real shape out of its GGUF file.

Two reasons this exists.

First, the shapes in catalog.py are recorded by hand and can be wrong. A wrong
shape means a wrong context estimate, which is the one number CIU exists to
get right. Once a file is on disk we read the truth from it.

Second, the stock gguf package does not know NF4DQ. It raises
`ValueError: 43 is not a valid GGMLQuantizationType` on any file using it,
which means no standard Python tooling can inspect the published models. We
register the type here so CIU can read them, and the same registration belongs
upstream in gguf-py/gguf/constants.py.
"""

from __future__ import annotations

from .budget import ModelShape

NF4DQ_TYPE_ID = 43
NF4DQ_BLOCK_SIZE = 1024
NF4DQ_TYPE_SIZE = 532


def register_nf4dq() -> bool:
    """Teach the gguf package about type 43. Returns False if gguf is absent."""
    try:
        from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
    except ImportError:
        return False

    if NF4DQ_TYPE_ID in GGMLQuantizationType._value2member_map_:
        return True

    obj = int.__new__(GGMLQuantizationType, NF4DQ_TYPE_ID)
    obj._name_ = "NF4DQ"
    obj._value_ = NF4DQ_TYPE_ID
    GGMLQuantizationType._value2member_map_[NF4DQ_TYPE_ID] = obj
    GGML_QUANT_SIZES[obj] = (NF4DQ_BLOCK_SIZE, NF4DQ_TYPE_SIZE)
    return True


def _kv(reader, *names):
    """First matching metadata value, or None. Key names vary by architecture."""
    for name in names:
        field = reader.fields.get(name)
        if field is None:
            continue
        try:
            return field.parts[field.data[0]].tolist()[0]
        except (IndexError, AttributeError, TypeError):
            continue
    return None


def read_shape(path: str) -> tuple[ModelShape | None, dict]:
    """Return (shape, raw metadata) for a GGUF file.

    Falls back to (None, {}) if the file cannot be read, so callers keep the
    catalogue's recorded shape rather than failing.
    """
    if not register_nf4dq():
        return None, {}

    try:
        from gguf import GGUFReader
        r = GGUFReader(path)
    except Exception:
        return None, {}

    arch = None
    f = r.fields.get("general.architecture")
    if f is not None:
        try:
            arch = bytes(f.parts[f.data[0]]).decode("utf-8")
        except Exception:
            arch = None

    prefix = f"{arch}." if arch else ""

    n_layer = _kv(r, f"{prefix}block_count")
    n_head_kv = _kv(r, f"{prefix}attention.head_count_kv")
    n_head = _kv(r, f"{prefix}attention.head_count")
    n_embd = _kv(r, f"{prefix}embedding_length")
    n_ctx_train = _kv(r, f"{prefix}context_length")

    if not (n_layer and n_head_kv and n_embd and n_head):
        return None, {"architecture": arch}

    head_dim = _kv(r, f"{prefix}attention.key_length") or (n_embd // n_head)

    # Hybrid architectures interleave full attention with linear attention.
    # Only the full-attention layers keep a cache that grows per token, so
    # counting all layers would overstate KV cost several-fold. llama.cpp
    # records the recurrent layer count where the architecture has one.
    n_attn_layer = None
    linear_count = _kv(r, f"{prefix}recurrent_layer_count",
                       f"{prefix}linear_attention.layer_count")
    if linear_count and 0 < linear_count < n_layer:
        n_attn_layer = n_layer - linear_count

    meta = {
        "architecture": arch,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "n_embd": n_embd,
        "head_dim": head_dim,
        "n_ctx_train": n_ctx_train,
        "n_attn_layer": n_attn_layer,
    }
    return ModelShape(n_layer=n_layer, n_head_kv=n_head_kv,
                      head_dim=head_dim, n_attn_layer=n_attn_layer), meta


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m ciu.gguf_read MODEL.gguf")
        raise SystemExit(1)
    shape, meta = read_shape(sys.argv[1])
    if shape is None:
        print("could not read shape; metadata:", meta)
    else:
        from .budget import kv_bytes_per_token
        for k, v in meta.items():
            print(f"{k:16s} {v}")
        print(f"{'cache layers':16s} {shape.cache_layers}")
        print(f"{'KV per token':16s} {kv_bytes_per_token(shape)} bytes")
