"""Read a GGUF's metadata without downloading the model.

GGUF puts its header and key-value block at the front of the file, before any
tensor data. Everything CIU needs to size a model lives there: layer count,
head counts, the hybrid attention split. So a range request for the first few
hundred kilobytes answers the question a 13 GiB download would.

That matters because the numbers shown before download are what someone uses
to decide whether to spend the bandwidth. Hand-recorded catalogue values go
stale and are easy to get wrong; this reads the truth from the file that will
actually be downloaded.

The format is documented at
https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .budget import ModelShape

MAGIC = b"GGUF"

# Value type tags from the GGUF spec.
(T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL,
 T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64) = range(13)

_FIXED = {
    T_UINT8: ("<B", 1), T_INT8: ("<b", 1),
    T_UINT16: ("<H", 2), T_INT16: ("<h", 2),
    T_UINT32: ("<I", 4), T_INT32: ("<i", 4),
    T_FLOAT32: ("<f", 4), T_BOOL: ("<?", 1),
    T_UINT64: ("<Q", 8), T_INT64: ("<q", 8),
    T_FLOAT64: ("<d", 8),
}

# Grow the read if the metadata block runs past the first chunk. Vocabularies
# push this up: a 150k-token tokenizer list is several megabytes on its own.
CHUNK = 1 << 20          # 1 MiB
MAX_READ = 24 << 20      # give up rather than pull the whole file


class _Cursor:
    """Reads forward through a buffer, raising if it runs past the end."""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise EOFError
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def scalar(self, tag: int):
        if tag in _FIXED:
            fmt, size = _FIXED[tag]
            return struct.unpack(fmt, self.take(size))[0]
        if tag == T_STRING:
            n = struct.unpack("<Q", self.take(8))[0]
            return self.take(n).decode("utf-8", "replace")
        raise ValueError(f"unknown GGUF value type {tag}")

    def value(self):
        tag = struct.unpack("<I", self.take(4))[0]
        if tag != T_ARRAY:
            return self.scalar(tag)

        # Arrays: element type, count, then the elements. Long ones (token
        # lists) are skipped rather than materialised; nothing CIU needs is
        # in an array.
        elem = struct.unpack("<I", self.take(4))[0]
        count = struct.unpack("<Q", self.take(8))[0]
        if elem in _FIXED:
            _, size = _FIXED[elem]
            self.take(size * count)
            return f"<{count} values>"
        if elem == T_STRING:
            for _ in range(count):
                n = struct.unpack("<Q", self.take(8))[0]
                self.take(n)
            return f"<{count} strings>"
        raise ValueError(f"unknown array element type {elem}")


def parse_metadata(buf: bytes) -> dict:
    """Parse the KV block from the start of a GGUF file.

    Raises EOFError if buf does not reach the end of the metadata, which the
    caller answers by reading more.
    """
    c = _Cursor(buf)
    if c.take(4) != MAGIC:
        raise ValueError("not a GGUF file")

    version = struct.unpack("<I", c.take(4))[0]
    if version < 2:
        raise ValueError(f"GGUF version {version} is too old to parse")

    n_tensors = struct.unpack("<Q", c.take(8))[0]
    n_kv = struct.unpack("<Q", c.take(8))[0]

    kv: dict = {"_version": version, "_n_tensors": n_tensors}
    for _ in range(n_kv):
        klen = struct.unpack("<Q", c.take(8))[0]
        key = c.take(klen).decode("utf-8", "replace")
        kv[key] = c.value()
    return kv


def shape_from_metadata(kv: dict) -> tuple[ModelShape | None, dict]:
    """Turn a parsed KV block into a ModelShape."""
    arch = kv.get("general.architecture")
    if not arch:
        return None, {}
    p = f"{arch}."

    n_layer = kv.get(f"{p}block_count")
    n_head = kv.get(f"{p}attention.head_count")
    n_head_kv = kv.get(f"{p}attention.head_count_kv")
    n_embd = kv.get(f"{p}embedding_length")
    if not (n_layer and n_head and n_head_kv and n_embd):
        return None, {"architecture": arch}

    head_dim = kv.get(f"{p}attention.key_length") or (n_embd // n_head)

    # Hybrid split. Qwen states an interval: every Nth layer uses full
    # attention, the rest carry a fixed state-space state. Interval 4 over 32
    # layers means 8 cache-bearing layers, so a quarter of the KV cost.
    n_attn_layer = None
    interval = kv.get(f"{p}full_attention_interval")
    if interval and interval > 1:
        n_attn_layer = n_layer // interval
    else:
        linear = kv.get(f"{p}recurrent_layer_count")
        if linear and 0 < linear < n_layer:
            n_attn_layer = n_layer - linear

    meta = {
        "architecture": arch,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "n_embd": n_embd,
        "head_dim": head_dim,
        "n_ctx_train": kv.get(f"{p}context_length"),
        "full_attention_interval": interval,
        "n_attn_layer": n_attn_layer,
        "hybrid": any(k.startswith(f"{p}ssm.") for k in kv),
        "name": kv.get("general.name"),
    }

    # A hybrid model whose split we could not read would be sized as if every
    # layer caches, overstating context cost several-fold. Refuse rather than
    # report a confidently wrong number.
    if meta["hybrid"] and n_attn_layer is None:
        return None, meta

    return ModelShape(n_layer=n_layer, n_head_kv=n_head_kv,
                      head_dim=head_dim, n_attn_layer=n_attn_layer), meta


def _resolve_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def read_remote(repo: str, filename: str, timeout: float = 20.0
                ) -> tuple[ModelShape | None, dict]:
    """Fetch just enough of a GGUF from HuggingFace to size it.

    Returns (shape, metadata). Reads in 1 MiB steps, growing only if the
    metadata block did not fit, so the usual cost is one request.
    """
    import httpx

    url = _resolve_url(repo, filename)
    size = CHUNK
    buf = b""

    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        while size <= MAX_READ:
            r = client.get(url, headers={"Range": f"bytes=0-{size - 1}"})
            if r.status_code not in (200, 206):
                return None, {"error": f"HTTP {r.status_code}"}
            buf = r.content

            # A 200 means the server ignored the range and sent everything.
            # Parse what arrived rather than asking again.
            try:
                kv = parse_metadata(buf)
            except EOFError:
                if r.status_code == 200:
                    return None, {"error": "metadata truncated"}
                size *= 4
                continue
            except ValueError as e:
                return None, {"error": str(e)}

            shape, meta = shape_from_metadata(kv)
            meta["bytes_read"] = len(buf)
            return shape, meta

    return None, {"error": "metadata larger than the read limit"}


def file_size(repo: str, filename: str, timeout: float = 15.0) -> int | None:
    """Total size of the remote file, from a HEAD request."""
    import httpx
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            r = client.head(_resolve_url(repo, filename))
            n = r.headers.get("content-length") or r.headers.get("x-linked-size")
            return int(n) if n else None
    except Exception:
        return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m ciu.gguf_remote REPO FILENAME")
        raise SystemExit(1)

    shape, meta = read_remote(sys.argv[1], sys.argv[2])
    for k, v in meta.items():
        print(f"{k:26s} {v}")
    if shape:
        from .budget import kv_bytes_per_token
        print(f"{'cache layers':26s} {shape.cache_layers} of {shape.n_layer}")
        print(f"{'KV per token':26s} {kv_bytes_per_token(shape)} bytes")
    size = file_size(sys.argv[1], sys.argv[2])
    if size:
        print(f"{'file size':26s} {size / 1024**3:.2f} GiB")
