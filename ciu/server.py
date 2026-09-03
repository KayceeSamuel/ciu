"""CIU: one local model, shared by everything on the machine.

Serves three things on one port:
  /            a status page
  /api/*       CIU's own endpoints for the page
  /v1/*        an OpenAI-compatible API, proxied to llama-server

The point of the proxy is a stable address. Applications point at
127.0.0.1:8674/v1 once and keep working when the model behind it changes.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from .budget import (CONTEXT_STEPS, GIB, budget_for, context_options,
                     max_fitting_context, round_context)
from .catalog import CATALOG, by_id
from .gguf_read import read_shape
from .gguf_remote import file_size as remote_size, read_remote
from .hardware import detect
from .runner import LlamaRunner, RunConfig, find_llama_server

PORT = 8674
MODEL_DIR = Path.home() / ".ciu" / "models"

app = FastAPI(title="CIU")
runner = LlamaRunner()

_download = {"active": False, "model_id": None, "pct": 0.0, "error": None}


def model_path(model) -> Path:
    return MODEL_DIR / model.filename


# Shapes are read once and cached. Reading a local GGUF memory-maps the whole
# file, and reading a remote one costs a network round trip, so neither belongs
# on a status poll that runs every 1.5 seconds.
_shape_cache: dict[str, object] = {}
_size_cache: dict[str, int] = {}


def shape_for(model):
    """The model's real shape, preferring measurement over hand-recorded values.

    Local file first, then the remote header, then the catalogue. The
    catalogue is a fallback for when the network is unavailable, not the
    source of truth: values typed by hand go stale and are easy to get wrong,
    and the numbers shown before download are what someone uses to decide
    whether to spend the bandwidth.
    """
    path = model_path(model)
    key = str(path)

    if key in _shape_cache:
        return _shape_cache[key] or model.shape

    if path.is_file():
        try:
            shape, _ = read_shape(key)
        except Exception:
            shape = None
        _shape_cache[key] = shape
        return shape or model.shape

    # Not downloaded. GGUF puts its metadata at the front of the file, so a
    # range request for the first megabyte answers the same question.
    remote_key = f"{model.repo}/{model.filename}"
    if remote_key in _shape_cache:
        return _shape_cache[remote_key] or model.shape

    try:
        shape, _ = read_remote(model.repo, model.filename)
    except Exception:
        shape = None
    _shape_cache[remote_key] = shape
    return shape or model.shape


def size_for(model) -> int:
    """Actual file size, from disk or from HuggingFace, falling back to the
    catalogue. A wrong size here is a wrong fit decision."""
    path = model_path(model)
    if path.is_file():
        return path.stat().st_size

    key = f"{model.repo}/{model.filename}"
    if key not in _size_cache:
        _size_cache[key] = remote_size(model.repo, model.filename) or 0
    return _size_cache[key] or model.size_bytes


# ----------------------------------------------------------------- CIU API

def _resident_bytes() -> int:
    """Memory the loaded model is holding, weights plus cache plus overhead."""
    if runner.state != "ready" or not runner.config:
        return 0
    model = next((m for m in CATALOG
                  if runner.config.model_path.endswith(m.filename)), None)
    if not model:
        return 0
    shape = shape_for(model)
    b = budget_for(size_for(model), shape, 0)
    return b.fixed_bytes + b.kv_bytes_at(runner.config.n_ctx)


@app.get("/api/status")
def status():
    # Runs on every poll, so everything in here must be cheap.
    resident = _resident_bytes()
    hw = detect(resident)

    # free_bytes now excludes whatever the loaded model holds, so this is
    # genuinely what a second model, or a longer context, could use.
    available = hw.free_bytes

    models = []
    for m in CATALOG:
        path = model_path(m)
        downloaded = path.is_file()
        shape = shape_for(m)
        b = budget_for(size_for(m), shape, available)
        n_ctx_train = getattr(shape, "n_ctx_train", None) or m.n_ctx_train
        opts = context_options(b, n_ctx_train)
        # Default to the largest context that fits entirely. Zero means the
        # model will not load at all on the memory currently free.
        max_ctx = max_fitting_context(b, n_ctx_train)
        default_ctx = max_ctx or (opts[0]["n_ctx"] if opts else 0)

        entry = m.to_dict()
        entry["size_gib"] = round(size_for(m) / GIB, 2)
        entry.update({
            "downloaded": downloaded,
            "budget": b.to_dict(),
            "suggested_ctx": round_context(b.max_context),
            "kv_per_token": b.kv_per_token,
            "contexts": opts,
            "default_ctx": default_ctx,
            "max_ctx": max_ctx,
            "n_layer": shape.n_layer,
            "n_ctx_train": n_ctx_train,
        })
        models.append(entry)

    return {
        "hardware": hw.to_dict(),
        "resident_gib": round(resident / GIB, 2),
        "models": models,
        "runner": runner.status(),
        "download": _download,
        "llama_server": find_llama_server(),
        "base_url": f"http://127.0.0.1:{PORT}/v1",
    }


@app.post("/api/download/{model_id}")
def download(model_id: str):
    model = by_id(model_id)
    if not model:
        raise HTTPException(404, "unknown model")
    if _download["active"]:
        raise HTTPException(409, "a download is already running")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def work():
        _download.update(active=True, model_id=model_id, pct=0.0, error=None)
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=model.repo,
                filename=model.filename,
                local_dir=str(MODEL_DIR),
            )
            _download["pct"] = 100.0
        except Exception as e:
            _download["error"] = str(e)
        finally:
            _download["active"] = False

    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


@app.post("/api/load/{model_id}")
def load(model_id: str, n_ctx: int | None = None, mtp: bool = False):
    model = by_id(model_id)
    if not model:
        raise HTTPException(404, "unknown model")

    path = model_path(model)
    if not path.is_file():
        raise HTTPException(400, "model not downloaded")

    hw = detect(_resident_bytes())
    shape = shape_for(model)
    b = budget_for(size_for(model), shape, hw.free_bytes)

    n_ctx_train = model.n_ctx_train
    max_ctx = max_fitting_context(b, n_ctx_train)

    # The estimate is advisory, never a veto. It is built from a model of how
    # llama.cpp allocates, and that model is approximate: an earlier version
    # of it refused everything on an 8GB machine that in fact runs the model
    # fine. So CIU says what it expects, then lets the backend be the judge.
    # If it really does not fit, the OOM is caught and reported plainly.
    ctx = n_ctx or max_ctx or CONTEXT_STEPS[0]

    expected = (b.fixed_bytes + b.kv_bytes_at(ctx)) / GIB
    tight = expected > hw.free_gib

    # A smaller batch shrinks the compute buffer, which is worth doing when
    # memory is the binding constraint.
    runner.start(RunConfig(
        model_path=str(path),
        n_ctx=ctx,
        backend=hw.backend,
        use_mtp=mtp and model.has_mtp,
        n_batch=256 if tight else None,
    ))
    return {
        "loading": True,
        "n_ctx": ctx,
        "expected_gib": round(expected, 2),
        "free_gib": hw.free_gib,
        "tight": tight,
    }


@app.post("/api/unload")
def unload():
    runner.stop()
    return {"stopped": True}


# ------------------------------------------------- OpenAI-compatible proxy

@app.api_route("/v1/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    """Forward to llama-server, preserving streaming.

    Applications hold this URL. Which model answers is CIU's business.
    """
    if runner.state != "ready":
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": f"No model loaded (state: {runner.state}). "
                           f"Open http://127.0.0.1:{PORT} to load one.",
                "type": "ciu_not_ready",
            }},
        )

    url = f"{runner.base_url}/v1/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}

    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(request.method, url, content=body,
                               headers=headers,
                               params=request.query_params)
    upstream = await client.send(req, stream=True)

    # A request that overruns the loaded context comes back from llama-server
    # as a 400 with a message about tokens. Applications tend to surface that
    # raw, which tells the user nothing useful, so it is rewritten here into
    # the thing they actually need to do.
    if upstream.status_code == 400 and runner.config:
        raw = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        text = raw.decode("utf-8", "replace").lower()
        if "context" in text or "n_ctx" in text or "exceed" in text:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": (
                        f"This conversation has filled the "
                        f"{runner.config.n_ctx}-token context. Start a new "
                        f"conversation, or reload the model with a larger "
                        f"context if memory allows."),
                    "type": "ciu_context_full",
                    "n_ctx": runner.config.n_ctx,
                }},
            )
        return JSONResponse(status_code=400,
                            content={"error": {"message": raw.decode("utf-8", "replace")}})

    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers={k: v for k, v in upstream.headers.items()
                 if k.lower() not in ("content-length", "transfer-encoding")},
    )


# ------------------------------------------------------------------- page

@app.get("/", response_class=HTMLResponse)
def index():
    page = Path(__file__).parent / "static" / "index.html"
    return page.read_text()


def main():
    import uvicorn
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"CIU on http://127.0.0.1:{PORT}")
    print(f"API   http://127.0.0.1:{PORT}/v1")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
