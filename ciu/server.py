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

from .budget import GIB, budget_for, round_context
from .catalog import CATALOG, by_id
from .gguf_read import read_shape
from .hardware import detect
from .runner import LlamaRunner, RunConfig, find_llama_server

PORT = 8674
MODEL_DIR = Path.home() / ".ciu" / "models"

app = FastAPI(title="CIU")
runner = LlamaRunner()

_download = {"active": False, "model_id": None, "pct": 0.0, "error": None}


def model_path(model) -> Path:
    return MODEL_DIR / model.filename


# Reading a GGUF memory-maps the whole file. The shape never changes, so it is
# read once per path and cached. Without this the status poll maps a multi-GiB
# file every 1.5 seconds, which stalls hard on a machine that is already tight
# on memory.
_shape_cache: dict[str, object] = {}


def shape_for(model):
    """Prefer the shape in the file over the one recorded in the catalogue."""
    path = model_path(model)
    key = str(path)

    if key in _shape_cache:
        return _shape_cache[key] or model.shape

    if not path.is_file():
        return model.shape

    try:
        shape, _ = read_shape(key)
    except Exception:
        shape = None

    _shape_cache[key] = shape
    return shape or model.shape


# ----------------------------------------------------------------- CIU API

@app.get("/api/status")
def status():
    # Runs on every poll, so everything in here must be cheap.
    hw = detect()

    # While a model is loaded its memory is already accounted for in free_bytes
    # on CUDA. On unified memory it is too, since it comes from system RAM.
    available = hw.free_bytes

    models = []
    for m in CATALOG:
        path = model_path(m)
        downloaded = path.is_file()
        shape = shape_for(m)
        b = budget_for(m.size_bytes, shape, available)
        entry = m.to_dict()
        entry.update({
            "downloaded": downloaded,
            "budget": b.to_dict(),
            "suggested_ctx": round_context(b.max_context),
            "kv_per_token": b.kv_per_token,
        })
        models.append(entry)

    return {
        "hardware": hw.to_dict(),
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

    hw = detect()
    shape = shape_for(model)
    b = budget_for(model.size_bytes, shape, hw.free_bytes)
    if not b.fits:
        raise HTTPException(
            400,
            f"{model.name} needs {b.fixed_bytes / GIB:.1f} GiB but only "
            f"{hw.free_gib:.1f} GiB is free",
        )

    ctx = n_ctx or round_context(b.max_context)
    if ctx > b.max_context:
        raise HTTPException(
            400, f"{ctx} tokens needs more memory than is free "
                 f"(max {b.max_context})")

    runner.start(RunConfig(
        model_path=str(path),
        n_ctx=ctx,
        backend=hw.backend,
        use_mtp=mtp and model.has_mtp,
    ))
    return {"loading": True, "n_ctx": ctx}


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
