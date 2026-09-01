# CIU

One local model, shared by everything on your machine.

CIU works out what your hardware can run, downloads it, keeps it loaded, and
puts an OpenAI-compatible API in front of it at a fixed address. Any tool that
speaks OpenAI can point at that address: a chat client, a Blender addon, an
editor extension. They share one model instance instead of each loading their
own and running you out of memory.

## What it needs

A build of the NF4DQ llama.cpp fork:

    git clone https://github.com/KayceeSamuel/llama.cpp
    cd llama.cpp
    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release  # omit -DGGML_CUDA on Apple Silicon
    cmake --build build -j --target llama-server

CIU looks for `llama-server` on PATH, at `~/llama.cpp/build/bin/`, and at
`~/.ciu/bin/`. Otherwise set `CIU_LLAMA_SERVER` to its path.

## Running it

    pip install fastapi uvicorn httpx huggingface_hub gguf
    python run.py

Then open http://127.0.0.1:8674

## Connecting a tool

Base URL: `http://127.0.0.1:8674/v1`. Any API key. For example:

    curl http://127.0.0.1:8674/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"hello"}]}'

## The number CIU exists to get right

Model size is not the useful question. "Will it run and how much conversation
will I get" is. Those differ because the KV cache grows with context and can
exceed the weights on a long conversation.

Hybrid models change the answer a lot. Qwen3.5-9B uses linear attention in 24
of its 32 layers, and those layers carry a fixed recurrent state rather than a
per-token cache, so its context costs roughly a quarter of what a conventional
model of the same shape would. CIU reads the layer split from the GGUF and
does the arithmetic rather than guessing from file size.

## Status

Working: hardware and backend detection, budget arithmetic, model download,
llama-server supervision, the OpenAI proxy, the status page.

Not yet: model switching when two apps want different models, idle eviction,
tool use, an installer.

Backends: CUDA and Metal. No Vulkan yet, so AMD and Intel GPUs fall back to a
scalar CPU path that is a correctness gate, not a usable speed.
