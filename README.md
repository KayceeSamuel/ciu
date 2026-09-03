# CIU

One local model, shared by everything on your machine.

CIU works out what your hardware can run, downloads it, keeps it loaded, and
puts an OpenAI-compatible API in front of it at a fixed address. Any tool that
speaks OpenAI can point at that address: a chat client, a Blender addon, an
editor extension. They share one model instance instead of each loading their
own and running you out of memory.

## Installing

One line. It installs into `~/.ciu` and touches nothing else.

**macOS and Linux**

    curl -fsSL https://raw.githubusercontent.com/KayceeSamuel/ciu/main/install.sh | sh

**Windows** (PowerShell)

    irm https://raw.githubusercontent.com/KayceeSamuel/ciu/main/install.ps1 | iex

Then, in a new terminal:

    ciu

Your browser opens at http://127.0.0.1:8674, where you pick a model that fits
your machine. Everything runs locally: no account, no API key, nothing leaves
the machine.

To remove it: `rm -rf ~/.ciu`, or on Windows `Remove-Item -Recurse $HOME\.ciu`

Requires macOS on Apple Silicon, Linux x86_64, or 64-bit Windows.

On Windows and Linux you want an NVIDIA GPU. CIU has CUDA and Metal kernels
but no Vulkan kernel yet, so AMD and Intel GPUs fall back to a CPU path that
runs at well under one token a second.

## Installing from source

Only needed if you want to change CIU itself.

    git clone https://github.com/KayceeSamuel/ciu
    cd ciu
    pip install fastapi uvicorn httpx huggingface_hub gguf
    python run.py

CIU needs `llama-server` from the NF4DQ fork. It looks on PATH, in
`~/.ciu/bin/`, and in `~/llama.cpp/build/bin/`; otherwise set
`CIU_LLAMA_SERVER` to its path. Prebuilt binaries are at
https://github.com/KayceeSamuel/llama.cpp/releases

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
