"""Start, watch and stop llama-server.

CIU is not an inference engine. llama-server already speaks the OpenAI API,
handles streaming and manages parallel slots. What it does not do is decide
which model to load, work out what fits, or stay at one address while the
model behind it changes. That is what this supervises.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

BACKEND_FLAGS = {
    "cuda":  ["-ngl", "99"],
    "metal": ["-ngl", "99"],
    "cpu":   ["-ngl", "0"],
}


def find_llama_server() -> str | None:
    """Locate llama-server: PATH, env override, or the usual build locations."""
    env = os.environ.get("CIU_LLAMA_SERVER")
    if env and Path(env).is_file():
        return env

    found = shutil.which("llama-server")
    if found:
        return found

    for candidate in (
        Path.home() / "llama.cpp/build/bin/llama-server",
        Path.home() / ".ciu/bin/llama-server",
        Path("/usr/local/bin/llama-server"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def free_port(start: int = 8700) -> int:
    for port in range(start, start + 200):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port in range")


@dataclass
class RunConfig:
    model_path: str
    n_ctx: int
    backend: str
    use_mtp: bool = False
    draft_n_max: int = 4
    flash_attn: bool = True


class LlamaRunner:
    """One llama-server process, with its lifecycle and log."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.config: RunConfig | None = None
        self.log: list[str] = []
        self.state = "stopped"      # stopped | loading | ready | error
        self.error: str | None = None
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str | None:
        return f"http://127.0.0.1:{self.port}" if self.port else None

    def _build_args(self, binary: str, cfg: RunConfig, port: int) -> list[str]:
        args = [
            binary,
            "-m", cfg.model_path,
            "-c", str(cfg.n_ctx),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--jinja",
        ]
        args += BACKEND_FLAGS.get(cfg.backend, BACKEND_FLAGS["cpu"])
        if cfg.flash_attn and cfg.backend != "cpu":
            args += ["-fa", "on"]
        if cfg.use_mtp:
            # Measured 1.79x on an L4, 1.54x on an A100. Depth 4 was the peak
            # of a sweep; 8 collapsed badly, so it is not exposed as a dial.
            args += ["--spec-type", "draft-mtp",
                     "--spec-draft-n-max", str(cfg.draft_n_max)]
        return args

    def start(self, cfg: RunConfig, timeout: int = 900) -> None:
        with self._lock:
            if self.state in ("loading", "ready"):
                self.stop()

            binary = find_llama_server()
            if not binary:
                self.state = "error"
                self.error = (
                    "llama-server not found. Build the NF4DQ fork, or set "
                    "CIU_LLAMA_SERVER to its path."
                )
                return

            self.port = free_port()
            self.config = cfg
            self.log = []
            self.error = None
            self.state = "loading"

            args = self._build_args(binary, cfg, self.port)
            self.proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )

        threading.Thread(target=self._pump_log, daemon=True).start()
        threading.Thread(target=self._await_ready, args=(timeout,),
                         daemon=True).start()

    def _pump_log(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        for line in self.proc.stdout:
            self.log.append(line.rstrip())
            # Loading a 13GiB model produces a lot of output; keep the tail.
            if len(self.log) > 400:
                del self.log[:200]

    def _await_ready(self, timeout: int) -> None:
        """Poll /health. The port opens long before the weights are loaded, so
        watching the socket alone reports ready far too early."""
        import httpx

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                self.state = "error"
                self.error = "llama-server exited during load"
                return
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    self.state = "ready"
                    return
            except Exception:
                pass
            time.sleep(1)

        self.state = "error"
        self.error = f"model did not load within {timeout}s"

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.port = None
        self.state = "stopped"

    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "base_url": self.base_url,
            "model_path": self.config.model_path if self.config else None,
            "n_ctx": self.config.n_ctx if self.config else None,
            "mtp": self.config.use_mtp if self.config else False,
            "log_tail": self.log[-25:],
        }
