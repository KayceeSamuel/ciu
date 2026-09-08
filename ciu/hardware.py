"""Detect what the machine can actually run.

Three things matter: which llama.cpp backend to use, how much memory is
available to it, and how much of that is already spoken for. Apple Silicon
needs separate handling because there is no discrete VRAM: the GPU draws from
system memory and the driver caps how much it will hand over.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict

GIB = 1024 ** 3


@dataclass
class Hardware:
    backend: str          # "cuda" | "metal" | "cpu"
    device_name: str
    total_bytes: int      # memory the backend can address
    free_bytes: int       # of that, currently unused
    unified: bool         # GPU shares system memory (Apple Silicon)
    note: str = ""        # anything the user should know

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def free_gib(self) -> float:
        return self.free_bytes / GIB

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_gib"] = round(self.total_gib, 2)
        d["free_gib"] = round(self.free_gib, 2)
        return d


def _run(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _detect_cuda() -> Hardware | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if not out or not out.strip():
        return None

    # First GPU only. Multi-GPU is a later problem: llama.cpp can split a model
    # across devices, but the budget arithmetic below assumes one pool.
    line = out.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None

    name, total_mib, free_mib = parts[0], int(parts[1]), int(parts[2])
    count = len(out.strip().splitlines())
    note = "" if count == 1 else f"{count} GPUs found; using the first only"

    return Hardware(
        backend="cuda",
        device_name=name,
        total_bytes=total_mib * 1024 * 1024,
        free_bytes=free_mib * 1024 * 1024,
        unified=False,
        note=note,
    )


def _detect_metal() -> Hardware | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None

    total = 0
    out = _run(["sysctl", "-n", "hw.memsize"])
    if out:
        total = int(out.strip())
    if not total:
        return None

    chip = "Apple Silicon"
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if out:
        chip = out.strip()

    # Metal will not hand the whole machine to one process. The driver reports
    # recommendedMaxWorkingSetSize, which is about 75% on 8GB machines and
    # closer to 80% higher up. We use 75% as the conservative figure rather
    # than promising memory the driver will refuse.
    addressable = int(total * 0.75)

    # Free memory: pages that are free or reclaimable. macOS counts inactive
    # pages as available and normally they are.
    #
    # But llama.cpp memory-maps the GGUF rather than reading it into a buffer,
    # and mapped file pages land in "inactive". So once a model is loaded, a
    # large part of its weights is counted here as free, which is how the page
    # came to claim 2.75 GiB free with a 4.35 GiB model resident. The caller
    # subtracts the loaded model back out; see hardware.detect(resident_bytes).
    free = 0
    out = _run(["vm_stat"])
    if out:
        page = 16384
        m = re.search(r"page size of (\d+) bytes", out)
        if m:
            page = int(m.group(1))
        counts = dict(re.findall(r"Pages ([a-z ]+):\s+(\d+)", out))
        for key in ("free", "inactive", "speculative"):
            if key in counts:
                free += int(counts[key]) * page
    free = min(free, addressable)

    return Hardware(
        backend="metal",
        device_name=chip,
        total_bytes=addressable,
        free_bytes=free,
        unified=True,
        note="Unified memory: the GPU draws from system RAM, so other apps "
             "compete for the same pool.",
    )


def _cpu_name() -> str:
    """A readable processor name.

    platform.processor() returns something useful on macOS and often
    something unreadable elsewhere: on Windows it gives strings like
    "ARMv8 (64-bit) Family 8 Model 0 Revision 0, QEMU". Prefer the
    platform-specific sources and fall back only when they fail.
    """
    sysname = platform.system()

    if sysname == "Windows":
        name = os.environ.get("PROCESSOR_IDENTIFIER", "")
        # Strip the family/model/stepping tail, which is noise to a user.
        name = re.sub(r"\s*(Family|Model|Stepping).*$", "", name).strip()
        if name:
            return name

    if sysname == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass

    if sysname == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()

    return platform.processor() or platform.machine() or "unknown CPU"


def _memory_bytes() -> tuple[int, int]:
    """(total, available) system memory, per platform."""
    sysname = platform.system()

    if sysname == "Windows":
        # Windows has no /proc. GlobalMemoryStatusEx is the documented call
        # and needs no third-party package.
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        try:
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys), int(st.ullAvailPhys)
        except Exception:
            return 0, 0

    if sysname == "Linux":
        total = avail = 0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
        except OSError:
            pass
        return total, (avail or int(total * 0.5))

    if sysname == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        total = int(out.strip()) if out else 0
        return total, int(total * 0.5)

    return 0, 0


def _detect_cpu() -> Hardware:
    total, free = _memory_bytes()

    return Hardware(
        backend="cpu",
        device_name=_cpu_name(),
        total_bytes=total,
        free_bytes=free,
        unified=True,
        note="No supported GPU found. NF4DQ has no SIMD CPU kernel, so "
             "generation will be very slow. CIU needs an NVIDIA GPU on "
             "Windows or Linux, or Apple Silicon on a Mac.",
    )


def detect(resident_bytes: int = 0) -> Hardware:
    """Return the best backend available, preferring GPU over CPU.

    resident_bytes is memory a model already holds. On unified memory it must
    be subtracted: llama.cpp maps the weights, mapped pages count as inactive,
    and inactive counts as free, so without this the model's own weights are
    reported as available to load another model into.

    CUDA reports device memory directly and already excludes what is in use,
    so the subtraction only applies to unified memory.
    """
    hw = _detect_cuda() or _detect_metal() or _detect_cpu()
    if resident_bytes and hw.unified:
        hw.free_bytes = max(hw.free_bytes - resident_bytes, 0)
    return hw


if __name__ == "__main__":
    hw = detect()
    print(f"backend : {hw.backend}")
    print(f"device  : {hw.device_name}")
    print(f"memory  : {hw.free_gib:.2f} free of {hw.total_gib:.2f} GiB")
    if hw.note:
        print(f"note    : {hw.note}")
