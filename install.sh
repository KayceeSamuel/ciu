#!/bin/sh
# CIU installer.
#
#   curl -fsSL https://raw.githubusercontent.com/KayceeSamuel/ciu/main/install.sh | sh
#
# Installs into ~/.ciu and touches nothing else. Python packages go in a
# private virtualenv so this cannot break a system Python or a conda setup.
# To remove everything: rm -rf ~/.ciu

set -eu

CIU_REPO="KayceeSamuel/ciu"
FORK_REPO="KayceeSamuel/llama.cpp"
PREFIX="${CIU_HOME:-$HOME/.ciu}"

RED=''; BOLD=''; DIM=''; OFF=''
if [ -t 1 ]; then
  RED=$(printf '\033[31m'); BOLD=$(printf '\033[1m')
  DIM=$(printf '\033[2m');  OFF=$(printf '\033[0m')
fi

say()  { printf '%s\n' "$*"; }
step() { printf '%s%s%s\n' "$BOLD" "$*" "$OFF"; }
dim()  { printf '%s%s%s\n' "$DIM" "$*" "$OFF"; }
die()  { printf '%s%s%s\n' "$RED" "$*" "$OFF" >&2; exit 1; }

# ---------------------------------------------------------------- platform

OS=$(uname -s)
ARCH=$(uname -m)

case "$OS" in
  Darwin)
    [ "$ARCH" = "arm64" ] || die "CIU needs an Apple Silicon Mac. This is $ARCH."
    ASSET="llama-server-macos-arm64.tar.gz"
    BACKEND="Metal"
    ;;
  Linux)
    [ "$ARCH" = "x86_64" ] || die "CIU needs x86_64 on Linux. This is $ARCH."
    if command -v nvidia-smi >/dev/null 2>&1; then
      ASSET="llama-server-linux-x64-cuda.tar.gz"
      BACKEND="CUDA"
    else
      ASSET="llama-server-linux-x64-cpu.tar.gz"
      BACKEND="CPU"
      say ""
      say "${BOLD}No NVIDIA GPU found.${OFF}"
      say "CIU has CUDA and Metal kernels but no Vulkan kernel yet, so AMD and"
      say "Intel GPUs fall back to the CPU. That works but generates at well"
      say "under one token a second, which is not usable for real work."
      say ""
      printf 'Install anyway? [y/N] '
      read -r reply </dev/tty || reply=n
      case "$reply" in y|Y) ;; *) die "Stopped." ;; esac
    fi
    ;;
  *)
    die "CIU supports macOS and Linux. Windows is not packaged yet."
    ;;
esac

command -v curl >/dev/null 2>&1 || die "curl is required."

PY=""
for c in python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  fi
done
[ -n "$PY" ] || die "Python 3.10 or newer is required. Install it and run this again."

say ""
step "Installing CIU"
dim  "$OS $ARCH, $BACKEND backend, into $PREFIX"
say ""

mkdir -p "$PREFIX/bin" "$PREFIX/models"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT INT TERM

# ------------------------------------------------------------ llama-server
# The NF4DQ fork, prebuilt. Building it from source takes about twenty
# minutes and a working compiler toolchain, which is the step this exists to
# remove.

step "1/3  Fetching the inference engine"
URL="https://github.com/$FORK_REPO/releases/latest/download/$ASSET"
if ! curl -fsSL "$URL" -o "$TMP/engine.tar.gz"; then
  say ""
  say "Could not download $ASSET."
  say "Check https://github.com/$FORK_REPO/releases for what is published."
  die "Stopped."
fi
tar -xzf "$TMP/engine.tar.gz" -C "$PREFIX/bin"
chmod +x "$PREFIX/bin/llama-server" 2>/dev/null || true
[ -x "$PREFIX/bin/llama-server" ] || die "The download did not contain llama-server."

# macOS quarantines anything fetched from the internet and will refuse to run
# it. Clearing the attribute here saves the user a security dialog they have
# no way to interpret.
if [ "$OS" = "Darwin" ]; then
  xattr -dr com.apple.quarantine "$PREFIX/bin" 2>/dev/null || true
fi

# -------------------------------------------------------------------- CIU

step "2/3  Fetching CIU"
curl -fsSL "https://github.com/$CIU_REPO/archive/refs/heads/main.tar.gz" \
  -o "$TMP/ciu.tar.gz" || die "Could not download CIU."
rm -rf "$PREFIX/app"
mkdir -p "$PREFIX/app"
tar -xzf "$TMP/ciu.tar.gz" -C "$PREFIX/app" --strip-components=1

# ---------------------------------------------------------------- packages
# A private virtualenv. Installing into the user's own Python is how you break
# someone's conda environment and get blamed for it.

step "3/3  Installing Python packages"
"$PY" -m venv "$PREFIX/venv" >/dev/null 2>&1 \
  || die "Could not create a virtualenv. On Debian or Ubuntu: apt install python3-venv"
"$PREFIX/venv/bin/pip" install --quiet --upgrade pip >/dev/null 2>&1 || true
"$PREFIX/venv/bin/pip" install --quiet \
  fastapi uvicorn httpx huggingface_hub gguf \
  || die "Could not install Python packages."

# --------------------------------------------------------------- launcher

cat > "$PREFIX/bin/ciu" <<EOF
#!/bin/sh
# Starts CIU and opens the page.
set -e
PREFIX="$PREFIX"
export CIU_LLAMA_SERVER="\$PREFIX/bin/llama-server"
cd "\$PREFIX/app"

URL="http://127.0.0.1:8674"
(
  sleep 2
  if command -v open >/dev/null 2>&1; then open "\$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "\$URL"
  fi
) >/dev/null 2>&1 &

exec "\$PREFIX/venv/bin/python" run.py
EOF
chmod +x "$PREFIX/bin/ciu"

# Put it on PATH if we can do so without surprising anyone.
LINKED=""
for d in /usr/local/bin "$HOME/.local/bin"; do
  if [ -d "$d" ] && [ -w "$d" ]; then
    ln -sf "$PREFIX/bin/ciu" "$d/ciu" && LINKED="$d"
    break
  fi
done

say ""
step "Done."
say ""
if [ -n "$LINKED" ] && command -v ciu >/dev/null 2>&1; then
  say "  Start it with:   ${BOLD}ciu${OFF}"
else
  say "  Start it with:   ${BOLD}$PREFIX/bin/ciu${OFF}"
  if [ -n "$LINKED" ]; then
    dim "  ($LINKED is not on your PATH yet; a new terminal will pick it up.)"
  fi
fi
say ""
say "  Your browser opens at http://127.0.0.1:8674, where you pick a model"
say "  that fits your machine. Everything runs locally."
say ""
dim  "  To remove: rm -rf $PREFIX"
say ""
