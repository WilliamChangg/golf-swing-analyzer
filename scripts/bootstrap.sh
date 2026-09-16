#!/usr/bin/env bash
#
# One-command development setup. Idempotent: safe to re-run.
#
# Verifies or installs the toolchain, then syncs dependencies and fetches the
# required ML models. Does not modify your shell profile.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# Tools installed by rustup/uv land here but are not on a non-interactive PATH.
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

bold "1. Checking prerequisites"

command -v node >/dev/null || die "Node.js not found. Install Node >= 20.19 (https://nodejs.org)."
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
[ "$NODE_MAJOR" -ge 20 ] || die "Node $(node -v) is too old; this project needs >= 20.19."
ok "node $(node -v)"

command -v ffmpeg >/dev/null || die "ffmpeg not found. Install with: brew install ffmpeg"
ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"

if ! command -v uv >/dev/null; then
  warn "uv not found; installing with Homebrew"
  command -v brew >/dev/null || die "Homebrew not found. Install uv manually: https://docs.astral.sh/uv/"
  brew install uv
fi
ok "uv $(uv --version | awk '{print $2}')"

if ! command -v cargo >/dev/null; then
  warn "Rust not found; installing with rustup"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile default --no-modify-path
  export PATH="$HOME/.cargo/bin:$PATH"
fi
ok "cargo $(cargo --version | awk '{print $2}')"

if [ "$(uname -s)" = "Darwin" ]; then
  xcode-select -p >/dev/null 2>&1 || die "Xcode Command Line Tools missing. Run: xcode-select --install"
  ok "Xcode CLT at $(xcode-select -p)"
fi

bold "2. Installing Node dependencies"
npm install
ok "npm workspaces installed"

bold "3. Creating the Python environment"
# --project keeps this working regardless of the caller's cwd.
uv sync --project python
ok "python 3.12 environment synced"

bold "4. Fetching ML models"
uv run --project python python scripts/download_models.py
ok "models present and verified"

bold "5. Verifying the environment"
uv run --project python analyzer doctor

cat <<'EOF'

Setup complete. Useful commands:

  npm run dev          launch the desktop app
  npm run doctor       environment health check (terminal)
  npm run check:all    every lint, typecheck and test suite

If `cargo` is not on your PATH in new shells, add this to ~/.zshrc:

  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

EOF
