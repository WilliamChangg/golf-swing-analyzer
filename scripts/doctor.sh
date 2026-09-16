#!/usr/bin/env bash
#
# Environment health check without launching the desktop app.
# Exits non-zero when a required component is missing or errored.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
exec uv run --project "$REPO_ROOT/python" analyzer doctor "$@"
